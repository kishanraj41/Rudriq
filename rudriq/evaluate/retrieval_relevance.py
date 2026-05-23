"""Retrieval relevance evaluator.

For each LLM call linked to upstream retrieved documents, scores how
semantically relevant those documents are to the LLM's input/prompt.

Mechanism: embed the LLM input and each linked upstream document
string, compute cosine similarity, report the mean and min relevance.
Low relevance suggests the retrieval step pulled documents that don't
match the query — a common root cause of RAG failures.

Degrades gracefully when fastembed is unavailable: returns DEGRADED
``EvalResult`` per LLM node rather than crashing.
"""

from __future__ import annotations

from typing import Any

from rudriq.core.schema import EdgeKind, TraceGraph, TraceNode
from rudriq.evaluate.base import EvalResult, EvalStatus
from rudriq.evaluate.embeddings import cosine_similarity, embed_texts


class RetrievalRelevanceEvaluator:
    metric = "retrieval_relevance"

    def __init__(self, low_relevance_threshold: float = 0.3) -> None:
        self.low_relevance_threshold = low_relevance_threshold

    def evaluate(self, graph: TraceGraph) -> list[EvalResult]:
        results: list[EvalResult] = []

        nodes_by_id = {n.node_id: n for n in graph.nodes}
        lineage_parents: dict[str, list[str]] = {}
        for e in graph.edges:
            if e.kind == EdgeKind.LINEAGE_LINK:
                lineage_parents.setdefault(e.child_id, []).append(e.parent_id)

        llm_nodes = [
            n for n in graph.nodes
            if n.kind.value.startswith("llm_") and n.node_id in lineage_parents
        ]

        if not llm_nodes:
            return [
                EvalResult(
                    metric=self.metric,
                    status=EvalStatus.SKIPPED,
                    score=None,
                    explanation="No LLM calls with linked upstream context to evaluate.",
                )
            ]

        for llm in llm_nodes:
            query_text = self._extract_query_text(llm)
            if not query_text:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                        explanation="LLM node has no extractable input text.",
                        node_id=llm.node_id,
                    )
                )
                continue

            doc_texts = self._extract_upstream_texts(
                lineage_parents[llm.node_id], nodes_by_id
            )
            if not doc_texts:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                        explanation="Linked upstream nodes have no extractable text.",
                        node_id=llm.node_id,
                    )
                )
                continue

            all_texts = [query_text] + doc_texts
            embeddings = embed_texts(all_texts)
            if embeddings is None:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.DEGRADED, score=None,
                        explanation=(
                            "fastembed not installed; cannot compute semantic "
                            "relevance. Install with: pip install 'rudriq[evaluate]'"
                        ),
                        node_id=llm.node_id,
                    )
                )
                continue

            query_vec = embeddings[0]
            doc_vecs = embeddings[1:]
            sims = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
            mean_sim = sum(sims) / len(sims)
            min_sim = min(sims)

            # Normalize cosine [-1, 1] to score [0, 1].
            score = max(0.0, (mean_sim + 1) / 2)

            low_count = sum(1 for s in sims if s < self.low_relevance_threshold)
            explanation = (
                f"Mean relevance {mean_sim:.2f}, min {min_sim:.2f} across "
                f"{len(doc_texts)} retrieved docs."
            )
            if low_count > 0:
                explanation += (
                    f" {low_count} doc(s) below relevance threshold "
                    f"{self.low_relevance_threshold} — possible retrieval issue."
                )

            results.append(
                EvalResult(
                    metric=self.metric,
                    status=EvalStatus.OK,
                    score=score,
                    explanation=explanation,
                    node_id=llm.node_id,
                    details={
                        "mean_similarity": mean_sim,
                        "min_similarity": min_sim,
                        "doc_count": len(doc_texts),
                        "low_relevance_count": low_count,
                    },
                )
            )

        return results

    def _extract_query_text(self, llm_node: TraceNode) -> str:
        """Pull the LLM input text from node metadata.

        Looks for common metadata keys where OpenLLMetry / our adapter
        stash the prompt or input. Falls back to empty string. Order
        matters: ``gen_ai.prompt`` is the OTel-semantic-conventions
        name; the rest are RudriQ/AutoLineage conventions.
        """
        md = llm_node.metadata or {}
        for key in ("gen_ai.prompt", "input_preview", "prompt", "input"):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _extract_upstream_texts(
        self,
        parent_ids: list[str],
        nodes_by_id: dict[str, TraceNode],
    ) -> list[str]:
        """Pull text content from linked upstream nodes."""
        texts: list[str] = []
        for pid in parent_ids:
            node = nodes_by_id.get(pid)
            if node is None:
                continue
            md = node.metadata or {}
            for key in ("output_preview", "content", "text", "value"):
                val = md.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append(val)
                    break
        return texts
