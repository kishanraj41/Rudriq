"""Coherence evaluator.

Scores whether an LLM response engages with its retrieved context, as
distinct from groundedness (are claims supported?). Coherence asks:
*does the response USE the context at all?*

Mechanism: compute the similarity between the whole response and the
whole concatenated context. High response-to-context similarity means
the response is "about" the same things as the retrieved docs. Low
similarity with linked context present suggests the model ignored
retrieval — a RAG anti-pattern where retrieval is decorative.

Contrast with the groundedness evaluator (per-sentence support check):
groundedness is a fine-grained "is each claim entailed?" signal;
coherence is a single holistic "is the response talking about the
right things?" signal. They disagree productively: high groundedness
+ low coherence means the model answered from its parametric knowledge
and happens to have produced text whose individual claims pattern-match
context — i.e., retrieval was decorative even though every claim looks
supported.

Degrades gracefully when fastembed is unavailable.
"""

from __future__ import annotations

from rudriq.core.schema import EdgeKind, TraceGraph, TraceNode
from rudriq.evaluate.base import EvalResult, EvalStatus
from rudriq.evaluate.embeddings import cosine_similarity, embed_texts


class CoherenceEvaluator:
    metric = "coherence"

    def __init__(self, low_coherence_threshold: float = 0.25) -> None:
        self.low_coherence_threshold = low_coherence_threshold

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
                    metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                    explanation="No LLM calls with linked upstream context to evaluate.",
                )
            ]

        for llm in llm_nodes:
            response_text = self._extract_response_text(llm)
            if not response_text:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                        explanation="LLM node has no extractable response text.",
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

            # Embed whole response and whole concatenated context as a pair.
            context_blob = "\n".join(doc_texts)
            embeddings = embed_texts([response_text, context_blob])
            if embeddings is None:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.DEGRADED, score=None,
                        explanation=(
                            "fastembed not installed; cannot compute coherence. "
                            "Install with: pip install 'rudriq[evaluate]'"
                        ),
                        node_id=llm.node_id,
                    )
                )
                continue

            sim = cosine_similarity(embeddings[0], embeddings[1])
            # Normalize cosine [-1, 1] to score [0, 1].
            score = max(0.0, (sim + 1) / 2)

            explanation = f"Response-to-context coherence {sim:.2f}."
            if sim < self.low_coherence_threshold:
                explanation += (
                    " Low coherence with linked context — the response may "
                    "be ignoring retrieval (retrieval-decorative anti-pattern)."
                )

            results.append(
                EvalResult(
                    metric=self.metric,
                    status=EvalStatus.OK,
                    score=score,
                    explanation=explanation,
                    node_id=llm.node_id,
                    details={"coherence_similarity": sim},
                )
            )
        return results

    def _extract_response_text(self, llm_node: TraceNode) -> str:
        md = llm_node.metadata or {}
        # rudriq.completion_preview is what our adapter stores when
        # content capture is enabled (Day 14). Others are legacy keys.
        for key in (
            "rudriq.completion_preview", "gen_ai.completion",
            "response_preview", "completion", "response", "output",
        ):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _extract_upstream_texts(
        self,
        parent_ids: list[str],
        nodes_by_id: dict[str, TraceNode],
    ) -> list[str]:
        texts: list[str] = []
        for pid in parent_ids:
            node = nodes_by_id.get(pid)
            if node is None:
                continue
            md = node.metadata or {}
            for key in (
                "rudriq.content_preview",
                "output_preview", "content", "text", "value",
            ):
                val = md.get(key)
                if isinstance(val, str) and val.strip():
                    texts.append(val)
                    break
        return texts
