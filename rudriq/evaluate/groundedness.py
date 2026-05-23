"""Groundedness (hallucination) evaluator.

For each LLM call with linked upstream context, scores how well the
response is grounded in the retrieved documents. Low groundedness
suggests the model is making claims not supported by its context —
the classic hallucination signature.

Mechanism: split the response into sentences, embed each, and for each
response sentence compute its max similarity to any retrieved document.
A response sentence with low max-similarity to all context is
"ungrounded." ``score = fraction of grounded sentences``.

Caveat (honest, surfaced in explanations): semantic similarity is a
*proxy* for entailment. This catches gross hallucination (response
about topic X when context is about Y) reliably; subtle factual errors
within the right topic are harder and a different class of evaluator
(NLI-based) would address them. We document this in the result so
downstream consumers don't over-trust the score.
"""

from __future__ import annotations

import re
from typing import Any

from rudriq.core.schema import EdgeKind, TraceGraph, TraceNode
from rudriq.evaluate.base import EvalResult, EvalStatus
from rudriq.evaluate.embeddings import cosine_similarity, embed_texts


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for the groundedness heuristic.

    Splits on sentence-ending punctuation followed by whitespace. Drops
    fragments shorter than 11 characters so we don't score noise like
    "Yes." or "Hi.".
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


class GroundednessEvaluator:
    metric = "groundedness"

    def __init__(self, grounded_threshold: float = 0.35) -> None:
        self.grounded_threshold = grounded_threshold

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

            response_sentences = _split_sentences(response_text)
            if not response_sentences:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                        explanation="Response too short to evaluate groundedness.",
                        node_id=llm.node_id,
                    )
                )
                continue

            all_texts = response_sentences + doc_texts
            embeddings = embed_texts(all_texts)
            if embeddings is None:
                results.append(
                    EvalResult(
                        metric=self.metric, status=EvalStatus.DEGRADED, score=None,
                        explanation=(
                            "fastembed not installed; cannot compute groundedness. "
                            "Install with: pip install 'rudriq[evaluate]'"
                        ),
                        node_id=llm.node_id,
                    )
                )
                continue

            n_sent = len(response_sentences)
            sent_vecs = embeddings[:n_sent]
            doc_vecs = embeddings[n_sent:]

            grounded_count = 0
            ungrounded_examples: list[str] = []
            for i, sv in enumerate(sent_vecs):
                max_sim = max(cosine_similarity(sv, dv) for dv in doc_vecs)
                if max_sim >= self.grounded_threshold:
                    grounded_count += 1
                elif len(ungrounded_examples) < 3:
                    snippet = response_sentences[i][:80]
                    ungrounded_examples.append(f"'{snippet}' (max sim {max_sim:.2f})")

            score = grounded_count / n_sent
            explanation = (
                f"{grounded_count}/{n_sent} response sentences grounded in "
                f"retrieved context (threshold {self.grounded_threshold})."
            )
            if ungrounded_examples:
                explanation += " Ungrounded: " + "; ".join(ungrounded_examples)
            explanation += (
                " Note: semantic-similarity heuristic; catches topic-level "
                "hallucination, not subtle factual errors."
            )

            results.append(
                EvalResult(
                    metric=self.metric,
                    status=EvalStatus.OK,
                    score=score,
                    explanation=explanation,
                    node_id=llm.node_id,
                    details={
                        "grounded_sentences": grounded_count,
                        "total_sentences": n_sent,
                        "ungrounded_examples": ungrounded_examples,
                    },
                )
            )

        return results

    def _extract_response_text(self, llm_node: TraceNode) -> str:
        md = llm_node.metadata or {}
        # rudriq.completion_preview is what our adapter stores when
        # content capture is enabled (Day 14).
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
