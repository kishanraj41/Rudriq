"""Consistency evaluator (within-run).

Scores response stability across LLM calls that received the same or
near-identical prompt within a single run. High response variance for
a repeated prompt is an instability signal: the system would not give
the same answer twice. This matters for audit — a regulator may ask
whether an AI decision is reproducible.

Mechanism:

1. Group LLM nodes by prompt similarity (prompts whose pairwise cosine
   similarity exceeds ``group_threshold`` land in the same group).
2. Within each group of size ≥ 2, embed all responses and compute the
   mean pairwise response similarity. High mean = consistent.
3. Singleton groups (unique prompts) are skipped — nothing to compare.

This is a *within-run* evaluator: it needs a run that repeats prompts
(agentic loops, retries, batch-similar items). Runs with all-distinct
prompts will report SKIPPED for every group, which is correct and
honest behavior, not a failure.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

from rudriq.core.schema import TraceGraph, TraceNode
from rudriq.evaluate.base import EvalResult, EvalStatus
from rudriq.evaluate.embeddings import cosine_similarity, embed_texts


class ConsistencyEvaluator:
    metric = "consistency"

    def __init__(
        self,
        group_threshold: float = 0.85,
        low_consistency_threshold: float = 0.7,
    ) -> None:
        # Prompts whose cosine similarity exceeds this land in the same group.
        self.group_threshold = group_threshold
        self.low_consistency_threshold = low_consistency_threshold

    def evaluate(self, graph: TraceGraph) -> list[EvalResult]:
        llm_nodes = [n for n in graph.nodes if n.kind.value.startswith("llm_")]

        records: list[tuple[TraceNode, str, str]] = []
        for n in llm_nodes:
            prompt = self._extract_prompt(n)
            response = self._extract_response(n)
            if prompt and response:
                records.append((n, prompt, response))

        if len(records) < 2:
            return [
                EvalResult(
                    metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                    explanation=(
                        "Need at least 2 LLM calls with prompt+response to "
                        "assess consistency; this run has fewer."
                    ),
                )
            ]

        # Embed all prompts in one batch so we can group them by similarity.
        prompts = [r[1] for r in records]
        prompt_embeddings = embed_texts(prompts)
        if prompt_embeddings is None:
            return [
                EvalResult(
                    metric=self.metric, status=EvalStatus.DEGRADED, score=None,
                    explanation=(
                        "fastembed not installed; cannot group prompts. "
                        "Install with: pip install 'rudriq[evaluate]'"
                    ),
                )
            ]

        groups = self._group_by_similarity(prompt_embeddings)

        results: list[EvalResult] = []
        evaluated_any_group = False

        for group_indices in groups:
            if len(group_indices) < 2:
                continue  # singleton — nothing to compare
            evaluated_any_group = True

            responses = [records[i][2] for i in group_indices]
            response_embeddings = embed_texts(responses)
            if response_embeddings is None:
                continue

            # Mean pairwise response similarity within the group.
            pair_sims = [
                cosine_similarity(
                    response_embeddings[a], response_embeddings[b]
                )
                for a, b in combinations(range(len(response_embeddings)), 2)
            ]
            mean_sim = sum(pair_sims) / len(pair_sims)
            score = max(0.0, (mean_sim + 1) / 2)

            group_node_ids = [records[i][0].node_id for i in group_indices]
            explanation = (
                f"{len(group_indices)} LLM calls with near-identical prompts; "
                f"mean pairwise response similarity {mean_sim:.2f}."
            )
            if mean_sim < self.low_consistency_threshold:
                explanation += (
                    " Low consistency — the same query produced divergent "
                    "responses (instability / non-reproducibility signal)."
                )

            results.append(
                EvalResult(
                    metric=self.metric,
                    status=EvalStatus.OK,
                    score=score,
                    explanation=explanation,
                    node_id=None,  # group-level result, not single-node
                    details={
                        "group_size": len(group_indices),
                        "group_node_ids": group_node_ids,
                        "mean_response_similarity": mean_sim,
                    },
                )
            )

        if not evaluated_any_group:
            return [
                EvalResult(
                    metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                    explanation=(
                        "No repeated prompts found; all LLM calls had distinct "
                        "inputs. Consistency requires repeated queries "
                        "(agentic loops, retries, batch-similar items)."
                    ),
                )
            ]

        return results

    def _group_by_similarity(self, embeddings: list[Any]) -> list[list[int]]:
        """Greedy single-pass clustering.

        Each item joins the first group whose representative it exceeds
        ``group_threshold`` with, else starts a new group. O(n·groups)
        — fine for the LLM-call counts we see (dozens to low hundreds
        per run). For thousand-plus, swap in a real clusterer.
        """
        groups: list[list[int]] = []
        reps: list[Any] = []  # representative embedding per group (first member)
        for i, emb in enumerate(embeddings):
            placed = False
            for g_idx, rep in enumerate(reps):
                if cosine_similarity(emb, rep) >= self.group_threshold:
                    groups[g_idx].append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])
                reps.append(emb)
        return groups

    def _extract_prompt(self, node: TraceNode) -> str:
        md = node.metadata or {}
        for key in (
            "rudriq.prompt_preview", "gen_ai.prompt", "prompt", "input",
        ):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    def _extract_response(self, node: TraceNode) -> str:
        md = node.metadata or {}
        for key in (
            "rudriq.completion_preview", "gen_ai.completion",
            "completion", "response", "output",
        ):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""
