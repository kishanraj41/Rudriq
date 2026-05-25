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

import re
from itertools import combinations
from typing import Any

from rudriq.core.schema import EdgeKind, TraceGraph, TraceNode
from rudriq.evaluate.base import EvalResult, EvalStatus
from rudriq.evaluate.embeddings import cosine_similarity, embed_texts


# Common delimiters separating retrieval context from the actual question
# in RAG prompt templates. Ordered by specificity (more specific patterns
# first so a generic "Ask:" doesn't pre-empt "User question:" if both
# happen to appear).
_QUESTION_MARKERS = (
    r"(?i)\buser\s+question\s*:\s*",
    r"(?i)\bquestion\s*:\s*",
    r"(?i)\bquery\s*:\s*",
    r"(?i)\bask\s*:\s*",
)


def _extract_question_from_user_message(
    user_msg: str,
    linked_context: list[str] | None = None,
) -> str:
    """Isolate the actual question from a user turn that may also embed
    retrieval context.

    Day-17 found that role-separation alone doesn't help if the
    pipeline glues ``Context:...Question:...`` into a single user
    turn (a common RAG template pattern). The user-message embedding
    is then dominated by the context — two different questions sharing
    similar retrieval collapse into one consistency group, producing
    a false 1.0 instead of an honest SKIP.

    Strategy, each step degrading safely to the next:

    1. **Marker extraction.** Look for the LAST occurrence of a
       ``Question:`` / ``Query:`` / ``Ask:`` style delimiter; return
       what follows. Using the LAST match (not the first) is
       deliberate — RAG templates put the question after the
       retrieved context, and the context itself can contain the
       same delimiter verbatim ("Doc says: Question: ...").
    2. **Context subtraction.** If linked upstream document strings
       are known, remove them verbatim from the message. Only adopt
       the cleaned result if it removed something substantial AND
       left a non-trivial remainder — this guards against
       over-zealous subtraction producing empty strings.
    3. **Fallback.** Return ``user_msg`` unchanged. Never worse than
       before the fix; the caller's grouping behavior is unaffected.
    """
    if not user_msg:
        return user_msg

    # 1. Marker extraction
    for pattern in _QUESTION_MARKERS:
        matches = list(re.finditer(pattern, user_msg))
        if matches:
            tail = user_msg[matches[-1].end():].strip()
            if tail:
                return tail

    # 2. Context subtraction
    if linked_context:
        cleaned = user_msg
        for doc in linked_context:
            if isinstance(doc, str) and len(doc) >= 20 and doc in cleaned:
                cleaned = cleaned.replace(doc, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Only adopt if substantial reduction AND non-trivial remainder.
        if (
            cleaned
            and len(cleaned) < len(user_msg) * 0.8
            and len(cleaned) >= 10
        ):
            return cleaned

    # 3. Fallback — never make things worse than before the fix.
    return user_msg


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
        nodes_by_id = {n.node_id: n for n in graph.nodes}
        # Build child -> [parent_id] for LINEAGE_LINK edges. The
        # question extractor uses the parents' content_preview text
        # for context subtraction (step 2 of its three-stage fallback).
        lineage_parents: dict[str, list[str]] = {}
        for e in graph.edges:
            if e.kind == EdgeKind.LINEAGE_LINK:
                lineage_parents.setdefault(e.child_id, []).append(e.parent_id)

        llm_nodes = [n for n in graph.nodes if n.kind.value.startswith("llm_")]

        records: list[tuple[TraceNode, str, str]] = []
        for n in llm_nodes:
            linked_context = self._collect_linked_context(
                lineage_parents.get(n.node_id, []), nodes_by_id,
            )
            prompt = self._extract_grouping_key(n, linked_context)
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

    def _extract_grouping_key(
        self,
        node: TraceNode,
        linked_context: list[str] | None = None,
    ) -> str:
        """The key consistency groups by: the isolated *question*.

        Successive Day-17 / Day-1-of-9 refinements layered on top of
        each other:

        * Day 15 (original): group by ``rudriq.prompt_preview`` → all
          RAG calls collapsed because retrieval dominated the embedding.
        * Day 17 Thread B: prefer ``rudriq.user_message_preview`` to
          drop the system-prompt prefix. Helps for retrieval-in-system
          templates; doesn't help when retrieval lives inside the user
          turn.
        * Day 1/9 (this): when the user turn itself contains
          ``Context:...Question:...``, isolate the question via marker
          extraction or context subtraction. Safe fallback to the full
          user message preserves the prior behavior — never worse.
        """
        md = node.metadata or {}
        user_msg = ""
        for key in (
            "rudriq.user_message_preview",
            "rudriq.prompt_preview", "gen_ai.prompt", "prompt", "input",
        ):
            val = md.get(key)
            if isinstance(val, str) and val.strip():
                user_msg = val
                break
        if not user_msg:
            return ""
        return _extract_question_from_user_message(user_msg, linked_context)

    def _collect_linked_context(
        self,
        parent_ids: list[str],
        nodes_by_id: dict[str, TraceNode],
    ) -> list[str]:
        """Gather linked upstream document text for context subtraction.

        Pulls each linked parent's preview text in the same priority
        order the other content-aware evaluators use. Returns the raw
        strings — the extractor decides whether/how to subtract.
        """
        texts: list[str] = []
        for pid in parent_ids:
            parent = nodes_by_id.get(pid)
            if parent is None:
                continue
            md = parent.metadata or {}
            for key in (
                "rudriq.content_preview",
                "output_preview", "content", "text", "value",
            ):
                v = md.get(key)
                if isinstance(v, str) and v.strip():
                    texts.append(v)
                    break
        return texts

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
