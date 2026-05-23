"""Drift evaluator (cross-run, baseline-relative).

Compares a current run against a baseline run to detect behavioral
shift. Drift is inherently temporal: it asks whether the system behaves
differently *now* than it did at baseline. A single run has no baseline,
so this evaluator requires a ``baseline_graph`` and SKIPS if none is
given.

Two drift signals are computed:

1. **Structural drift** — has the operation mix changed? Counter of
   ``library.operation`` strings, compared by normalized L1 distance.
   Catches "the pipeline emits twice as many LLM calls now" or
   "embeddings step disappeared." Reported as ``drift_structural``.

2. **Response drift** — for LLM calls aligned across runs by prompt
   similarity, how much have the responses changed? High response
   change for the same prompt = behavioral drift. Reported as
   ``drift_response``.

Node alignment across runs uses prompt-similarity matching, since
``node_id`` differs between runs (each run generates fresh UUIDs).
Unaligned current-run nodes (new operations with no baseline
counterpart) are surfaced separately as "new behavior."
"""

from __future__ import annotations

from collections import Counter

from rudriq.core.schema import TraceGraph, TraceNode
from rudriq.evaluate.base import EvalResult, EvalStatus
from rudriq.evaluate.embeddings import cosine_similarity, embed_texts


class DriftEvaluator:
    metric = "drift"

    def __init__(
        self,
        baseline_graph: TraceGraph | None = None,
        alignment_threshold: float = 0.85,
        high_drift_threshold: float = 0.4,
    ) -> None:
        # Injected by the CLI after loading ``--baseline-run-id``. The
        # framework's evaluate(graph) signature can't carry it; that's
        # why drift is special-cased in cli._cmd_evaluate.
        self.baseline_graph = baseline_graph
        self.alignment_threshold = alignment_threshold
        self.high_drift_threshold = high_drift_threshold

    def evaluate(self, graph: TraceGraph) -> list[EvalResult]:
        if self.baseline_graph is None:
            return [
                EvalResult(
                    metric=self.metric, status=EvalStatus.SKIPPED, score=None,
                    explanation=(
                        "Drift requires a baseline run. Pass --baseline-run-id "
                        "to compare against. Drift is cross-run by nature; a "
                        "single run has no baseline to drift from."
                    ),
                )
            ]

        results: list[EvalResult] = []
        results.append(self._structural_drift(graph))
        results.extend(self._response_drift(graph))
        return results

    # -----------------------------------------------------------------
    # Structural drift: operation-mix shift
    # -----------------------------------------------------------------

    def _structural_drift(self, current: TraceGraph) -> EvalResult:
        def op_mix(g: TraceGraph) -> Counter[str]:
            return Counter(f"{n.library}.{n.operation}" for n in g.nodes)

        assert self.baseline_graph is not None  # narrowed by caller
        base_mix = op_mix(self.baseline_graph)
        cur_mix = op_mix(current)
        all_ops = set(base_mix) | set(cur_mix)

        base_total = sum(base_mix.values()) or 1
        cur_total = sum(cur_mix.values()) or 1
        # L1 distance between the two op-fraction distributions. Range
        # [0, 2]; we normalize to [0, 1] so the score reads naturally.
        l1 = sum(
            abs(base_mix[op] / base_total - cur_mix[op] / cur_total)
            for op in all_ops
        )
        drift_magnitude = l1 / 2
        score = 1.0 - drift_magnitude  # 1.0 = identical mix

        changed: list[str] = []
        for op in sorted(all_ops):
            b, c = base_mix[op], cur_mix[op]
            if b != c:
                changed.append(f"{op}: {b}→{c}")

        explanation = (
            f"Structural drift {drift_magnitude:.2f} (0=identical mix)."
        )
        if changed:
            explanation += " Changes: " + "; ".join(changed[:5])
            if len(changed) > 5:
                explanation += f" (+{len(changed) - 5} more)"

        return EvalResult(
            metric="drift_structural",
            status=EvalStatus.OK,
            score=score,
            explanation=explanation,
            details={"drift_magnitude": drift_magnitude, "changed_ops": changed},
        )

    # -----------------------------------------------------------------
    # Response drift: align by prompt similarity, compare responses
    # -----------------------------------------------------------------

    def _response_drift(self, current: TraceGraph) -> list[EvalResult]:
        cur_llms = self._llm_records(current)
        assert self.baseline_graph is not None
        base_llms = self._llm_records(self.baseline_graph)

        if not cur_llms or not base_llms:
            return [
                EvalResult(
                    metric="drift_response", status=EvalStatus.SKIPPED, score=None,
                    explanation=(
                        "Need LLM calls with prompt+response in both runs to "
                        "measure response drift."
                    ),
                )
            ]

        # Embed all prompts (current + baseline) in one batch so they
        # share the same model state and the cosine numbers are
        # directly comparable.
        cur_prompts = [r["prompt"] for r in cur_llms]
        base_prompts = [r["prompt"] for r in base_llms]
        all_prompt_emb = embed_texts(cur_prompts + base_prompts)
        if all_prompt_emb is None:
            return [
                EvalResult(
                    metric="drift_response", status=EvalStatus.DEGRADED, score=None,
                    explanation=(
                        "fastembed not installed; cannot align runs. "
                        "Install with: pip install 'rudriq[evaluate]'"
                    ),
                )
            ]

        n_cur = len(cur_prompts)
        cur_prompt_emb = all_prompt_emb[:n_cur]
        base_prompt_emb = all_prompt_emb[n_cur:]

        # Greedy alignment: each current call gets matched to its best
        # baseline prompt; pairs below ``alignment_threshold`` count as
        # unaligned ("new behavior"). Greedy keeps it O(n·m); a
        # Hungarian-style optimal matcher would be marginally better
        # for the regulator-comparison story but isn't worth the
        # complexity at typical LLM-call counts.
        aligned_pairs: list[tuple[int, int, float]] = []
        unaligned_cur: list[int] = []
        for ci, ce in enumerate(cur_prompt_emb):
            best_bi, best_sim = None, -1.0
            for bi, be in enumerate(base_prompt_emb):
                s = cosine_similarity(ce, be)
                if s > best_sim:
                    best_sim, best_bi = s, bi
            if best_bi is not None and best_sim >= self.alignment_threshold:
                aligned_pairs.append((ci, best_bi, best_sim))
            else:
                unaligned_cur.append(ci)

        if not aligned_pairs:
            return [
                EvalResult(
                    metric="drift_response", status=EvalStatus.OK, score=0.0,
                    explanation=(
                        f"No current LLM calls aligned to baseline (none "
                        f"exceeded prompt-similarity {self.alignment_threshold}). "
                        f"The workload appears entirely new vs baseline — "
                        f"maximal behavioral drift."
                    ),
                    details={"unaligned_current": len(unaligned_cur)},
                )
            ]

        cur_responses = [cur_llms[ci]["response"] for ci, _, _ in aligned_pairs]
        base_responses = [base_llms[bi]["response"] for _, bi, _ in aligned_pairs]
        resp_emb = embed_texts(cur_responses + base_responses)
        if resp_emb is None:
            return [
                EvalResult(
                    metric="drift_response", status=EvalStatus.DEGRADED, score=None,
                    explanation="fastembed not installed; cannot compare responses.",
                )
            ]

        n_pairs = len(aligned_pairs)
        cur_resp_emb = resp_emb[:n_pairs]
        base_resp_emb = resp_emb[n_pairs:]

        drifts: list[float] = []
        high_drift_examples: list[str] = []
        for i in range(n_pairs):
            sim = cosine_similarity(cur_resp_emb[i], base_resp_emb[i])
            # 1.0 - normalized_similarity → 0 = identical, 1 = opposite.
            drift = 1.0 - max(0.0, (sim + 1) / 2)
            drifts.append(drift)
            if drift > self.high_drift_threshold and len(high_drift_examples) < 3:
                ci = aligned_pairs[i][0]
                prompt_snip = cur_llms[ci]["prompt"][:60]
                high_drift_examples.append(f"'{prompt_snip}' drift {drift:.2f}")

        mean_drift = sum(drifts) / len(drifts)
        score = 1.0 - mean_drift  # 1.0 = no response drift

        explanation = (
            f"{n_pairs} LLM calls aligned to baseline; mean response drift "
            f"{mean_drift:.2f} (0=identical responses)."
        )
        if unaligned_cur:
            explanation += (
                f" {len(unaligned_cur)} current call(s) had no baseline match "
                f"(new behavior)."
            )
        if high_drift_examples:
            explanation += " High-drift: " + "; ".join(high_drift_examples)

        return [
            EvalResult(
                metric="drift_response",
                status=EvalStatus.OK,
                score=score,
                explanation=explanation,
                details={
                    "mean_drift": mean_drift,
                    "aligned_pairs": n_pairs,
                    "unaligned_current": len(unaligned_cur),
                    "high_drift_examples": high_drift_examples,
                },
            )
        ]

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _llm_records(self, graph: TraceGraph) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for n in graph.nodes:
            if not n.kind.value.startswith("llm_"):
                continue
            md = n.metadata or {}
            prompt = ""
            for key in (
                "rudriq.prompt_preview", "gen_ai.prompt", "prompt", "input",
            ):
                v = md.get(key)
                if isinstance(v, str) and v.strip():
                    prompt = v
                    break
            response = ""
            for key in (
                "rudriq.completion_preview", "gen_ai.completion",
                "completion", "response", "output",
            ):
                v = md.get(key)
                if isinstance(v, str) and v.strip():
                    response = v
                    break
            if prompt and response:
                records.append(
                    {"node_id": n.node_id, "prompt": prompt, "response": response}
                )
        return records
