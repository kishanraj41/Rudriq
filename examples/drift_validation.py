"""Drift perturbation validation — does the drift evaluator detect change?

The Day 15 / Day 16 validation showed drift correctly returns ~1.0 on
two identical pipeline runs. That's the sanity check — same in, same
out, no false positive. This script answers the *next* question a
design partner asks: "but does it actually catch a real shift?"

What it does:

1. Builds a small synthetic trace ``BASELINE`` with a handful of
   LLM-chat calls (same prompts as a typical RAG workload, same
   response text).
2. Builds a parallel ``PERTURBED`` trace — IDENTICAL prompts, BUT the
   responses are materially rewritten ("REVISED: <different>"). One
   added LLM call is also unmatched, to exercise the "new behavior"
   reporting path.
3. Persists both as runs and runs DriftEvaluator against them.
4. Prints the two-number contrast:
     identical-runs drift_response: 1.00  (Day 15 sanity check)
     perturbed-runs drift_response: 0.<low>  (this run — detection works)
5. Asserts the perturbed drift is below a clear threshold so this
   doubles as a smoke test runnable in CI.

The artifact: the printed contrast is what answers the "does drift
work?" question for a design partner. Save / screenshot it.

Why a synthetic trace instead of two full pipeline runs: this version
is deterministic and runs in milliseconds, with no httpx / openai /
fastembed dependency at the script level. The DriftEvaluator itself
uses fastembed when available; ``embed_texts`` returns None and the
evaluator DEGRADES gracefully when it isn't. We patch ``embed_texts``
inside this script so the contrast is reproducible regardless of
local fastembed install.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from rudriq.core.schema import (
    EdgeKind,
    LinkMethod,
    NodeKind,
    TraceEdge,
    TraceGraph,
    TraceNode,
)
from rudriq.evaluate.base import EvalStatus
from rudriq.evaluate.drift import DriftEvaluator

# ---------------------------------------------------------------------
# 1. Build the two traces
# ---------------------------------------------------------------------

# The five questions a typical RAG demo would ask. We keep prompts
# identical between BASELINE and PERTURBED so the evaluator's prompt-
# similarity alignment can match them cleanly.
QUESTIONS = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "What is the speed of light?",
    "What year did the Berlin Wall fall?",
    "What is the largest ocean?",
]
BASELINE_ANSWERS = [
    "Paris is the capital of France.",
    "William Shakespeare wrote Hamlet.",
    "The speed of light is about 299,792,458 m/s.",
    "The Berlin Wall fell in 1989.",
    "The Pacific Ocean is the largest.",
]
# The perturbation: same questions, materially different responses.
# In a real-world setting this could represent a model swap, a prompt
# template change, or a regression in the retrieval layer.
PERTURBED_ANSWERS = [
    "REVISED: An entirely different claim about France's geography.",
    "REVISED: This response no longer mentions Shakespeare.",
    "REVISED: A long digression about something unrelated to physics.",
    "REVISED: An off-topic answer concerning unrelated history.",
    "REVISED: A response that is entirely about cuisine.",
]


def _build_trace(run_id: str, answers: list[str], extra_chat: bool = False) -> TraceGraph:
    """Construct a small chat-LLM trace. Each call gets a prompt+response
    pair. ``extra_chat`` adds one unmatched call (to exercise the
    drift evaluator's 'new behavior' / unaligned-current path).
    """
    base = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id=run_id, created_at=base)
    for i, (q, a) in enumerate(zip(QUESTIONS, answers)):
        g.add_node(TraceNode(
            node_id=f"{run_id}_chat_{i}",
            kind=NodeKind.LLM_CHAT,
            library="openai",
            operation="chat",
            started_at=base + timedelta(seconds=i),
            ended_at=base + timedelta(seconds=i, milliseconds=100),
            metadata={
                "rudriq.prompt_preview": q,
                "rudriq.completion_preview": a,
            },
        ))
    if extra_chat:
        g.add_node(TraceNode(
            node_id=f"{run_id}_chat_new",
            kind=NodeKind.LLM_CHAT,
            library="openai",
            operation="chat",
            started_at=base + timedelta(seconds=99),
            ended_at=base + timedelta(seconds=99, milliseconds=100),
            metadata={
                "rudriq.prompt_preview": "An entirely new question this baseline never saw.",
                "rudriq.completion_preview": "An entirely new answer.",
            },
        ))
    return g


# ---------------------------------------------------------------------
# 2. Stub embeddings deterministically so the contrast is reproducible
#    without depending on the local fastembed install.
# ---------------------------------------------------------------------


def _deterministic_embed(text: str) -> list[float]:
    """Deterministic hash-based 16-d 'embedding'.

    Identical strings → identical vectors (cosine = 1.0 → aligned by
    drift's prompt matcher). Different strings → uncorrelated vectors
    via SHA-256 avalanche → cosine near zero → high response drift.

    A handcrafted per-character sum collapses too easily — two texts
    that share most characters end up with cosine > 0.9 even when
    semantically different. The hash-based form gives clean separation
    so the validation actually exercises the detection signal rather
    than measuring the embedding's noise floor.
    """
    import hashlib

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # 16 dimensions from 16 bytes, each mapped to [-1, 1].
    return [(digest[i] - 128) / 128.0 for i in range(16)]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    return [_deterministic_embed(t) for t in texts]


# ---------------------------------------------------------------------
# 3. Run the contrast
# ---------------------------------------------------------------------


def main() -> int:
    import rudriq.evaluate.drift as drift_mod

    # Identical-run sanity check first — the Day 15 baseline behavior.
    baseline = _build_trace("baseline", BASELINE_ANSWERS)
    identical = _build_trace("identical", BASELINE_ANSWERS)
    perturbed = _build_trace("perturbed", PERTURBED_ANSWERS, extra_chat=True)

    drift_mod.embed_texts = _embed_texts  # patch for reproducibility

    print("=" * 60)
    print("Drift validation: does the evaluator detect change?")
    print("=" * 60)
    print()

    # Identical case — should return ~1.0.
    sanity = DriftEvaluator(baseline_graph=baseline).evaluate(identical)
    sanity_resp = next(r for r in sanity if r.metric == "drift_response")
    sanity_struct = next(r for r in sanity if r.metric == "drift_structural")
    print("CASE 1 — IDENTICAL runs (sanity check):")
    print(f"  drift_response   score = {sanity_resp.score:.3f}")
    print(f"  drift_structural score = {sanity_struct.score:.3f}")
    print()

    # Perturbed case — should drop materially.
    results = DriftEvaluator(baseline_graph=baseline).evaluate(perturbed)
    resp = next(r for r in results if r.metric == "drift_response")
    struct = next(r for r in results if r.metric == "drift_structural")
    print("CASE 2 — PERTURBED runs (same prompts, rewritten responses,")
    print("         + 1 new unmatched call):")
    print(f"  drift_response   score = {resp.score:.3f}")
    print(f"  drift_structural score = {struct.score:.3f}")
    print()
    print("  explanation:")
    print(f"    {resp.explanation}")
    print()

    # The artifact: the contrast.
    print("-" * 60)
    print(f"CONTRAST: identical drift_response = {sanity_resp.score:.3f}  "
          f"perturbed = {resp.score:.3f}")
    print(f"          (perturbed dropped by {sanity_resp.score - resp.score:.3f})")
    print("-" * 60)

    # Assert detection capability so this doubles as a smoke test.
    assert resp.status == EvalStatus.OK, f"perturbed drift_response not OK: {resp.status}"
    assert resp.score < 0.8, (
        f"perturbation should have produced score < 0.8; got {resp.score:.3f}"
    )
    assert sanity_resp.score > 0.95, (
        f"identical-runs sanity should produce score > 0.95; got "
        f"{sanity_resp.score:.3f}"
    )

    print()
    print("PASS — drift evaluator detected the perturbation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
