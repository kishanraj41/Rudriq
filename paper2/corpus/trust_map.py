"""
Phase 4 . failure-cluster formalization -> the TRUST-ENVELOPE MAP.

Not a re-listing of failures. The synthesis that answers the reviewer's (and
the deploying buyer's) question: *given all the failure modes, what is the
honest envelope of when this linker should and shouldn't be trusted?*

Two things this analysis makes explicit:

1. **Bounded FP mechanisms vs. the structural recall ceiling are different in
   kind.** The hash-aliasing and substring-collision FPs are *bounded* - they
   fire only in characterised conditions and respect their boundaries
   (near-misses stay silent). The paraphrase ceiling is *not a bug that fires
   in an edge case* - it is a whole class of true edges no live strategy can
   catch (the `name_match` strategy is a stub returning None). Conflating them
   would misrepresent both: one is a precision risk in a known regime, the
   other is a coverage gap with a roadmap.

2. **The failure surface EXPLAINS the calibration result.** The FP mechanisms
   land in the *lower* confidence bands (content_hash 0.8, substring 0.7) -
   which is *why* calibration came out monotonic. So thresholding on confidence
   excludes specific, characterised failure modes. That turns calibration from
   "monotonic, nice" into a deployment guideline: *threshold at >= X to exclude
   these FP modes* - computed here from real data, not asserted.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from paper2.corpus.scoring import (
    Rate,
    ScoredCall,
    materialize_arm_a,
    materialize_arm_c,
)

# N=900 corpus parameters, matching the scale-up run, so the trust-map shares
# ONE coherent corpus with the rest of §5 (the seeded-determinism claim must
# hold across every results table, not just within one).
N_PER_CLASS = 100
CLEAN_TN_N = 200
DECOY_N = 25
THRESHOLDS = (1.0, 0.95, 0.8, 0.7)


@dataclass
class ThresholdPoint:
    threshold: float
    precision: Rate
    recall: Rate
    admitted_fp_modes: list[str]


def threshold_sweep(calls: list[ScoredCall], thresholds=THRESHOLDS) -> list[ThresholdPoint]:
    """For each confidence threshold, treat links below it as no-link, then
    score. Shows the precision/recall trade-off and which FP modes are admitted."""
    in_scope = [c for c in calls if c.trichotomy != "out_of_scope"]
    positives = [c for c in in_scope if c.trichotomy == "linked_true"]
    points: list[ThresholdPoint] = []
    for t in thresholds:
        accepted = [c for c in in_scope if c.fired and c.fired_confidence >= t]
        tp = [c for c in accepted if c.correct]
        precision = Rate(len(tp), len(accepted))
        recall_hits = [c for c in positives if c.fired and c.fired_confidence >= t and c.correct]
        recall = Rate(len(recall_hits), len(positives))
        admitted = sorted({c.mechanism for c in accepted if not c.correct})
        points.append(ThresholdPoint(t, precision, recall, admitted))
    return points


def main() -> int:
    print("=" * 80)
    print("Phase 4 . TRUST-ENVELOPE MAP (failure clusters -> deployment guideline)")
    print("=" * 80)

    calls = materialize_arm_a(N_PER_CLASS) + materialize_arm_c(CLEAN_TN_N, DECOY_N)
    sweep = threshold_sweep(calls)

    # ---- Cluster 1: bounded FP mechanisms (fire in characterised regimes) ----
    print("\n[Cluster 1] BOUNDED FP MECHANISMS - fire only in characterised conditions,")
    print("            respect their boundaries (the matching near-miss stays silent).")
    bounded = [
        ("content_hash aliasing", "content_hash", 0.8, "wrong-parent",
         "two upstream nodes with identical content; returns most-recent. "
         "Boundary held: one-char-off (hash-adjacent) stays silent."),
        ("substring generic-collision", "substring", 0.7, "spurious-link",
         "a generic >=20-char phrase incidentally fully contained in an unrelated "
         "prompt. Boundary held: 18-char below-floor near-miss stays silent."),
    ]
    for name, strat, band, kind, desc in bounded:
        print(f"  - {name}: {strat} @ conf {band} ({kind} FP). {desc}")

    # ---- Cluster 2: structural recall ceiling (different in KIND) ----
    print("\n[Cluster 2] STRUCTURAL RECALL CEILING - NOT an edge-case bug; a whole class")
    print("            of true edges no live strategy can recover.")
    print("  - paraphrased/derived links (genuine derivation, no shared verbatim surface).")
    print("    Cause: the `name_match` strategy is a stub returning None (v0.0.2). This is a")
    print("    KNOWN COVERAGE GAP WITH A ROADMAP (implement name_match), not a misfire. It is")
    print("    reported as a recall ceiling, never folded into the FP mechanisms above.")

    # ---- Cluster 3: latency regime (cross-ref S15) ----
    print("\n[Cluster 3] LATENCY REGIME (cross-ref methodology S15, overhead):")
    print("  - content_hash ~1000x in-process (storage I/O; batch/defer; flat O(1)).")
    print("  - substring O(n) in registry size (inverted-index roadmap; watch at 50k cap).")
    print("  - common path (object_identity hit) ~9.5us = <0.1% of an LLM call.")

    # ---- The connection to calibration: the deployment guideline ----
    print("\n--- DEPLOYMENT GUIDELINE: confidence threshold excludes characterised FP modes ---")
    print("  (The FP mechanisms concentrate in the LOWER bands -- which is WHY calibration is")
    print("   monotonic. Thresholding is therefore a precision/recall knob with known content.)")
    print(f"\n  {'threshold':>10} | {'precision':>26} | {'recall':>26} | admits FP modes")
    for p in sweep:
        prec = f"{p.precision.point:.3f} [{p.precision.wilson[0]:.2f},{p.precision.wilson[1]:.2f}]" if p.precision.n else "n/a"
        rec = f"{p.recall.point:.3f} [{p.recall.wilson[0]:.2f},{p.recall.wilson[1]:.2f}]" if p.recall.n else "n/a"
        modes = ", ".join(p.admitted_fp_modes) if p.admitted_fp_modes else "(none)"
        print(f"  conf >= {p.threshold:<4} | {prec:>26} | {rec:>26} | {modes}")

    # Derive the FP-intolerant threshold: lowest threshold with precision == 1.0.
    clean = [p for p in sweep if p.precision.point == 1.0 and p.precision.n > 0]
    # Lowest clean threshold: accepts the MOST links while still precision 1.0.
    fp_intolerant = min((p.threshold for p in clean), default=None)
    print("\n  Synthesis (the trust envelope):")
    if fp_intolerant is not None:
        guide = next(p for p in sweep if p.threshold == fp_intolerant)
        print(f"  * FP-INTOLERANT deployments (audit / regulated): threshold conf >= {fp_intolerant}")
        print(f"    -> precision ~1.0, recall {guide.recall.point:.3f}; excludes ALL characterised")
        print(f"       FP modes (content_hash aliasing @0.8, substring collision @0.7).")
    recall_max = sweep[-1]
    print(f"  * RECALL-MAXIMISING deployments: threshold conf >= {recall_max.threshold}")
    print(f"    -> recall {recall_max.recall.point:.3f}, but admits {', '.join(recall_max.admitted_fp_modes) or '(none)'}")
    print("  * BLIND SPOT (both regimes): paraphrase/derived links (recall ceiling) are")
    print("    unrecoverable at ANY threshold until name_match is implemented.")

    # Artifact.
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "trust_map.json"
    payload = {
        "trust_envelope": {
            "fp_intolerant_threshold": fp_intolerant,
            "recall_maximising_threshold": recall_max.threshold,
            "structural_blind_spot": "paraphrased_derived (name_match stub) - unrecoverable at any threshold",
        },
        "clusters": {
            "bounded_fp": [
                {"name": n, "strategy": s, "band": b, "kind": k} for n, s, b, k, _ in bounded
            ],
            "structural_recall_ceiling": {
                "class": "paraphrased_derived",
                "cause": "name_match strategy is a stub (returns None)",
                "roadmap": "implement name_match",
            },
            "latency_regime": {
                "content_hash": "~1000x in-process; storage I/O; batch/defer",
                "substring": "O(n) registry scan; inverted-index roadmap",
                "common_path_us": 9.5,
            },
        },
        "threshold_sweep": [
            {
                "threshold": p.threshold,
                "precision": p.precision.point,
                "precision_ci": list(p.precision.wilson),
                "recall": p.recall.point,
                "recall_ci": list(p.recall.wilson),
                "admits_fp_modes": p.admitted_fp_modes,
            }
            for p in sweep
        ],
    }
    artifact.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(f"\n  artifact: {artifact}")

    print("\n" + "=" * 80)
    print("RESULT: trust-envelope map synthesised from real data - bounded-FP separated from")
    print("structural recall-ceiling, failure surface connected to calibration as a threshold guide.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
