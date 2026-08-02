"""
Phase 3 . scoring-harness checkpoint.

Runs the harness against Arms A + C (re-materialised against the real linker)
and prints the full report, surfacing the definitional sharpenings the
harness forced - the whole reason to build it before Arm B's annotation labor.

Run:  python -m paper2.corpus.verify_harness     (from repo root)
"""

from __future__ import annotations

import sys

from paper2.corpus.scoring import (
    FPR_MIN_DENOMINATOR,
    materialize_arm_a,
    materialize_arm_c,
    score,
)

# Arm A: 8 per class (48 LINKED-TRUE calls). Arm C: 30 clean-TN (the FPR
# denominator, sized to the methodology floor) + 3 of each decoy.
N_PER_CLASS = 8
CLEAN_TN_N = 30
DECOY_N = 3


def main() -> int:
    print("=" * 78)
    print("Phase 3 . scoring-harness checkpoint (Arms A + C, real linker)")
    print("=" * 78)

    calls = materialize_arm_a(N_PER_CLASS) + materialize_arm_c(CLEAN_TN_N, DECOY_N)
    r = score(calls)

    print(f"\ncalls scored: {r.n_calls}  "
          f"(arm_a={sum(1 for c in calls if c.source == 'arm_a')}, "
          f"arm_c={sum(1 for c in calls if c.source == 'arm_c')})")

    print("\n--- Calibration: precision per confidence band (the headline result) ---")
    for b, rate in r.band_precision.items():
        print(f"  conf {b:<5}: {rate.render()}")
    print(f"  monotonic (precision falls with confidence): {r.monotonic}   [{r.monotonicity_detail}]")

    print("\n--- Precision per strategy ---")
    for m, rate in r.strategy_precision.items():
        print(f"  {m:<16}: {rate.render()}")

    print("\n--- Recall per mechanism class (strategy-appropriate, S13.2) ---")
    for mech, rate in r.mechanism_recall.items():
        print(f"  {mech:<24}: {rate.render()}")

    print("\n--- Headline rates (each with its uncertainty) ---")
    print(f"  link-level recall : {r.recall.render()}")
    print(f"  blended precision : {r.precision.render()}")
    print(f"  representative FPR: {r.fpr.render()}")

    print("\n--- Adversarial FP surface (reported SEPARATELY from FPR - S2) ---")
    if r.adversarial_surface:
        for kind, n in sorted(r.adversarial_surface.items()):
            print(f"  {kind:<30}: {n} induced FP(s) - surface confirmed")
    else:
        print("  (none - would be a suspicion flag that decoys are too weak)")

    print("\n--- Pre-emption: TWO claims, structurally separated ---")
    print(f"  [claim 1 . behavioural, Arm A high-N] masking rate: {r.preemption_masking.render()}")
    print(f"  [claim 2 . correctness, Arm B] PENDING - no synthetic evidence by design "
          f"(symmetric construction); awaits annotation")
    print(f"  sharpest check (production scored wrong where a shadow strategy would be right): "
          f"{r.production_fp_shadow_tp}")

    print("\n--- OUT-OF-SCOPE handling ---")
    print(f"  excluded from precision/recall/FPR: {r.out_of_scope_excluded}  ({r.coverage_note})")

    # ---- Assertions on the forced sharpenings ----
    print("\n--- Checkpoint assertions ---")
    failures = []

    def _check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    # S1/S2: representative FPR uses clean-TN only and is ~0 (linker doesn't
    # FP on representative negatives), while the adversarial surface is real.
    _check("representative FPR estimable at the methodology floor",
           r.fpr.estimable and r.fpr.n >= FPR_MIN_DENOMINATOR,
           f"n={r.fpr.n}")
    _check("representative FPR is ~0 (no FP on clean negatives)",
           r.fpr.k == 0, f"k={r.fpr.k}")
    _check("adversarial FP surface is non-empty (arm genuinely adversarial)",
           len(r.adversarial_surface) > 0, f"{r.adversarial_surface}")
    _check("adversarial decoys are NOT in the FPR denominator (S2)",
           r.fpr.n == CLEAN_TN_N, f"fpr_n={r.fpr.n} expected clean_tn={CLEAN_TN_N}")

    # Calibration: high bands perfectly precise; the FP surface lands in the
    # lower bands (content_hash 0.8 aliasing, substring collision).
    high = [r.band_precision[b].point for b in (1.0, 0.95) if b in r.band_precision]
    _check("confidence-1.0 and 0.95 bands are perfectly precise",
           all(p == 1.0 for p in high), f"{high}")

    # Recall: paraphrased-derived is the recall ceiling (0 recall by design).
    _check("paraphrased_derived recall is 0 (the recall ceiling, FN by design)",
           r.mechanism_recall.get("paraphrased_derived")
           and r.mechanism_recall["paraphrased_derived"].k == 0,
           f"{r.mechanism_recall.get('paraphrased_derived').render() if r.mechanism_recall.get('paraphrased_derived') else 'missing'}")

    # The sharpest pre-emption check confirms symmetric construction held.
    _check("no case where production scored wrong but a shadow strategy was right "
           "(confirms symmetric construction)",
           r.production_fp_shadow_tp == 0, f"count={r.production_fp_shadow_tp}")

    # Behavioural masking is the real Arm A pre-emption finding.
    _check("behavioural pre-emption masking measured on collisions",
           r.preemption_masking.n > 0 and r.preemption_masking.k > 0,
           r.preemption_masking.render())

    print("\n" + "=" * 78)
    if failures:
        print(f"RESULT: {len(failures)} FAILED - {', '.join(failures)}")
        return 1
    print("RESULT: ALL CHECKS PASSED - harness scores correctly, sharpenings enforced, "
          "numbers travel with their uncertainty.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
