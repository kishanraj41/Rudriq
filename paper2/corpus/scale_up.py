"""
Phase 3->4 · scale-up checkpoint.

Scales Arms A + C to deliberately-chosen N so the calibration intervals
tighten enough to *quote*, and reports which quantities are now quotable as
bare linker properties versus which remain corpus-composition-dependent (the
fourth definitional wrinkle scaling was expected to surface).

Target N, chosen BY THE INTERVAL WIDTH each quotable quantity needs:
  * representative FPR (p~=0): clean-TN N=200 -> Wilson upper ~= 0.019  ("< 2%")
  * per-mechanism recall (p=1.0 / p=0): n_per_class=100 -> lower ~0.96 / upper ~0.037
  * object_identity precision (p=1.0, ~300 links): lower ~= 0.987       ("> 98%")

Run:  python -m paper2.corpus.scale_up     (from repo root)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from paper2.corpus.scoring import (
    materialize_arm_a,
    materialize_arm_c,
    score,
    wilson_interval,
)

N_PER_CLASS = 100
CLEAN_TN_N = 200
DECOY_N = 25


def _hw(rate) -> float:
    lo, hi = rate.wilson
    return (hi - lo) / 2.0


def _rate(rate) -> dict:
    return {
        "point": rate.point,
        "ci": list(rate.wilson),
        "k": rate.k,
        "n": rate.n,
        "estimable": rate.estimable,
    }


def _persist(r) -> Path:
    """Serialize the scored scale-up run.

    The paper's Results section quotes THIS corpus (N=900), not the small
    verification corpora that verify_assembler/verify_arm_c ship. Without this
    artifact the released JSON would understate the run the numbers came from.
    """
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "scale_up_n900.json"
    payload = {
        "corpus_version": "scale-up/0.1.0",
        "config": {
            "n_per_class": N_PER_CLASS,
            "clean_tn_n": CLEAN_TN_N,
            "decoy_n": DECOY_N,
            "n_calls": r.n_calls,
        },
        "calibration": {
            "monotonic": r.monotonic,
            "detail": r.monotonicity_detail,
            "band_precision": {str(b): _rate(v) for b, v in r.band_precision.items()},
            "composition_dependent_values": True,
        },
        "strategy_precision": {k: _rate(v) for k, v in r.strategy_precision.items()},
        "mechanism_recall": {k: _rate(v) for k, v in r.mechanism_recall.items()},
        "fpr_representative": _rate(r.fpr),
        "preemption": {
            "behavioural_masking": _rate(r.preemption_masking),
            "production_fp_shadow_tp": r.production_fp_shadow_tp,
            "correctness_pending_arm_b": r.preemption_correctness_pending,
        },
        "adversarial_surface": r.adversarial_surface,
        "out_of_scope_excluded": r.out_of_scope_excluded,
    }
    artifact.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return artifact


def main() -> int:
    print("=" * 80)
    print("Phase 3->4 . scale-up checkpoint (deliberate N for quotable intervals)")
    print("=" * 80)
    print(f"target N: arm_a={N_PER_CLASS}/class, clean_tn={CLEAN_TN_N}, decoy={DECOY_N}/kind")

    t0 = time.time()
    calls = materialize_arm_a(N_PER_CLASS) + materialize_arm_c(CLEAN_TN_N, DECOY_N)
    r = score(calls)
    dt = time.time() - t0
    print(f"materialised + scored {len(calls)} calls in {dt:.1f}s")

    print("\n--- QUOTABLE as bare linker properties (composition-robust) ---")

    print("\n  Calibration STRUCTURE (the headline):")
    print(f"    monotonic precision-vs-confidence: {r.monotonic}   [{r.monotonicity_detail}]")

    print("\n  object_identity precision (composition-robust: it ~never FPs):")
    print(f"    {r.strategy_precision.get('object_identity').render()}")

    print("\n  Per-mechanism recall (strategy-appropriate, conditional on the class):")
    for mech in ("verbatim_object", "element_of_collection", "exact_copy",
                 "templated_containment", "strategy_collision",
                 "paraphrased_derived", "hash_aliasing"):
        if mech in r.mechanism_recall:
            print(f"    {mech:<24}: {r.mechanism_recall[mech].render()}")

    print("\n  Representative FPR (clean-TN pool only, S2):")
    print(f"    {r.fpr.render()}   half-width={_hw(r.fpr):.4f}")

    print("\n  Pre-emption behavioural masking rate (Arm A collisions; the literal figure):")
    print(f"    {r.preemption_masking.render()}")
    print(f"    sharpest check (production wrong where a shadow strategy would be right): "
          f"{r.production_fp_shadow_tp}")

    print("\n--- NOT quotable as bare linker properties (composition-DEPENDENT) ---")
    print("  Per-band precision VALUES for the inferred strategies depend on the")
    print("  decoy:TP ratio we chose, NOT on the linker alone. Interval tightness")
    print("  does not make a composition-dependent number a linker property.")
    for b, rate in r.band_precision.items():
        print(f"    conf {b:<5}: {rate.render()}")
    print(f"  (band 0.8 = exact_copy[{N_PER_CLASS}] vs hash_aliasing[{DECOY_N}]; "
          f"band 0.7 = templated[{N_PER_CLASS}] vs generic_collision[{DECOY_N}] — "
          f"the VALUES are set by these chosen ratios.)")

    print("\n--- Interval-width check: did scaling deliver quotable tightness? ---")
    failures = []

    def _check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    oi = r.strategy_precision.get("object_identity")
    _check("object_identity precision lower-bound > 0.98",
           oi.wilson[0] > 0.98, f"CI={oi.wilson}")
    _check("representative FPR upper-bound < 0.02 (quotable 'FPR < 2%')",
           r.fpr.wilson[1] < 0.02, f"CI={r.fpr.wilson}")
    catchable_ok = all(
        r.mechanism_recall[m].wilson[0] > 0.95
        for m in ("verbatim_object", "element_of_collection", "exact_copy",
                  "templated_containment", "strategy_collision")
        if m in r.mechanism_recall
    )
    _check("catchable-class recall lower-bounds > 0.95", catchable_ok)
    para = r.mechanism_recall.get("paraphrased_derived")
    _check("paraphrase recall-ceiling upper-bound < 0.05 (quotable ceiling)",
           para.wilson[1] < 0.05, f"CI={para.wilson}")
    _check("calibration still monotonic at scale", r.monotonic, r.monotonicity_detail)
    _check("symmetric-construction check still 0 at scale",
           r.production_fp_shadow_tp == 0, f"count={r.production_fp_shadow_tp}")

    print("\n--- THE FOURTH WRINKLE (what scaling surfaced) ---")
    print("  Tightening intervals does NOT make inferred-strategy band-precision")
    print("  values quotable -- they are (linker x composition), not linker-only.")
    print("  Quotable now: monotonic STRUCTURE, object_identity precision,")
    print("  per-mechanism RECALL, representative FPR. The inferred-strategy")
    print("  precision VALUES need a representative input distribution -> Arm B.")

    print(f"\n  artifact: {_persist(r)}")

    print("\n" + "=" * 80)
    if failures:
        print(f"RESULT: {len(failures)} interval(s) not yet tight enough - {', '.join(failures)}")
        return 1
    print("RESULT: scale-up delivered quotable intervals on the composition-ROBUST")
    print("quantities; composition-dependent band values explicitly flagged for Arm B.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
