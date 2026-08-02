"""
Phase 2A · Arm C checkpoint.

Proves Arm C's inverted gate behaves correctly in BOTH directions, and that
the arm is genuinely adversarial (induces real false positives):

  1. SHOULD-TRIP decoys actually trip (real linker), AND the gate hard-fails
     a WEAK should-trip decoy (fabricated: should-trip + silent shadow).
  2. SHOULD-STAY-SILENT fire is RECORDED as FP, not hard-failed (fabricated:
     clean-TN + firing shadow -> passed=True, fp_recorded=True). This is the
     path that must never delete the benchmark's own findings.
  3. INDUCED FP RATE is non-zero -- or an explicit suspicion flag that the
     decoys need hardening (zero-FP-on-adversarial = "benchmark you can't
     lose").
  4. TRUE-NEGATIVE pool size reported (it is the FPR denominator).
  5. BYTE-IDENTICAL reproduction (consistency with Arm A's discipline).

Writes paper2/corpus/artifacts/arm_c.json.
Run:  python -m paper2.corpus.verify_arm_c     (from repo root)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from paper2.corpus.arm_c import (
    AdversarialKind,
    ArmCCase,
    arm_c_gate,
    arm_c_to_canonical_json,
    assemble_arm_c,
)
from paper2.corpus.mechanisms import Trichotomy
from paper2.corpus.primitives import ShadowResult, StrategyOutcome

_failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def _silent_shadow() -> ShadowResult:
    return ShadowResult(
        production=StrategyOutcome("production", None, "", 0.0),
        isolated=(
            StrategyOutcome("object_identity", None, "object_identity", 1.0),
            StrategyOutcome("content_hash", None, "content_hash", 0.8),
            StrategyOutcome("substring", None, "substring", 0.7),
            StrategyOutcome("name_match", None, "name_match", 0.5),
        ),
    )


def _firing_shadow(parent: str, method: str = "substring", conf: float = 0.5) -> ShadowResult:
    return ShadowResult(
        production=StrategyOutcome("production", parent, method, conf),
        isolated=(
            StrategyOutcome("object_identity", None, "object_identity", 1.0),
            StrategyOutcome("content_hash", None, "content_hash", 0.8),
            StrategyOutcome("substring", parent, "substring", conf),
            StrategyOutcome("name_match", None, "name_match", 0.5),
        ),
    )


CLEAN_TN_N = 10
DECOY_N = 2


def main() -> int:
    print("=" * 78)
    print("Phase 2A . Arm C checkpoint - adversarial/negative arm, inverted gate")
    print("=" * 78)

    results = assemble_arm_c(clean_tn_n=CLEAN_TN_N, decoy_n=DECOY_N)

    # ---- Proof 1: should-trip decoys trip (real linker) ----
    print("\nProof 1 - should-trip decoys actually trip (real linker)")
    should_trip = [r for r in results if r.case.expect_fire]
    for r in should_trip:
        _check(f"{r.case.case_id} ({r.case.kind.value}) tripped -> FP",
               r.verdict.passed and r.verdict.fp_recorded,
               r.verdict.detail)

    # ---- Proof 1b: gate teeth - a WEAK should-trip decoy hard-fails ----
    print("\nProof 1b - gate hard-fails a WEAK should-trip decoy (fabricated silence)")
    weak = ArmCCase(
        case_id="fake-weak-decoy",
        kind=AdversarialKind.SUBSTRING_GENERIC_COLLISION,
        trichotomy=Trichotomy.TRUE_NEGATIVE,
        genuine_source=None,
        expect_fire=True,
        llm_input="x",
        notes="fabricated weak decoy that does not fire",
    )
    weak_verdict = arm_c_gate(weak, _silent_shadow())
    _check("weak should-trip decoy is REJECTED (passed=False)", not weak_verdict.passed,
           weak_verdict.detail)

    # ---- Proof 2: should-stay-silent FIRE recorded as FP, not hard-failed ----
    print("\nProof 2 - should-stay-silent fire RECORDED as FP (not hard-failed)")
    tn_that_fires = ArmCCase(
        case_id="fake-tn-fires",
        kind=AdversarialKind.CLEAN_TN,
        trichotomy=Trichotomy.TRUE_NEGATIVE,
        genuine_source=None,
        expect_fire=False,
        llm_input="x",
        notes="fabricated TN that the linker (hypothetically) false-positives on",
    )
    fired_verdict = arm_c_gate(tn_that_fires, _firing_shadow("some-unrelated-node"))
    _check("TN-that-fired is NOT hard-failed (passed=True)", fired_verdict.passed)
    _check("TN-that-fired IS recorded as a false positive (fp_recorded=True)",
           fired_verdict.fp_recorded, fired_verdict.detail)

    # ---- Proof 3: induced FP rate non-zero (else suspicion flag) ----
    print("\nProof 3 - induced FP rate over the negative pool")
    tn_pool = [r for r in results
               if r.case.kind is AdversarialKind.CLEAN_TN]
    induced_fps = [r for r in results if r.verdict.fp_recorded]
    rate = len(induced_fps) / max(1, len(tn_pool))
    print(f"         induced FPs: {len(induced_fps)}  |  clean-TN denominator: {len(tn_pool)}  "
          f"|  rate: {rate:.3f}")
    if len(induced_fps) == 0:
        _check("SUSPICION FLAG: zero induced FPs -> decoys too weak, not a win", False,
               "harden the decoys before celebrating the linker")
    else:
        _check("induced FP rate is non-zero (arm is genuinely adversarial)", True,
               f"{len(induced_fps)} FPs demonstrate the linker's failure surface")

    # ---- Proof 4: TN pool size (the FPR denominator) ----
    print("\nProof 4 - TRUE-NEGATIVE pool size (the FPR denominator)")
    clean_silent = [r for r in tn_pool if not r.verdict.fired]
    _check(f"clean-TN pool sized at {len(tn_pool)} (denominator); "
           f"{len(clean_silent)} correctly silent",
           len(tn_pool) >= 10,
           "size is a tunable knob; >=30 recommended for a tight Wilson interval at scale")

    # ---- Proof 5: byte-identical reproduction ----
    print("\nProof 5 - byte-identical reproduction")
    j1 = arm_c_to_canonical_json(assemble_arm_c(CLEAN_TN_N, DECOY_N), CLEAN_TN_N, DECOY_N)
    j2 = arm_c_to_canonical_json(assemble_arm_c(CLEAN_TN_N, DECOY_N), CLEAN_TN_N, DECOY_N)
    _check("two assemblies produce byte-identical JSON", j1 == j2,
           f"len={len(j1)} equal={j1 == j2}")

    # Write artifact + summary.
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "arm_c.json"
    artifact.write_text(j1, encoding="utf-8")

    by_kind = Counter(r.case.kind.value for r in results)
    fp_by_kind = Counter(r.case.kind.value for r in results if r.verdict.fp_recorded)
    print("\n" + "-" * 78)
    print(f"entries     : {len(results)}  {dict(by_kind)}")
    print(f"FPs by kind : {dict(fp_by_kind)}")
    print(f"artifact    : {artifact}  ({len(j1)} bytes)")

    print("\n" + "=" * 78)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED - {', '.join(_failures)}")
        return 1
    print("RESULT: ALL PROOFS PASSED - Arm C is genuinely adversarial, inverted "
          "gate correct, FPs measured not deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
