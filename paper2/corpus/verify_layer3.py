"""
Phase 2A . Layer 3 checkpoint verification.

Proves the strategy-collision constructor and its four-part gate against the
REAL linker, plus two carry-forwards that gate-green alone does not cover:

  1. COLLISION: object_identity->P_A masks substring->P_B in production;
     shadow surfaces the masked P_B. Built symmetric AND asymmetric
     (primary_source designated) to show the harness can express both the
     behavioural and the correctness framing.
  2. DERIVATION-TRUTH (the layer-2 fix): paraphrased-derived prompts are a
     deterministic, injective function of their source -> distinct per source
     and provably bound to it, not silent non-edges.
  3. GATE TEETH: the four-part gate REJECTS a fabricated collision whose two
     strategies resolve to the same parent (the (c) P_A!=P_B check is not
     vacuous).

Run:  python -m paper2.corpus.verify_layer3     (from repo root)
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from paper2.corpus.gate import gate
from paper2.corpus.mechanisms import (
    ConstructedCase,
    MechanismClass,
    Trichotomy,
    build_paraphrased_derived,
    build_strategy_collision,
    _source_value,
    _spell_digits,
)
from paper2.corpus.primitives import (
    ShadowResult,
    StrategyOutcome,
    isolated_pipeline,
    shadow_capture,
)

_failures: list[str] = []


def _check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -  {detail}" if detail else ""))
    if not ok:
        _failures.append(label)


def _fmt(shadow: ShadowResult) -> str:
    parts = [
        f"{o.name}={o.parent_id}@{o.confidence}" if o.fired else f"{o.name}=."
        for o in shadow.isolated
    ]
    prod = shadow.production
    prod_s = f"{prod.method}:{prod.parent_id}@{prod.confidence}" if prod.fired else ".(silent)"
    return f"prod={prod_s} | iso[{', '.join(parts)}]"


def proof_collisions(tmp: Path) -> None:
    print("\nProof 1 - strategy-collision (symmetric + asymmetric)")
    for designate in (False, True):
        for idx in range(2):
            db = tmp / f"col-{designate}-{idx}.duckdb"
            with isolated_pipeline(db):
                case = build_strategy_collision(_storage_noop(), idx, designate_primary=designate)
                shadow = shadow_capture(case.llm_input)
                verdict = gate(case, shadow)
            kind = "asym" if designate else "sym "
            print(f"  [{ 'PASS' if verdict.passed else 'FAIL'}] {case.case_id} ({kind}) "
                  f"primary_source={case.primary_source}")
            print(f"         {_fmt(shadow)}")
            if not verdict.passed:
                for label, ok, detail in verdict.failed_checks():
                    print(f"         FAILED: {label} [{detail}]")
                _failures.append(case.case_id)
            # The masked parent is the pre-emption raw material: production
            # picked the object_identity parent; substring's parent is masked.
            masked = shadow.isolated_by_name("substring").parent_id
            picked = shadow.production.parent_id
            _check(f"{case.case_id}: substring parent {masked} masked by production pick {picked}",
                   masked is not None and picked is not None and masked != picked)


def proof_derivation_truth(tmp: Path) -> None:
    print("\nProof 2 - paraphrased-derived derivation-truth (the layer-2 fix)")
    prompts = {}
    for idx in range(2):
        with isolated_pipeline(tmp / f"pd-{idx}.duckdb") as storage:
            case = build_paraphrased_derived(storage, idx)
            shadow = shadow_capture(case.llm_input)
        prompts[idx] = case.llm_input
        # Linker still correctly silent (recall ceiling).
        _check(f"pd-{idx}: production silent (FN by design)", not shadow.production.fired)
        # The derived datum (transcoded value) is present and source-specific.
        words = _spell_digits(_source_value(idx))
        _check(f"pd-{idx}: derived datum '{words}' present in prompt", words in case.llm_input)
    _check("pd-0 and pd-1 prompts DIFFER (source-specific, not generic)",
           prompts[0] != prompts[1],
           f"identical={prompts[0] == prompts[1]}")


def proof_gate_teeth() -> None:
    print("\nProof 3 - collision gate has teeth (rejects a fake collision)")
    # Fabricate a 'collision' whose two strategies resolve to the SAME parent
    # — a co-satisfiable single-parent pipeline mislabelled as a collision.
    fake = ConstructedCase(
        case_id="fake-collision",
        mechanism_class=MechanismClass.STRATEGY_COLLISION,
        trichotomy=Trichotomy.LINKED_TRUE,
        true_parent_id=None,
        llm_input=["irrelevant"],
        expected_method="object_identity",
        expected_confidences=(0.95,),
        collision_parents=(("object_identity", "SAME"), ("substring", "SAME")),
    )
    fake_shadow = ShadowResult(
        production=StrategyOutcome("production", "SAME", "object_identity", 0.95),
        isolated=(
            StrategyOutcome("object_identity", "SAME", "object_identity", 0.95),
            StrategyOutcome("content_hash", None, "content_hash", 0.8),
            StrategyOutcome("substring", "SAME", "substring", 0.7),
            StrategyOutcome("name_match", None, "name_match", 0.5),
        ),
    )
    verdict = gate(fake, fake_shadow)
    _check("gate REJECTS fake collision (P_A == P_B)", not verdict.passed,
           f"passed={verdict.passed}")
    # And the specific failing check is (c).
    failed_labels = [c[0] for c in verdict.failed_checks()]
    _check("the (c) distinct-parents check is the one that fires",
           any(lbl.startswith("(c)") for lbl in failed_labels),
           f"failed={failed_labels}")


class _NoopStorage:
    """Collision construction never persists to storage (it uses the identity
    and content surfaces only), so the constructor's storage arg is unused.
    Pass a no-op to keep the call signature honest without a real DB."""
    def save_node(self, *a, **k):  # pragma: no cover - never called
        raise AssertionError("collision constructor unexpectedly used storage")


def _storage_noop() -> _NoopStorage:
    return _NoopStorage()


def main() -> int:
    print("=" * 78)
    print("Phase 2A . Layer 3 checkpoint - strategy-collision + four-part gate")
    print("=" * 78)
    with tempfile.TemporaryDirectory(prefix="rudriq_p2_l3_") as td:
        tmp = Path(td)
        proof_collisions(tmp)
        proof_derivation_truth(tmp)
        proof_gate_teeth()

    print("\n" + "=" * 78)
    if _failures:
        print(f"RESULT: {len(_failures)} FAILED - {', '.join(_failures)}")
        return 1
    print("RESULT: ALL PROOFS PASSED - collision constructor + gate are correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
