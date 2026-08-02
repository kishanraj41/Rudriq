"""
Phase 2A · Arm C — the adversarial / negative arm.

Arm C exists to answer the sharpest reviewer objection — *"you built a
benchmark you can't lose"* — and to populate the TRUE-NEGATIVE pool that
FPR mathematically requires. It is the arm where the linker is *supposed*
to fail (or abstain), and where the benchmark's credibility actually lives:
a reviewer trusts a paper that quantifies its tool's failure surface far
more than one that reports perfection.

The inverted three-regime gate
------------------------------------------------------------------------
Unlike every gate so far (where a fire-where-silence-expected was a corpus
bug), Arm C contains pipelines with *opposite* pass conditions:

* **CLEAN_TN / NEAR_MISS** (``expect_fire=False``) — correct behaviour is
  silence. The construction is *always valid*; a fire is **not** a gate
  failure, it is a **measured false positive** — the exact event FPR exists
  to count. The gate **records** it, never hard-fails it. Hard-failing here
  would delete the benchmark's own findings.
* **SHOULD_TRIP** (``expect_fire=True``) — crafted to demonstrate the FP
  surface exists. Pass condition is **the decoy actually fires (as an FP)**.
  A non-fire means the decoy was too weak to exercise the surface — *that*
  is the corpus bug, and the gate **hard-fails** it.

The load-bearing distinction: for the first kind a fire is a *result to
measure*; for the second a non-fire is a *bug to fix*.

The TRUE-NEGATIVE pool is the FPR denominator (§3.1), so its size and
representativeness matter for the metric's stability — it is sized
deliberately, not as filler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from paper2.corpus.mechanisms import Trichotomy
from paper2.corpus.primitives import (
    FROZEN_TS,
    DuckDBStorage,
    ShadowResult,
    isolated_pipeline,
    register_identity_surface,
    register_storage_surface,
    shadow_capture,
)

CORPUS_VERSION = "arm-c/0.1.0"


class AdversarialKind(str, Enum):
    CLEAN_TN = "clean_true_negative"
    SUBSTRING_NEAR_MISS = "substring_near_miss"
    SUBSTRING_GENERIC_COLLISION = "substring_generic_collision"
    HASH_ADJACENT_NEAR_MISS = "hash_adjacent_near_miss"
    HASH_ALIASING = "hash_aliasing"


@dataclass(frozen=True)
class ArmCCase:
    case_id: str
    kind: AdversarialKind
    trichotomy: Trichotomy
    # The genuine upstream source, if one exists (HASH_ALIASING has a real
    # source the linker mis-attributes). ``None`` for pure negatives, where
    # *any* link is a false positive.
    genuine_source: str | None
    expect_fire: bool          # construction intent
    llm_input: Any
    notes: str


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def _generic_phrase(idx: int) -> str:
    # A generic ~40-char phrase plausibly recurring across unrelated contexts —
    # the realistic substring collision surface (>= 20 chars so it is not
    # length-filtered).
    return "Summary of key findings and recommendations follows below"


def _fresh_query(idx: int) -> str:
    return f"What is the projected delivery window for shipment batch {idx}?"


def _hash_payload(idx: int) -> str:
    return f"reconciled ledger snapshot for accounting period {idx} (final)"


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def build_clean_true_negative(storage: DuckDBStorage, idx: int) -> ArmCCase:
    """Genuinely-new input, an unrelated upstream registered. Correct
    behaviour: silence. This is the FPR denominator."""
    register_identity_surface(f"Unrelated upstream document number {idx} content.", f"tn-up-{idx}")
    return ArmCCase(
        case_id=f"c-cleantn-{idx}",
        kind=AdversarialKind.CLEAN_TN,
        trichotomy=Trichotomy.TRUE_NEGATIVE,
        genuine_source=None,
        expect_fire=False,
        llm_input=_fresh_query(idx),
        notes="genuinely-new query; silence is correct; FPR denominator.",
    )


def build_substring_near_miss(storage: DuckDBStorage, idx: int) -> ArmCCase:
    """A shared string ONE char below the 20-char floor, fully present in the
    prompt. Correct behaviour: silence (the floor holds). Boundary probe — if
    it fires, the floor is weaker than documented and the fire is recorded as
    an FP, not hard-failed."""
    below_floor = "edge case phrase19"  # 18 chars, < min_match_length=20
    assert len(below_floor) < 20
    register_identity_surface(below_floor, f"nm-up-{idx}")
    return ArmCCase(
        case_id=f"c-subnearmiss-{idx}",
        kind=AdversarialKind.SUBSTRING_NEAR_MISS,
        trichotomy=Trichotomy.TRUE_NEGATIVE,
        genuine_source=None,
        expect_fire=False,
        llm_input=f"Note {idx}: the phrase '{below_floor}' appears but is too short to link.",
        notes="shared string below 20-char floor; silence expected; boundary probe.",
    )


def build_substring_generic_collision(storage: DuckDBStorage, idx: int) -> ArmCCase:
    """A generic >= 20-char phrase registered as upstream, then fully embedded
    in an UNRELATED call's prompt. substring links the call to the generic
    phrase though the call did not derive from it -> a spurious-link false
    positive. SHOULD_TRIP: must fire, else the decoy is too weak."""
    generic = _generic_phrase(idx)
    assert len(generic) >= 20
    register_identity_surface(generic, f"gc-up-{idx}")
    prompt = (
        f"Quarterly briefing for team {idx}. {generic}. "
        f"Please draft next steps for the unrelated logistics workstream."
    )
    return ArmCCase(
        case_id=f"c-genericcollision-{idx}",
        kind=AdversarialKind.SUBSTRING_GENERIC_COLLISION,
        trichotomy=Trichotomy.TRUE_NEGATIVE,  # no genuine source; the link is spurious
        genuine_source=None,
        expect_fire=True,
        llm_input=prompt,
        notes="generic phrase incidentally contained in unrelated prompt; substring FP @ 0.5.",
    )


def build_hash_adjacent_near_miss(storage: DuckDBStorage, idx: int) -> ArmCCase:
    """An input ONE character off a persisted node's content -> different
    SHA-256 -> content_hash correctly silent. Boundary probe for content_hash
    exactness. Silence expected; a fire would be a (near-impossible) hash
    collision FP, recorded not hard-failed."""
    payload = _hash_payload(idx)
    register_storage_surface(payload, f"ha-node-{idx}", storage)
    adjacent = payload + "."  # one char different -> different hash
    return ArmCCase(
        case_id=f"c-hashadjacent-{idx}",
        kind=AdversarialKind.HASH_ADJACENT_NEAR_MISS,
        trichotomy=Trichotomy.TRUE_NEGATIVE,
        genuine_source=None,
        expect_fire=False,
        llm_input=adjacent,
        notes="input one char off a persisted hash; content_hash exactness boundary.",
    )


def build_hash_aliasing(storage: DuckDBStorage, idx: int) -> ArmCCase:
    """Two persisted nodes with IDENTICAL content: a real source (earlier) and
    a decoy (later). find_nodes_by_hash returns most-recent-first, so the
    linker attributes the input to the DECOY, not the real source -> a
    wrong-parent false positive. SHOULD_TRIP: must fire (to the decoy)."""
    payload = _hash_payload(idx)
    real_id, decoy_id = f"alias-real-{idx}", f"alias-decoy-{idx}"
    # Real registered earlier; decoy later -> decoy is most-recent.
    register_storage_surface(payload, real_id, storage, started_at=FROZEN_TS)
    register_storage_surface(
        payload, decoy_id, storage, started_at=FROZEN_TS + timedelta(minutes=1),
    )
    return ArmCCase(
        case_id=f"c-hashaliasing-{idx}",
        kind=AdversarialKind.HASH_ALIASING,
        trichotomy=Trichotomy.LINKED_TRUE,  # a real source exists...
        genuine_source=real_id,             # ...but the linker mis-attributes.
        expect_fire=True,
        llm_input=payload,
        notes="aliased identical-content nodes; content_hash returns most-recent decoy -> wrong-parent FP.",
    )


# Decoy kinds and the clean-TN denominator are sized separately: the TN pool
# is the FPR denominator, the decoys probe the surface.
_DECOY_CONSTRUCTORS = (
    ("substring_near_miss", build_substring_near_miss),
    ("substring_generic_collision", build_substring_generic_collision),
    ("hash_adjacent_near_miss", build_hash_adjacent_near_miss),
    ("hash_aliasing", build_hash_aliasing),
)


# ---------------------------------------------------------------------------
# The inverted three-regime gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmCVerdict:
    case_id: str
    kind: str
    expect_fire: bool
    fired: bool
    fired_parent: str | None
    fp_recorded: bool   # the linker emitted a false positive
    passed: bool        # CONSTRUCTION validity (not "linker was correct")
    detail: str


def arm_c_gate(case: ArmCCase, shadow: ShadowResult) -> ArmCVerdict:
    prod = shadow.production
    fired = prod.fired
    fired_parent = prod.parent_id
    # An FP is any emitted link that is not to the genuine source. For pure
    # negatives (genuine_source=None) any fire is an FP; for aliasing, a link
    # to the decoy (!= real) is an FP.
    fp = fired and fired_parent != case.genuine_source

    if case.expect_fire:
        # SHOULD_TRIP: the decoy must actually trip (as an FP). A non-fire is a
        # weak-decoy CONSTRUCTION bug -> hard-fail.
        passed = fired and fp
        detail = (
            f"should-trip: tripped ({prod.method} -> {fired_parent})"
            if passed else "WEAK DECOY: did not trip (construction bug)"
        )
    else:
        # CLEAN_TN / NEAR_MISS: construction is always valid. A fire is a
        # MEASURED FP, recorded, never hard-failed.
        passed = True
        detail = (
            f"fire RECORDED as FP ({prod.method} -> {fired_parent})"
            if fired else "correctly silent (TN)"
        )

    return ArmCVerdict(
        case_id=case.case_id,
        kind=case.kind.value,
        expect_fire=case.expect_fire,
        fired=fired,
        fired_parent=fired_parent,
        fp_recorded=fp,
        passed=passed,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class ArmCResult:
    case: ArmCCase
    verdict: ArmCVerdict


def assemble_arm_c(clean_tn_n: int = 10, decoy_n: int = 2) -> list[ArmCResult]:
    """Build the Arm C corpus: ``clean_tn_n`` clean true-negatives (the FPR
    denominator) plus ``decoy_n`` of each decoy kind. Runs the inverted gate
    per pipeline. Deterministic; no timestamps sampled.

    Note: unlike Arm A, assembly does NOT raise on a recorded FP — that is a
    measured result. It WOULD surface a weak should-trip decoy (passed=False),
    which the checkpoint treats as a corpus bug.
    """
    import tempfile

    results: list[ArmCResult] = []
    with tempfile.TemporaryDirectory(prefix="rudriq_armc_") as td:
        base = Path(td)
        for idx in range(clean_tn_n):
            db = base / f"cleantn-{idx}.duckdb"
            with isolated_pipeline(db) as storage:
                case = build_clean_true_negative(storage, idx)
                shadow = shadow_capture(case.llm_input)
                verdict = arm_c_gate(case, shadow)
            results.append(ArmCResult(case, verdict))
        for name, builder in _DECOY_CONSTRUCTORS:
            for idx in range(decoy_n):
                db = base / f"{name}-{idx}.duckdb"
                with isolated_pipeline(db) as storage:
                    case = builder(storage, idx)
                    shadow = shadow_capture(case.llm_input)
                    verdict = arm_c_gate(case, shadow)
                results.append(ArmCResult(case, verdict))
    return results


def arm_c_to_canonical_json(results: list[ArmCResult], clean_tn_n: int, decoy_n: int) -> str:
    tn_pool = [r for r in results if r.case.trichotomy is Trichotomy.TRUE_NEGATIVE
               and r.case.kind is AdversarialKind.CLEAN_TN]
    induced_fps = [r for r in results if r.verdict.fp_recorded]
    payload = {
        "corpus_version": CORPUS_VERSION,
        "arm": "C",
        "config": {"clean_tn_n": clean_tn_n, "decoy_n": decoy_n},
        "pool_mapping": {
            "arm": "C",
            "role": "adversarial/negative",
            "provides_pools": ["true_negative"],
            "fpr_denominator_pool": "true_negative(clean)",
            "fpr_denominator_n": len(tn_pool),
        },
        "induced_fp_count": len(induced_fps),
        "induced_fp_rate_over_negatives": (
            round(len(induced_fps) / max(1, len(tn_pool)), 6)
        ),
        "entries": [
            {
                "case_id": r.case.case_id,
                "kind": r.case.kind.value,
                "trichotomy": r.case.trichotomy.value,
                "genuine_source": r.case.genuine_source,
                "expect_fire": r.case.expect_fire,
                "fired": r.verdict.fired,
                "fired_parent": r.verdict.fired_parent,
                "fp_recorded": r.verdict.fp_recorded,
                "gate_passed": r.verdict.passed,
                "detail": r.verdict.detail,
                "rendered_input": r.case.llm_input,
                "notes": r.case.notes,
            }
            for r in results
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True)
