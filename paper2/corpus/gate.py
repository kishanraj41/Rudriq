"""
Phase 2A · Layer 2 — the three-regime construction-consistency gate.

The gate proves a generated pipeline tests *what it claims to test*: it
asserts the linker's behaviour is consistent with how the dataflow was
*constructed*, per the locked per-class table. A pipeline that fails the
gate is a **corpus bug**, hard-failed at generation time — never a passing
test that silently lies.

Per-class regimes (locked at the layer-1 checkpoint, with evidence)
-------------------------------------------------------------------
* **verbatim-object / element-of-collection** — ``production == intended``
  to the constructed parent at the expected confidence. Substring co-fires
  in isolation (the doc genuinely is in the content registry); that is
  *accepted*, not a cheat — so the assertion is on the production winner,
  not isolation-silence.
* **exact-copy** — ``production == content_hash`` to the parent, *plus*
  enforceable **isolation-silence**: object_identity and substring must NOT
  fire even in isolation (guaranteed by storage-surface-exclusivity).
* **templated-containment** — ``production == substring`` to the parent.
* **paraphrased-derived** — production **silent** (a true edge the live
  strategies cannot recover). Any isolation fire means the "paraphrase"
  leaked full containment → corpus bug.
* **true-negative** — production **silent** (no edge exists; silence is
  correct). Same linker behaviour as paraphrased, opposite scoring meaning
  — the gate validates construction (silence); the trichotomy label drives
  scoring.

Parent-correctness (``fired-parent == constructed-parent``) rides on every
production-fires regime: object_identity firing to the *wrong* parent via
leakage is the same bug class as the wrong strategy firing.
"""

from __future__ import annotations

from dataclasses import dataclass

from paper2.corpus.mechanisms import ConstructedCase, MechanismClass, Trichotomy
from paper2.corpus.primitives import ShadowResult


@dataclass
class GateVerdict:
    case_id: str
    regime: str
    passed: bool
    checks: list[tuple[str, bool, str]]

    def failed_checks(self) -> list[tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]


def _production_fires_regime(
    case: ConstructedCase,
    shadow: ShadowResult,
    expected_method: str,
    *,
    enforce_isolation_silence: bool,
) -> list[tuple[str, bool, str]]:
    prod = shadow.production
    checks: list[tuple[str, bool, str]] = []

    checks.append((
        f"production fires {expected_method}",
        prod.fired and prod.method == expected_method,
        f"got method={prod.method!r} fired={prod.fired}",
    ))
    # Parent-correctness — the leakage guard.
    checks.append((
        "fired-parent == constructed-parent",
        prod.parent_id == case.true_parent_id,
        f"got parent={prod.parent_id!r} expected={case.true_parent_id!r}",
    ))
    checks.append((
        f"confidence in {case.expected_confidences}",
        prod.confidence in case.expected_confidences,
        f"got confidence={prod.confidence}",
    ))

    if enforce_isolation_silence:
        for name in ("object_identity", "substring"):
            outcome = shadow.isolated_by_name(name)
            checks.append((
                f"isolation-silence: {name} does NOT fire",
                not outcome.fired,
                f"got {name}={outcome.parent_id!r}",
            ))
    return checks


def _silence_regime(
    case: ConstructedCase,
    shadow: ShadowResult,
    *,
    paraphrase_leak_check: bool,
) -> list[tuple[str, bool, str]]:
    prod = shadow.production
    checks: list[tuple[str, bool, str]] = [(
        "production is silent",
        not prod.fired,
        f"got method={prod.method!r} parent={prod.parent_id!r}",
    )]
    if paraphrase_leak_check:
        # Any strategy firing in isolation means the construction leaked a
        # full containment — the "paraphrase" is secretly a substring case.
        leaked = shadow.fired_in_isolation
        checks.append((
            "no strategy fires in isolation (no leaked containment)",
            len(leaked) == 0,
            f"leaked={[o.name for o in leaked]}",
        ))
    return checks


def _collision_regime(
    case: ConstructedCase,
    shadow: ShadowResult,
) -> list[tuple[str, bool, str]]:
    """The four-part collision condition (§13.4), via shadow capture."""
    parents = dict(case.collision_parents)  # method -> expected parent
    oi_expected = parents.get("object_identity")
    sub_expected = parents.get("substring")

    oi = shadow.isolated_by_name("object_identity")
    sub = shadow.isolated_by_name("substring")
    prod = shadow.production

    checks: list[tuple[str, bool, str]] = [
        # (a) strategy A fires in isolation -> P_A
        ("(a) object_identity fires in isolation -> P_A",
         oi.fired and oi.parent_id == oi_expected,
         f"got {oi.parent_id!r} expected {oi_expected!r}"),
        # (b) strategy B fires in isolation -> P_B
        ("(b) substring fires in isolation -> P_B",
         sub.fired and sub.parent_id == sub_expected,
         f"got {sub.parent_id!r} expected {sub_expected!r}"),
        # (c) DISTINCT parents — the load-bearing "is this a real collision"
        ("(c) P_A != P_B (genuine collision, not co-satisfiable single-parent)",
         oi.fired and sub.fired and oi.parent_id != sub.parent_id,
         f"P_A={oi.parent_id!r} P_B={sub.parent_id!r}"),
        # (d) production picks the higher-precedence strategy, masking P_B
        ("(d) production picks object_identity -> P_A (substring P_B masked)",
         prod.fired and prod.method == "object_identity"
         and prod.parent_id == oi_expected,
         f"production={prod.method}:{prod.parent_id}"),
    ]
    # Consistency: if a primary source is designated, it must be one of the
    # collision parents (the harness can only score a designation that exists).
    if case.primary_source is not None:
        checks.append((
            "primary_source (if designated) is one of the collision parents",
            case.primary_source in (oi_expected, sub_expected),
            f"primary={case.primary_source!r} parents={(oi_expected, sub_expected)}",
        ))
    return checks


def gate(case: ConstructedCase, shadow: ShadowResult) -> GateVerdict:
    """Validate one constructed case against real linker output."""
    cls = case.mechanism_class

    if cls in (MechanismClass.VERBATIM_OBJECT, MechanismClass.ELEMENT_OF_COLLECTION):
        regime = "production-fires(object_identity)"
        checks = _production_fires_regime(
            case, shadow, "object_identity", enforce_isolation_silence=False,
        )
    elif cls is MechanismClass.EXACT_COPY:
        regime = "production-fires(content_hash)+isolation-silence"
        checks = _production_fires_regime(
            case, shadow, "content_hash", enforce_isolation_silence=True,
        )
        # Surface the construction-time storage-exclusive invariant too.
        checks.append((
            "construction invariant: storage-exclusive",
            case.invariants.get("storage_exclusive", False),
            f"invariants={case.invariants}",
        ))
    elif cls is MechanismClass.TEMPLATED_CONTAINMENT:
        regime = "production-fires(substring)"
        checks = _production_fires_regime(
            case, shadow, "substring", enforce_isolation_silence=False,
        )
    elif cls is MechanismClass.STRATEGY_COLLISION:
        regime = "collision(four-part shadow condition)"
        checks = _collision_regime(case, shadow)
    elif cls is MechanismClass.PARAPHRASED_DERIVED:
        regime = "silence(inverted: any fire = leaked containment)"
        checks = _silence_regime(case, shadow, paraphrase_leak_check=True)
    elif cls is None and case.trichotomy is Trichotomy.TRUE_NEGATIVE:
        regime = "silence(true-negative: silence is correct)"
        checks = _silence_regime(case, shadow, paraphrase_leak_check=False)
    else:
        return GateVerdict(
            case_id=case.case_id,
            regime="UNKNOWN",
            passed=False,
            checks=[("recognised mechanism/trichotomy", False,
                     f"class={cls} trichotomy={case.trichotomy}")],
        )

    passed = all(ok for _, ok, _ in checks)
    return GateVerdict(case_id=case.case_id, regime=regime, passed=passed, checks=checks)
