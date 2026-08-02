"""
Phase 3 . the scoring harness.

Reads the corpus (re-materialised deterministically - the assembler proved
re-materialisation works), runs ``correlate`` in production + shadow modes,
and scores each call against ground truth per methodology §2/§3.

Three definitional sharpenings the harness *forced* (surfaced at the
checkpoint, because finding them now is the whole point of building the
harness before Arm B's annotation labor):

  S1. **Two distinct notions of "false positive."** A *precision-FP* is any
      incorrect emitted link - a wrong-parent link on a positive OR any link
      on a negative. The *FPR-FP* is narrower: a link on a TRUE-NEGATIVE.
      Wrong-parent links on positives are precision-FPs but are NOT in the
      FPR negative pool. Folding them together would corrupt FPR.

  S2. **The FPR denominator is the *representative* clean-TN pool, NOT the
      adversarial decoys.** Including the decoys (built to be hard) inflates
      the headline rate - the exact overclaim the 0.400 caveat warned about.
      So FPR is computed over CLEAN_TN only; the adversarial decoys are
      reported *separately* as "FP surface exists," never folded into FPR.

  S3. **Blended precision/FPR are corpus-composition-dependent** and are
      therefore NOT headline numbers. The meaningful results are per-strategy
      and per-confidence-band (calibration). The harness refuses to present a
      blended rate as if it were the answer.

Every rate is reported WITH its Wilson interval, and a rate from an
undersized denominator is flagged "not estimable," never printed as a point
estimate.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from paper2.corpus.mechanisms import Trichotomy
from paper2.corpus.primitives import isolated_pipeline, shadow_capture

# Methodology floor for a usable Wilson interval on a rate (§ Arm C note).
FPR_MIN_DENOMINATOR = 30


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class ScoredCall:
    source: str                       # "arm_a" | "arm_c"
    case_id: str
    trichotomy: str                   # linked_true | true_negative | out_of_scope
    mechanism: str                    # mechanism class or Arm C kind
    acceptable_parents: frozenset     # ground-truth correct parents ({} for pure negatives)
    is_clean_negative: bool           # CLEAN_TN -> the FPR denominator
    is_adversarial: bool              # decoy -> FP-surface, NOT the FPR denominator
    fired: bool
    fired_parent: str | None
    fired_method: str
    fired_confidence: float
    isolated: tuple                   # tuple[(name, parent, method, conf), ...]

    @property
    def correct(self) -> bool:
        return self.fired and self.fired_parent in self.acceptable_parents


# ---------------------------------------------------------------------------
# Re-materialisation
# ---------------------------------------------------------------------------


def _iso_tuple(shadow) -> tuple:
    return tuple((o.name, o.parent_id, o.method, o.confidence) for o in shadow.isolated)


def materialize_arm_a(n_per_class: int = 8) -> list[ScoredCall]:
    from paper2.corpus.assembler import ARM_A_FULL

    calls: list[ScoredCall] = []
    with tempfile.TemporaryDirectory(prefix="rudriq_score_a_") as td:
        base = Path(td)
        for name, builder in ARM_A_FULL:
            for idx in range(n_per_class):
                with isolated_pipeline(base / f"{name}-{idx}.duckdb") as storage:
                    case = builder(storage, idx)
                    shadow = shadow_capture(case.llm_input)
                if case.collision_parents:
                    acceptable = frozenset(p for _, p in case.collision_parents)
                elif case.true_parent_id is not None:
                    acceptable = frozenset({case.true_parent_id})
                else:
                    acceptable = frozenset()
                calls.append(ScoredCall(
                    source="arm_a",
                    case_id=case.case_id,
                    trichotomy=case.trichotomy.value,
                    mechanism=case.mechanism_class.value if case.mechanism_class else "none",
                    acceptable_parents=acceptable,
                    is_clean_negative=False,
                    is_adversarial=False,
                    fired=shadow.production.fired,
                    fired_parent=shadow.production.parent_id,
                    fired_method=shadow.production.method,
                    fired_confidence=shadow.production.confidence,
                    isolated=_iso_tuple(shadow),
                ))
    return calls


def materialize_arm_c(clean_tn_n: int = 30, decoy_n: int = 3) -> list[ScoredCall]:
    from paper2.corpus.arm_c import (
        AdversarialKind,
        _DECOY_CONSTRUCTORS,
        build_clean_true_negative,
    )

    calls: list[ScoredCall] = []
    with tempfile.TemporaryDirectory(prefix="rudriq_score_c_") as td:
        base = Path(td)

        def _emit(case, shadow) -> None:
            acceptable = (
                frozenset({case.genuine_source}) if case.genuine_source else frozenset()
            )
            calls.append(ScoredCall(
                source="arm_c",
                case_id=case.case_id,
                trichotomy=case.trichotomy.value,
                mechanism=case.kind.value,
                acceptable_parents=acceptable,
                is_clean_negative=case.kind is AdversarialKind.CLEAN_TN,
                is_adversarial=case.kind is not AdversarialKind.CLEAN_TN,
                fired=shadow.production.fired,
                fired_parent=shadow.production.parent_id,
                fired_method=shadow.production.method,
                fired_confidence=shadow.production.confidence,
                isolated=_iso_tuple(shadow),
            ))

        for idx in range(clean_tn_n):
            with isolated_pipeline(base / f"cleantn-{idx}.duckdb") as storage:
                case = build_clean_true_negative(storage, idx)
                shadow = shadow_capture(case.llm_input)
            _emit(case, shadow)
        for name, builder in _DECOY_CONSTRUCTORS:
            for idx in range(decoy_n):
                with isolated_pipeline(base / f"{name}-{idx}.duckdb") as storage:
                    case = builder(storage, idx)
                    shadow = shadow_capture(case.llm_input)
                _emit(case, shadow)
    return calls


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass
class Rate:
    k: int
    n: int
    estimable: bool = True
    label: str = ""

    @property
    def point(self) -> float | None:
        return (self.k / self.n) if self.n else None

    @property
    def wilson(self) -> tuple[float, float]:
        return wilson_interval(self.k, self.n)

    def render(self) -> str:
        if self.n == 0:
            return f"n=0 (no data){(' - ' + self.label) if self.label else ''}"
        lo, hi = self.wilson
        base = f"{self.point:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (k={self.k}, n={self.n})"
        if not self.estimable:
            return f"NOT ESTIMABLE - denominator undersized (n={self.n} < {FPR_MIN_DENOMINATOR}); " \
                   f"surface confirmed, rate withheld"
        return base + ((f"  - {self.label}") if self.label else "")


@dataclass
class ScoreReport:
    n_calls: int
    precision: Rate
    recall: Rate
    fpr: Rate
    band_precision: dict            # confidence -> Rate
    strategy_precision: dict        # method -> Rate
    mechanism_recall: dict          # mechanism -> Rate
    monotonic: bool
    monotonicity_detail: str
    adversarial_surface: dict       # kind -> #FPs induced
    preemption_masking: Rate        # Arm A behavioural masking rate
    preemption_correctness_pending: bool
    production_fp_shadow_tp: int    # the sharpest check (expect 0 if symmetric)
    out_of_scope_excluded: int
    coverage_note: str = ""


def score(calls: list[ScoredCall]) -> ScoreReport:
    in_scope = [c for c in calls if c.trichotomy != "out_of_scope"]
    oos = [c for c in calls if c.trichotomy == "out_of_scope"]

    emitted = [c for c in in_scope if c.fired]
    tp_links = [c for c in emitted if c.correct]
    precision = Rate(len(tp_links), len(emitted),
                     label="composition-dependent (S3): read per-band/strategy, not blended")

    positives = [c for c in in_scope if c.trichotomy == "linked_true"]
    recall_hits = [c for c in positives if c.correct]
    recall = Rate(len(recall_hits), len(positives), label="link-level (found a correct parent)")

    # S2: FPR denominator is the CLEAN-TN pool only, never the adversarial decoys.
    clean_neg = [c for c in in_scope if c.is_clean_negative]
    fpr_fps = [c for c in clean_neg if c.fired]
    fpr = Rate(len(fpr_fps), len(clean_neg),
               estimable=len(clean_neg) >= FPR_MIN_DENOMINATOR,
               label="representative clean-TN pool only (decoys excluded, S2)")

    # Per-band precision (calibration).
    band_precision: dict[float, Rate] = {}
    for b in sorted({c.fired_confidence for c in emitted}, reverse=True):
        cs = [c for c in emitted if c.fired_confidence == b]
        band_precision[b] = Rate(sum(1 for c in cs if c.correct), len(cs))

    # Per-strategy precision.
    strategy_precision: dict[str, Rate] = {}
    for m in sorted({c.fired_method for c in emitted}):
        cs = [c for c in emitted if c.fired_method == m]
        strategy_precision[m] = Rate(sum(1 for c in cs if c.correct), len(cs))

    # Per-mechanism recall (strategy-appropriate recall, §13.2).
    mechanism_recall: dict[str, Rate] = {}
    for mech in sorted({c.mechanism for c in positives}):
        cs = [c for c in positives if c.mechanism == mech]
        mechanism_recall[mech] = Rate(sum(1 for c in cs if c.correct), len(cs))

    # Monotonicity of calibration: precision non-increasing as confidence falls.
    ordered = [band_precision[b].point for b in sorted(band_precision, reverse=True)
               if band_precision[b].point is not None]
    monotonic = all(ordered[i] >= ordered[i + 1] - 1e-9 for i in range(len(ordered) - 1))
    monotonicity_detail = " >= ".join(f"{p:.2f}" for p in ordered) or "n/a"

    # Adversarial surface - reported SEPARATELY from FPR (S2).
    adversarial_surface: dict[str, int] = {}
    for c in in_scope:
        if c.is_adversarial and c.fired and not c.correct:
            adversarial_surface[c.mechanism] = adversarial_surface.get(c.mechanism, 0) + 1

    # Pre-emption - claim 1 (behavioural masking, Arm A high-N synthetic).
    collisions = [c for c in in_scope if c.mechanism == "strategy_collision"]
    masked = []
    for c in collisions:
        other = [p for (_n, p, _m, _conf) in c.isolated
                 if p is not None and p != c.fired_parent]
        if c.fired and other:
            masked.append(c)
    preemption_masking = Rate(len(masked), len(collisions),
                              label="behavioural: production masked a different fired parent")

    # The sharpest check: production's pick scored NOT-correct, but a shadow
    # strategy would have hit an acceptable parent. Expect 0 if symmetric
    # construction produced symmetric outcomes.
    production_fp_shadow_tp = 0
    for c in in_scope:
        if not c.correct:
            if any(p in c.acceptable_parents for (_n, p, _m, _conf) in c.isolated if p):
                production_fp_shadow_tp += 1

    return ScoreReport(
        n_calls=len(calls),
        precision=precision,
        recall=recall,
        fpr=fpr,
        band_precision=band_precision,
        strategy_precision=strategy_precision,
        mechanism_recall=mechanism_recall,
        monotonic=monotonic,
        monotonicity_detail=monotonicity_detail,
        adversarial_surface=adversarial_surface,
        preemption_masking=preemption_masking,
        preemption_correctness_pending=True,
        production_fp_shadow_tp=production_fp_shadow_tp,
        out_of_scope_excluded=len(oos),
        coverage_note="OUT_OF_SCOPE excluded from precision/recall/FPR per §3.1; "
                      "reported only as a coverage gap.",
    )
