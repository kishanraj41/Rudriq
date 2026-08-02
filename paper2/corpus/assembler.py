"""
Phase 2A · corpus assembler — compose seeded pipelines into a versioned,
serializable Arm A corpus that proves its own properties.

Two commitments, both baked into the *artifact* (not the harness code that
reads it), because the whole build's principle is that the artifact should
prove its own properties rather than require trust in the code:

1. **Byte-identical reproducibility.** The corpus is a deterministic
   function of (constructor set, ``n_per_class``). It carries no timestamps,
   no ``id()`` values, no sampled randomness — every field is a pure function
   of ``(constructor, idx)``. Serialized canonically (sorted keys), the same
   config reproduces the same bytes, so a reviewer can re-run and diff.
   :func:`verify_assembler` proves this by assembling twice and diffing.

2. **Arm→pool mapping in the artifact.** Each entry carries its scoring
   ``pool``; the corpus header carries an explicit ``pool_mapping`` declaring
   that Arm A *provides* the LINKED-TRUE pool and that the TRUE-NEGATIVE /
   OUT-OF-SCOPE pools are sourced from Arm C. The scoring harness reads the
   pool assignment from the corpus, and a reviewer inspecting the serialized
   file sees it without reading the harness.

Self-proving: assembly runs each pipeline's gate *as part of assembly* and
**refuses to produce a corpus if any gate fails** — a shipped corpus is, by
construction, one in which every pipeline tests what it claims.

Re-materialization: entries store ``(constructor, idx)``; the harness
re-invokes the same builder to reproduce the identical live pipeline. The
builder registry below is the single source of truth for that mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from paper2.corpus.gate import gate
from paper2.corpus.mechanisms import (
    ARM_A_CONSTRUCTORS,
    ConstructedCase,
    DuckDBStorage,
    Trichotomy,
    build_strategy_collision,
)
from paper2.corpus.primitives import isolated_pipeline, shadow_capture

CORPUS_VERSION = "arm-a/0.1.0"

# Trichotomy -> scoring pool. Arm A only ever produces LINKED_TRUE; the
# mapping is explicit (not implicit) so it can be serialized and audited.
_POOL_OF_TRICHOTOMY = {
    Trichotomy.LINKED_TRUE: "linked_true",
    Trichotomy.TRUE_NEGATIVE: "true_negative",
    Trichotomy.OUT_OF_SCOPE: "out_of_scope",
}

# The explicit arm->pool commitment, serialized into every corpus header.
POOL_MAPPING = {
    "arm": "A",
    "role": "positive/control",
    "provides_pools": ["linked_true"],
    "true_negative_pool_source": "arm_c",
    "out_of_scope_pool_source": "arm_c",
    "note": (
        "Arm A is the LINKED-TRUE positive/control arm. The negative pools "
        "(true_negative, out_of_scope) that FPR/specificity require are "
        "sourced from Arm C, never from Arm A. The labeler fixture in "
        "mechanisms.LABELER_FIXTURES is not corpus membership."
    ),
}


def _collision_symmetric(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    # Arm A collisions are symmetric (primary_source=None): the finding is
    # behavioural ("what gets masked"), never a synthetic correctness claim.
    return build_strategy_collision(storage, idx, designate_primary=False)


# The Arm A corpus: the five LINKED-TRUE mechanism classes plus the symmetric
# strategy-collision. The single source of truth for re-materialization.
ARM_A_FULL: tuple[tuple[str, Callable[[DuckDBStorage, int], ConstructedCase]], ...] = (
    ARM_A_CONSTRUCTORS + (("strategy_collision", _collision_symmetric),)
)

CONSTRUCTOR_REGISTRY: dict[str, Callable[[DuckDBStorage, int], ConstructedCase]] = dict(
    ARM_A_FULL
)


@dataclass(frozen=True)
class CorpusEntry:
    case_id: str
    constructor: str
    idx: int
    mechanism_class: str | None
    trichotomy: str
    pool: str
    true_parent_id: str | None
    collision_parents: list[list[str]]
    primary_source: str | None
    expected_method: str | None
    expected_confidences: list[float]
    gate_regime: str
    gate_passed: bool
    rendered_input: Any
    notes: str


@dataclass
class Corpus:
    corpus_version: str
    arm: str
    config: dict
    pool_mapping: dict
    entries: list[CorpusEntry] = field(default_factory=list)


class CorpusGateError(RuntimeError):
    """Raised when a pipeline fails its gate during assembly — a corpus with a
    failing gate must never ship."""


def _entry_from_case(
    constructor: str, idx: int, case: ConstructedCase, regime: str, passed: bool,
) -> CorpusEntry:
    return CorpusEntry(
        case_id=case.case_id,
        constructor=constructor,
        idx=idx,
        mechanism_class=case.mechanism_class.value if case.mechanism_class else None,
        trichotomy=case.trichotomy.value,
        pool=_POOL_OF_TRICHOTOMY[case.trichotomy],
        true_parent_id=case.true_parent_id,
        collision_parents=[list(p) for p in case.collision_parents],
        primary_source=case.primary_source,
        expected_method=case.expected_method,
        expected_confidences=list(case.expected_confidences),
        gate_regime=regime,
        gate_passed=passed,
        rendered_input=case.llm_input,
        notes=case.notes,
    )


def assemble(n_per_class: int = 3, tmp_dir: Path | str | None = None) -> Corpus:
    """Build the Arm A corpus, running each pipeline's gate as part of
    assembly. Raises :class:`CorpusGateError` on any gate failure.

    Deterministic in ``(ARM_A_FULL order, n_per_class)`` — no timestamps, no
    randomness. ``tmp_dir`` is only the scratch location for per-pipeline
    isolated DuckDBs; it does not affect the serialized output.
    """
    import tempfile

    corpus = Corpus(
        corpus_version=CORPUS_VERSION,
        arm="A",
        config={
            "n_per_class": n_per_class,
            "constructors": [name for name, _ in ARM_A_FULL],
        },
        pool_mapping=POOL_MAPPING,
    )

    ctx = tempfile.TemporaryDirectory(prefix="rudriq_assemble_")
    base = Path(tmp_dir) if tmp_dir is not None else Path(ctx.name)
    try:
        for constructor, builder in ARM_A_FULL:
            for idx in range(n_per_class):
                db = base / f"{constructor}-{idx}.duckdb"
                with isolated_pipeline(db) as storage:
                    case = builder(storage, idx)
                    shadow = shadow_capture(case.llm_input)
                    verdict = gate(case, shadow)
                if not verdict.passed:
                    failed = [c[0] for c in verdict.failed_checks()]
                    raise CorpusGateError(
                        f"{case.case_id}: gate failed ({verdict.regime}); "
                        f"checks={failed}. Corpus not shippable."
                    )
                corpus.entries.append(
                    _entry_from_case(constructor, idx, case, verdict.regime, verdict.passed)
                )
    finally:
        ctx.cleanup()

    return corpus


def to_canonical_json(corpus: Corpus) -> str:
    """Serialize to canonical JSON: sorted keys, stable separators, ASCII.

    Byte-identical for identical input. Entry *order* is the deterministic
    generation order (constructor order × idx); key order within objects is
    sorted. Together these make the artifact diff-stable across runs.
    """
    payload = {
        "corpus_version": corpus.corpus_version,
        "arm": corpus.arm,
        "config": corpus.config,
        "pool_mapping": corpus.pool_mapping,
        "entries": [
            {
                "case_id": e.case_id,
                "constructor": e.constructor,
                "idx": e.idx,
                "mechanism_class": e.mechanism_class,
                "trichotomy": e.trichotomy,
                "pool": e.pool,
                "true_parent_id": e.true_parent_id,
                "collision_parents": e.collision_parents,
                "primary_source": e.primary_source,
                "expected_method": e.expected_method,
                "expected_confidences": e.expected_confidences,
                "gate_regime": e.gate_regime,
                "gate_passed": e.gate_passed,
                "rendered_input": e.rendered_input,
                "notes": e.notes,
            }
            for e in corpus.entries
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True)
