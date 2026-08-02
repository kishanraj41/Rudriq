"""
Phase 2A · Layer 2 — mechanism-class constructors (Arm A, LINKED-TRUE).

Five constructors, one per §13.2 mechanism class, each building a pipeline
whose true ``(data -> LLM call)`` edge is known *by construction*. Each
emits a :class:`ConstructedCase` carrying:

* the **mechanism-class tag** (construction-truth — how the dataflow was
  built, independent of what the linker does), and
* the **trichotomy label** (§3.1) — here always ``LINKED_TRUE`` for the
  five mechanism classes, with one ``TRUE_NEGATIVE`` constructor included
  *solely to exercise and prove the labeler* (the negative pool proper is
  Arm C's job).

The constructors encode the two construction disciplines locked at the
layer-1 checkpoint, with evidence:

* **exact-copy is storage-surface-exclusive** — persisted via the storage
  surface and asserted (construction-time) to have never touched the
  identity surface, so its isolation-silence gate is enforceable.
* **element-of-collection uses explicit per-element registration**, never
  slice-alignment — because ``register_object_identity`` registers only
  ``id(obj)`` and ``id(obj[0])``, so a non-zero-aligned slice silently
  fails to fire 0.95 (probe row ``b2``). Explicit-element registration is
  position-independent *and* mirrors AutoLineage's real per-element
  population, so these pipelines behave like real ones.

The substring matcher is **full containment** (one string wholly inside
the other), length-floored at 20 chars — *not* n-gram overlap. So
templated-containment embeds the source verbatim (the RAG pattern) and
paraphrased-derived simply avoids full containment (any genuine reword).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from rudriq.linker import register_object_identity

from paper2.corpus.primitives import (
    DuckDBStorage,
    identity_surface_touched,
    register_identity_surface,
    register_storage_surface,
)


class MechanismClass(str, Enum):
    """How the dataflow was constructed — construction-truth, never derived
    from which strategy fired."""

    VERBATIM_OBJECT = "verbatim_object"
    ELEMENT_OF_COLLECTION = "element_of_collection"
    EXACT_COPY = "exact_copy"
    TEMPLATED_CONTAINMENT = "templated_containment"
    PARAPHRASED_DERIVED = "paraphrased_derived"
    STRATEGY_COLLISION = "strategy_collision"


class Trichotomy(str, Enum):
    """§3.1 ground-truth class. Drives scoring (recall vs FPR pools), not the
    construction-consistency gate."""

    LINKED_TRUE = "linked_true"
    TRUE_NEGATIVE = "true_negative"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class ConstructedCase:
    """A single benchmark pipeline with ground truth known by construction.

    ``true_parent_id`` is the node the LLM call's input genuinely derives
    from (``None`` for TRUE_NEGATIVE — no edge exists). ``expected_method``
    / ``expected_confidences`` express what a *correct* linker should emit;
    ``None`` / ``()`` mean "should stay silent" (paraphrased-derived, which
    is a true edge the live strategies cannot recover — an FN by design —
    and true-negative, where silence is correct).
    """

    case_id: str
    mechanism_class: MechanismClass | None
    trichotomy: Trichotomy
    true_parent_id: str | None
    llm_input: Any
    expected_method: str | None
    expected_confidences: tuple[float, ...]
    notes: str = ""
    # exact-copy records its storage-exclusive invariant result so the gate
    # can surface it as an explicit check rather than an implicit assumption.
    invariants: dict[str, bool] = field(default_factory=dict)

    # Strategy-collision only (§13.4). ``collision_parents`` records, per
    # strategy expected to fire in isolation, the DISTINCT parent it should
    # resolve to: a tuple of ``(method, parent_id)``. ``primary_source`` is
    # the optional designation of which parent is the genuine source (the
    # "more correct" link) when the construction is asymmetric; ``None`` when
    # the collision is symmetric (both edges equally true and the finding is
    # purely behavioural — "which parent gets masked"). This lets the harness
    # express both the behavioural framing ("what precedence masks") and the
    # correctness framing ("precedence sometimes masks the better link").
    collision_parents: tuple[tuple[str, str], ...] = ()
    primary_source: str | None = None


# ---------------------------------------------------------------------------
# Deterministic seeds (no randomness — §5). Built at call time so each is a
# distinct Python object (distinct id()), which object-identity needs.
# ---------------------------------------------------------------------------


def _doc(idx: int) -> str:
    return (
        f"Record {idx}: quarterly revenue for business unit {idx} reached "
        f"{1000 + idx * 7} thousand units across all reporting regions."
    )


def _filler(idx: int, k: int) -> str:
    return f"Unrelated filler passage {idx}-{k} with sufficient length to scan."


def _alt_doc(idx: int) -> str:
    return (
        f"Memo {idx}: supply-chain lead times shortened by {3 + idx} days "
        f"following the logistics consolidation across the northern hubs."
    )


_DIGIT_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _source_value(idx: int) -> int:
    """The distinguishing datum carried by the source record."""
    return 1420 + idx


def _spell_digits(n: int) -> str:
    """Digit-by-digit transcoding: 1421 -> 'one four two one'. Injective in
    ``n`` and lexically disjoint from the decimal form (no full containment
    in either direction), so it is a genuine derivation the substring matcher
    cannot recover."""
    return " ".join(_DIGIT_WORDS[c] for c in str(n))


def _structured_source(idx: int) -> str:
    # Compact, machine-style encoding — the *data*. Carries the value as
    # decimal digits (``raw_value=1421``).
    return f"metric_id=Q{idx}REV;raw_value={_source_value(idx)};unit=usd_cents;flag=ok"


def _derived_prose(idx: int) -> str:
    # A derivation whose distinguishing content is a deterministic, injective
    # function of THIS source's value (the digits transcoded to words) — so
    # the prompt is provably bound to this source and distinct per source,
    # yet shares no verbatim surface with it. This is what makes the case a
    # genuine recall-ceiling FN (a real edge the linker cannot catch) rather
    # than two unrelated strings that merely happen not to contain each other.
    spelled = _spell_digits(_source_value(idx))
    return (
        f"The reporting period's headline figure, transcribed from the source "
        f"record, reads {spelled} when written out as a sequence of words."
    )


# ---------------------------------------------------------------------------
# Constructors — each takes the pipeline's isolated storage and returns a
# fully-constructed, ground-truthed case. Registrations happen here, inside
# the caller's isolated_pipeline context.
# ---------------------------------------------------------------------------


def build_verbatim_object(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    """Input *is* the registered object → object_identity @ 1.0."""
    node_id = f"vo-{idx}"
    doc = _doc(idx)
    register_identity_surface(doc, node_id)
    return ConstructedCase(
        case_id=node_id,
        mechanism_class=MechanismClass.VERBATIM_OBJECT,
        trichotomy=Trichotomy.LINKED_TRUE,
        true_parent_id=node_id,
        llm_input=doc,  # same object
        expected_method="object_identity",
        expected_confidences=(1.0,),
        notes="input is the registered object; substring co-fires in isolation (accepted).",
    )


def build_element_of_collection(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    """A registered element sits *inside* a fresh collection → object_identity
    @ 0.95. Explicit-element registration, element placed at a NON-zero index
    to prove position-independence (the locked correction)."""
    node_id = f"eoc-{idx}"
    elem = _doc(idx)
    register_identity_surface(elem, node_id)  # explicit element, not a parent slice
    # elem at index 1 (non-zero) — would silently fail under slice-alignment.
    llm_input = [_filler(idx, 0), elem, _filler(idx, 1)]
    return ConstructedCase(
        case_id=node_id,
        mechanism_class=MechanismClass.ELEMENT_OF_COLLECTION,
        trichotomy=Trichotomy.LINKED_TRUE,
        true_parent_id=node_id,
        llm_input=llm_input,
        expected_method="object_identity",
        expected_confidences=(0.95,),
        notes="registered element at non-zero index; proves explicit-element registration.",
    )


def build_exact_copy(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    """A content-equal but identity-distinct copy → content_hash @ 0.8.
    Storage-surface-exclusive: the construction-time invariant asserts the
    identity surface was never touched, so isolation-silence holds."""
    node_id = f"ec-{idx}"
    data = _doc(idx)
    register_storage_surface(data, node_id, storage)  # storage surface ONLY
    # Hard construction-time invariant: identity surface must be untouched.
    touched = identity_surface_touched(data, node_id)
    if touched:
        raise AssertionError(
            f"exact-copy {node_id}: identity surface was touched; storage-"
            f"exclusivity broken — substring would become co-satisfiable."
        )
    # Content-equal, identity-distinct copy (fresh object, same characters).
    llm_input = "".join(list(data))
    return ConstructedCase(
        case_id=node_id,
        mechanism_class=MechanismClass.EXACT_COPY,
        trichotomy=Trichotomy.LINKED_TRUE,
        true_parent_id=node_id,
        llm_input=llm_input,
        expected_method="content_hash",
        expected_confidences=(0.8,),
        notes="storage-exclusive; only content_hash satisfiable (isolation-silence).",
        invariants={"storage_exclusive": not touched},
    )


def build_templated_containment(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    """The source doc is embedded verbatim in a chat prompt (the RAG pattern)
    → substring @ 0.5/0.7 via full containment."""
    node_id = f"tc-{idx}"
    doc = _doc(idx)
    register_identity_surface(doc, node_id)
    llm_input = [
        {"role": "system", "content": "Answer using only the provided context."},
        {"role": "user", "content": f"Context:\n{doc}\n\nQuestion: please summarise."},
    ]
    return ConstructedCase(
        case_id=node_id,
        mechanism_class=MechanismClass.TEMPLATED_CONTAINMENT,
        trichotomy=Trichotomy.LINKED_TRUE,
        true_parent_id=node_id,
        llm_input=llm_input,
        expected_method="substring",
        expected_confidences=(0.5, 0.7),
        notes="source fully contained in prompt; substring is the intended winner.",
    )


def build_paraphrased_derived(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    """A genuine derivation that shares no verbatim containment with its
    source → ALL live strategies correctly stay silent. A true LINKED_TRUE
    edge the current linker cannot recover: the recall ceiling, an FN by
    construction. Verified at construction time to have no full containment
    in either direction (the only thing that fires substring)."""
    node_id = f"pd-{idx}"
    source = _structured_source(idx)
    register_identity_surface(source, node_id)  # the linker *can* see it...
    prompt = _derived_prose(idx)  # ...but cannot match this derivation.

    value_digits = str(_source_value(idx))
    value_words = _spell_digits(_source_value(idx))

    # (i) Derivation-truth: the prompt's distinguishing content is a
    # deterministic, injective transcoding of THIS source's datum. The source
    # carries the value as digits; the prompt carries it as words. Asserting
    # both halves present, in their respective surfaces, proves the prompt
    # genuinely derived from this source (and, being injective in the value,
    # is distinct per source) — not a silent non-edge that merely fails to
    # link. This is the check that makes gate-green *sufficient* for this
    # class, not just necessary.
    if value_digits not in source:
        raise AssertionError(f"pd {node_id}: source missing its datum {value_digits!r}.")
    if value_words not in prompt:
        raise AssertionError(f"pd {node_id}: prompt missing derived datum {value_words!r}.")
    # (ii) Lexical disjointness of the transcoding — the derived surface is
    # absent from the source and vice versa, so no surface bridges them.
    if value_words in source or value_digits in prompt:
        raise AssertionError(f"pd {node_id}: transcoding leaked across surfaces.")
    # (iii) No full containment either direction — the only thing that fires
    # substring. If this fails the "paraphrase" is secretly a substring case.
    if source in prompt or prompt in source:
        raise AssertionError(
            f"paraphrased-derived {node_id}: source/prompt have full "
            f"containment; substring would fire — not a clean paraphrase."
        )
    return ConstructedCase(
        case_id=node_id,
        mechanism_class=MechanismClass.PARAPHRASED_DERIVED,
        trichotomy=Trichotomy.LINKED_TRUE,  # a true edge...
        true_parent_id=node_id,             # ...to this parent...
        llm_input=prompt,
        expected_method=None,               # ...that the linker MISSES (FN).
        expected_confidences=(),
        notes=(
            "recall ceiling: construction-provable derivation (digit->word "
            "transcoding of the source datum), injective per source, FN by "
            "design. Free-form semantic paraphrase (derivation established by "
            "annotation, not construction) is Arm B's job."
        ),
    )


def build_true_negative(storage: DuckDBStorage, idx: int) -> ConstructedCase:
    """Labeler exercise: a genuinely-new input with NO upstream parent in
    scope. Registries are non-empty (an unrelated doc is registered, as in a
    real pipeline), but the call's input derives from none of it → silence is
    *correct* (TN, not FN). Proves the labeler distinguishes this from
    paraphrased-derived: identical linker behaviour (silence), opposite
    scoring meaning."""
    distractor_id = f"tn-distractor-{idx}"
    register_identity_surface(_doc(idx), distractor_id)  # unrelated upstream
    fresh_query = f"What is the weather forecast for delivery region {idx} tomorrow?"
    return ConstructedCase(
        case_id=f"tn-{idx}",
        mechanism_class=None,
        trichotomy=Trichotomy.TRUE_NEGATIVE,
        true_parent_id=None,  # no edge exists
        llm_input=fresh_query,
        expected_method=None,
        expected_confidences=(),
        notes="genuinely-new input; silence is correct (TN). Distinct from paraphrase FN.",
    )


def build_strategy_collision(
    storage: DuckDBStorage,
    idx: int,
    *,
    designate_primary: bool = False,
) -> ConstructedCase:
    """§13.4 — one input satisfying TWO strategies at DIFFERENT parents.

    Construction (the Proof-F pattern, now a first-class constructor):

    * ``doc_a`` is identity-registered to ``P_A`` with ``extract_content=
      False`` — so it cannot self-match via substring — and placed as an
      *element* of the input list (-> object_identity @ 0.95).
    * ``doc_b`` is identity-registered to ``P_B`` (content surface) and
      embedded *verbatim* in a sibling string of the same input (-> substring).

    Both are genuine edges by construction (the input really contains both).
    The load-bearing construction check is **P_A != P_B**: equal parents
    would be a co-satisfiable single-parent pipeline mislabelled as a
    collision. Production picks object_identity (higher precedence); the
    substring parent P_B is *masked* — visible only in the shadow. That mask
    is the raw material of the pre-emption finding.

    ``designate_primary`` toggles the framing (the question raised at the
    layer-3 checkpoint): when ``False`` the collision is **symmetric** (both
    edges equally true; the finding is purely behavioural — *what gets
    masked*). When ``True`` it is **asymmetric**, designating the *masked*
    substring parent ``P_B`` as the genuine ``primary_source`` — modelling
    "precedence masked the better link," a *correctness* cost, not just a
    behavioural one. Built both ways so the harness can express either; the
    corpus's choice of which to populate is the open question.
    """
    p_a, p_b = f"col-A-{idx}", f"col-B-{idx}"
    doc_a = _doc(idx)        # identity parent
    doc_b = _alt_doc(idx)    # substring parent
    if p_a == p_b or doc_a == doc_b:
        raise AssertionError(f"collision {idx}: parents/docs not distinct.")
    register_object_identity(doc_a, p_a, extract_content=False)  # identity ONLY
    register_identity_surface(doc_b, p_b)                        # substring surface
    # doc_a at a non-zero index (element match), doc_b embedded verbatim.
    llm_input = [doc_a, f"Reference note: {doc_b} (end of reference)."]

    primary = p_b if designate_primary else None
    return ConstructedCase(
        case_id=f"col-{idx}" + ("-asym" if designate_primary else "-sym"),
        mechanism_class=MechanismClass.STRATEGY_COLLISION,
        trichotomy=Trichotomy.LINKED_TRUE,
        true_parent_id=primary,  # None (symmetric) or the genuine source (P_B)
        llm_input=llm_input,
        expected_method="object_identity",   # production winner (precedence)
        expected_confidences=(0.95,),
        notes=(
            "collision: object_identity->P_A masks substring->P_B in production; "
            + ("asymmetric (P_B designated genuine source -> precedence masks "
               "the better link)" if designate_primary
               else "symmetric (both true; finding is what-gets-masked)")
        ),
        collision_parents=(("object_identity", p_a), ("substring", p_b)),
        primary_source=primary,
    )


_Builder = Callable[[DuckDBStorage, int], ConstructedCase]

# Arm A proper: the five LINKED-TRUE mechanism classes (ascending gate
# complexity). These are the positive/control arm.
ARM_A_CONSTRUCTORS: tuple[tuple[str, _Builder], ...] = (
    ("verbatim_object", build_verbatim_object),
    ("element_of_collection", build_element_of_collection),
    ("exact_copy", build_exact_copy),
    ("templated_containment", build_templated_containment),
    ("paraphrased_derived", build_paraphrased_derived),
)

# NOT Arm A membership: a fixture proving the trichotomy labeler distinguishes
# TN-silence from paraphrase FN-silence. The TRUE-NEGATIVE *pool* proper is
# Arm C's job; the corpus assembler must map Arm A -> LINKED_TRUE and the
# negative pool -> Arm C, never pulling TN from this fixture into Arm A.
LABELER_FIXTURES: tuple[tuple[str, _Builder], ...] = (
    ("true_negative", build_true_negative),
)

# What the layer-2 checkpoint exercises: Arm A + the labeler fixture.
CONSTRUCTORS: tuple[tuple[str, _Builder], ...] = ARM_A_CONSTRUCTORS + LABELER_FIXTURES
