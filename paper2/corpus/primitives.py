"""
Phase 2A · Layer 1 — pipeline primitives for the linker-accuracy corpus.

This is the *verified foundation*. It carries no mechanism-class
"difficulty" logic (that is layer 2); it only proves we can construct a
known ``(data -> LLM call)`` edge through the real linker and observe
*exactly* which strategy fires — the precondition for every claim Paper 2
makes.

Two registration surfaces (the §13.2 / correction-#2 distinction that is
invisible from the API names)
------------------------------------------------------------------------
* **Identity surface** — :func:`register_object_identity`. Populates the
  *in-process* ``_object_registry`` (feeds ``object_identity``) and, with
  ``extract_content=True``, ``_content_registry`` (feeds ``substring``).
* **Storage surface** — :meth:`DuckDBStorage.save_node` of a
  :class:`TraceNode` carrying ``content_hash``. ``link_by_content_hash``
  reads **persisted storage** via ``find_nodes_by_hash`` — *not* the
  in-process registries. A primitive that knew only the identity surface
  literally could not construct an exact-copy (content_hash) pipeline:
  content_hash would never fire and the gate would hard-fail with a
  confusing "nothing fired" instead of a clear cause.

Isolation (two leakage surfaces, not one)
------------------------------------------------------------------------
A scaling corpus can manufacture phantom links two ways:
* **Registry leakage** — a prior pipeline's ``id(obj) -> node_id`` entry
  survives into the next pipeline's ``correlate`` call.
* **Storage leakage** — a prior pipeline's persisted ``content_hash`` node
  is still visible to the next pipeline's ``find_nodes_by_hash`` lookup.
:func:`isolated_pipeline` closes *both*: it clears the in-process
registries **and** swaps in a fresh isolated DuckDB per pipeline.

Shadow capture
------------------------------------------------------------------------
:func:`shadow_capture` runs every strategy in isolation *and* production
``correlate``, recording what each strategy *would* have returned. The
strategy-collision constructor (layer 3) and the three-regime gate
(layer 4) both depend on it — so it lives here, at layer 1, built early.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from rudriq.core.schema import (
    LinkMethod,
    NodeKind,
    TraceNode,
    compute_content_hash,
)
from rudriq.linker import (
    clear_object_registry,
    correlate,
    link_by_content_hash,
    link_by_name_match,
    link_by_object_identity,
    link_by_substring,
    register_object_identity,
)
from rudriq.storage import duckdb_backend
from rudriq.storage.duckdb_backend import (
    DuckDBStorage,
    reset_default_storage_for_tests,
)

# Determinism (methodology §5): timestamps are *injected*, never sampled,
# so corpora diff byte-for-byte across runs. A single frozen instant is
# sufficient for layer 1 — node ordering within find_nodes_by_hash is not
# exercised by single-match constructions.
FROZEN_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# The production strategy order, named, for shadow capture. Mirrors
# linker._DEFAULT_STRATEGIES — kept explicit (not imported) so the shadow
# report is self-describing and a reordering of the production tuple is a
# visible diff here, not a silent behavioural change.
ISOLATED_STRATEGIES: tuple[tuple[str, Any], ...] = (
    ("object_identity", link_by_object_identity),
    ("content_hash", link_by_content_hash),
    ("substring", link_by_substring),
    ("name_match", link_by_name_match),
)


# ---------------------------------------------------------------------------
# Outcome records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyOutcome:
    """What one strategy returned for one input.

    ``method`` is the strategy's *own* declared method string (e.g.
    ``"object_identity"``); ``parent_id`` is ``None`` when the strategy did
    not fire. ``fired`` is the convenience predicate ``parent_id is not
    None``.
    """

    name: str
    parent_id: str | None
    method: str
    confidence: float

    @property
    def fired(self) -> bool:
        return self.parent_id is not None


@dataclass(frozen=True)
class ShadowResult:
    """The full per-strategy picture for one LLM input.

    ``production`` is what ``correlate`` (ordered, first-match-wins) emits —
    the number the product delivers. ``isolated`` is what each strategy
    returns *alone*, exposing pre-emption: a strategy that fires in
    isolation but is masked in production because a higher-precedence
    strategy fired first.
    """

    production: StrategyOutcome
    isolated: tuple[StrategyOutcome, ...]

    def isolated_by_name(self, name: str) -> StrategyOutcome:
        for outcome in self.isolated:
            if outcome.name == name:
                return outcome
        raise KeyError(name)

    @property
    def fired_in_isolation(self) -> tuple[StrategyOutcome, ...]:
        return tuple(o for o in self.isolated if o.fired)


# ---------------------------------------------------------------------------
# Registration surfaces
# ---------------------------------------------------------------------------


def register_identity_surface(obj: Any, node_id: str) -> None:
    """Identity surface: feeds ``object_identity`` and ``substring``.

    Thin wrapper over :func:`register_object_identity` named to make the
    *surface* explicit at every call site, so a constructor's choice of
    surface is auditable from the code alone. ``extract_content=True``
    (the default) is what populates the substring content registry.
    """
    register_object_identity(obj, node_id, extract_content=True)


def register_storage_surface(
    data: Any,
    node_id: str,
    storage: DuckDBStorage,
    *,
    kind: NodeKind = NodeKind.DATA_READ,
    library: str = "benchmark",
    operation: str = "seed",
    run_id: str = "benchmark-run",
    started_at: datetime = FROZEN_TS,
) -> str:
    """Storage surface: feeds ``content_hash`` *only*.

    Persists a :class:`TraceNode` whose ``content_hash`` is
    ``compute_content_hash(data)``. ``link_by_content_hash`` will later
    recover ``node_id`` when an LLM input hashes to the same value.

    Deliberately does **not** touch the identity surface — so for a clean
    exact-copy pipeline exactly one strategy is satisfiable and the
    isolation shadow stays unambiguous. Co-satisfiability would still let
    production pick content_hash by precedence, but single-satisfiability
    is the construction choice that keeps the per-strategy gate clean.

    Returns the content hash actually stored. Raises if ``data`` is not
    hashable by the linker (``compute_content_hash`` returned ``None``) —
    an unhashable seed cannot back a content_hash pipeline, and silently
    storing a ``NULL`` hash would later masquerade as "content_hash never
    fired."
    """
    content_hash = compute_content_hash(data)
    if content_hash is None:
        raise ValueError(
            f"data for node {node_id!r} is not hashable by the linker "
            f"(compute_content_hash returned None); it cannot back a "
            f"content_hash pipeline."
        )
    node = TraceNode(
        node_id=node_id,
        kind=kind,
        library=library,
        operation=operation,
        started_at=started_at,
        ended_at=started_at,
        content_hash=content_hash,
    )
    storage.save_node(node, run_id)
    return content_hash


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def new_isolated_storage(db_path: Path | str) -> DuckDBStorage:
    """Swap the linker's *default* storage for a fresh isolated DuckDB.

    Returns the new backend. Because ``link_by_content_hash`` resolves
    storage lazily via ``get_default_storage`` at call time, resetting the
    module singleton is sufficient for the content_hash strategy to read
    this isolated DB — no monkeypatching of the strategy needed.
    """
    return reset_default_storage_for_tests(db_path=db_path)


def identity_surface_touched(obj: Any, node_id: str) -> bool:
    """True if ``obj`` / ``node_id`` appears in the in-process identity or
    content registries.

    Used by the exact-copy constructor to assert its storage-exclusive
    invariant at *construction* time — so a future edit that accidentally
    routes exact-copy data through the identity surface (making substring
    co-satisfiable and breaking the class's isolation-silence gate) is
    caught immediately, not as a confusing gate failure later. Read
    without locks: the benchmark generator is single-threaded.
    """
    from rudriq import linker

    return id(obj) in linker._object_registry or node_id in linker._content_registry


@contextlib.contextmanager
def isolated_pipeline(db_path: Path | str) -> Iterator[DuckDBStorage]:
    """Enter a clean pipeline context: fresh registries *and* fresh storage.

    Closes both leakage surfaces (registry and storage) on entry. On exit,
    clears the in-process registries again so a caller that forgets to wrap
    the next pipeline cannot inherit this one's identity entries; the
    isolated DuckDB is closed.
    """
    clear_object_registry()
    storage = new_isolated_storage(db_path)
    try:
        yield storage
    finally:
        clear_object_registry()
        with contextlib.suppress(Exception):
            storage.close()


# ---------------------------------------------------------------------------
# Shadow capture
# ---------------------------------------------------------------------------


def shadow_capture(
    llm_input: Any,
    span_attributes: dict[str, Any] | None = None,
) -> ShadowResult:
    """Run every strategy in isolation *and* production ``correlate``.

    This is the primitive the collision constructor and the three-regime
    gate are built on: it is the only way to observe what a *masked*
    strategy would have returned, which is the entire basis of the
    strategy-pre-emption finding (§13.4).
    """
    attrs = span_attributes if span_attributes is not None else {}

    isolated: list[StrategyOutcome] = []
    for name, strategy in ISOLATED_STRATEGIES:
        parent_id, method, confidence = strategy(llm_input, attrs)
        isolated.append(
            StrategyOutcome(
                name=name,
                parent_id=parent_id,
                method=method,
                confidence=confidence,
            )
        )

    prod_parent, prod_method, prod_conf = correlate(llm_input, attrs)
    production = StrategyOutcome(
        name="production",
        parent_id=prod_parent,
        method=prod_method or LinkMethod.UNKNOWN.value,
        confidence=prod_conf,
    )

    return ShadowResult(production=production, isolated=tuple(isolated))
