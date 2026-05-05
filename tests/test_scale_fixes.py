"""
Regression tests for the bugs surfaced by Day 8's realistic pipeline
(examples/realistic_rag_pipeline.py).

Each bug came from running ~250 ops at scale and observing what broke.
Unit tests with 4-node fixtures missed all five.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    from rudriq.storage import duckdb_backend
    from rudriq.linker import clear_object_registry
    from rudriq.processors.linking import clear_input_registry

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "scale.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    clear_object_registry()
    clear_input_registry()
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Bug 1: linker only checked first element of list inputs.
# At scale, batched embeddings pass slices like texts[50:100] whose first
# element is texts[50], not texts[0]. Walking only texts[0] would never
# match any element of slice [50:100]. Fix: walk ALL elements.
# ---------------------------------------------------------------------------


def test_linker_walks_all_list_elements_not_only_first(isolated_storage):
    from rudriq.linker import (
        link_by_object_identity,
        register_object_identity,
        clear_object_registry,
    )

    clear_object_registry()

    # Register element at index 50 only — index 0 is unregistered.
    elements = [object() for _ in range(100)]
    register_object_identity(elements[50], "elem-50-lid")

    # Pass a slice that does NOT contain index 0 — first element is
    # elements[50].
    parent_id, method, conf = link_by_object_identity(elements[40:60], {})
    assert parent_id == "elem-50-lid"
    assert method == "object_identity"
    assert conf == 0.95

    # Also a slice where matching element is NOT first.
    parent_id, _, _ = link_by_object_identity([elements[10], elements[50]], {})
    assert parent_id == "elem-50-lid"


# ---------------------------------------------------------------------------
# Bug 2: pd.Series.tolist propagation only registered the whole-list lid.
# After list slicing, a fresh slice list has no entry. Elements were not
# registered. Fix: tolist registers parent lid for EACH element.
# ---------------------------------------------------------------------------


def test_tolist_registers_each_element_against_parent_lid(isolated_storage):
    pytest.importorskip("pandas")
    import pandas as pd

    from rudriq.linker import (
        _object_registry,
        register_object_identity,
        clear_object_registry,
    )
    from rudriq.processors.auto_capture import install_pandas_lid_propagation

    clear_object_registry()

    orig_tolist = pd.Series.tolist
    orig_getitem = pd.DataFrame.__getitem__
    try:
        install_pandas_lid_propagation()

        s = pd.Series(["a", "b", "c", "d", "e"])
        register_object_identity(s, "series-PARENT")

        result = s.tolist()
        # Whole list registered.
        assert _object_registry.get(id(result)) == "series-PARENT"
        # Each element registered (the v0.0.7 fix).
        for elem in result:
            assert _object_registry.get(id(elem)) == "series-PARENT"
    finally:
        pd.Series.tolist = orig_tolist
        pd.DataFrame.__getitem__ = orig_getitem


# ---------------------------------------------------------------------------
# Bug 3: assign_id callback only registered identity, never saved a stub
# TraceNode. AL's read_csv hook fires assign_id but never calls record(),
# so read_csv outputs left no node in DuckDB.
# ---------------------------------------------------------------------------


def test_assign_id_callback_saves_stub_tracenode(isolated_storage, monkeypatch):
    """A direct test of the stub-callback factory; bypasses the pandas
    layer to keep the assertion focused."""
    pytest.importorskip("autolineage")
    from autolineage.core.tracker import UnifiedTracker

    import rudriq.auto as _auto

    monkeypatch.setattr(_auto, "_autolineage_run_id", {"value": "scale-test-3"})

    tracker = UnifiedTracker()
    cb = _auto._make_assign_id_stub_callback(tracker)

    sentinel = {"text": "doc"}
    lid = tracker.assign_id(sentinel, source="pandas.read_csv",
                            filepath="/tmp/example.csv")
    cb(sentinel, lid)

    graph = isolated_storage.load_run("scale-test-3")
    assert graph is not None
    matching = [n for n in graph.nodes if n.node_id == lid]
    assert len(matching) == 1, (
        "Expected stub TraceNode for assign_id; "
        "without this fix, read_csv outputs leave no DuckDB node."
    )
    n = matching[0]
    assert n.library == "pandas"
    assert n.operation == "read_csv"


# ---------------------------------------------------------------------------
# Bug 4: post_record callback never registered identity for record.child_id.
# Result: AL transforms that record() but skip assign_id for the result
# (sort_values, head, reset_index when attrs are preserved) had their
# outputs missing from rudriq's _object_registry.
# ---------------------------------------------------------------------------


def test_post_record_callback_registers_identity_via_lid_to_obj(
    isolated_storage, monkeypatch,
):
    pytest.importorskip("autolineage")
    from autolineage.core.tracker import UnifiedTracker
    from autolineage.core import TransformationRecord

    import rudriq.auto as _auto
    from rudriq.linker import _object_registry, clear_object_registry

    clear_object_registry()
    monkeypatch.setattr(_auto, "_autolineage_run_id", {"value": "scale-test-4"})

    tracker = UnifiedTracker()

    # Inject an object into AL's _lid_to_obj. AL stores via
    # WeakValueDictionary, which requires the value to support weak
    # references — lists, dicts, and tuples don't. Use a small custom
    # class that does (a stand-in for a tracked DataFrame/array/etc.).
    class _Tracked:
        pass

    payload = _Tracked()
    tracker._lid_to_obj["lid-via-record"] = payload

    cb = _auto._make_mirror_callback(tracker)
    cb(TransformationRecord(
        library="pandas",
        category="transform",
        operation="sort_values",
        child_id="lid-via-record",
        parent_ids=[],
        duration_ms=1.0,
    ))

    # The payload's id() should now be registered against the lid.
    assert _object_registry.get(id(payload)) == "lid-via-record"


# ---------------------------------------------------------------------------
# Bug 6: thread-local fallback for inputs when no active recording span
# exists at wrapper time. OpenLLMetry's openai instrumentor does not
# always present a recording span to inner wrappers — without this
# fallback, batched embeddings landed unlinked.
# ---------------------------------------------------------------------------


def test_set_recent_input_fallback_consumed_by_on_end(isolated_storage):
    from rudriq.processors.linking import (
        consume_recent_input_fallback,
        set_recent_input_fallback,
    )

    payload = {"input": "hello"}
    set_recent_input_fallback(payload)
    consumed = consume_recent_input_fallback()
    assert consumed is payload
    # Second consume returns None (single-shot semantics).
    assert consume_recent_input_fallback() is None


def test_processor_uses_thread_local_fallback_when_no_span_id(
    isolated_storage,
):
    """End-to-end of the fallback path: register an upstream object,
    set the thread-local fallback (simulating our wrapper firing
    outside an active span), emit a gen_ai.* span via the SDK
    TracerProvider, verify the linker still finds the upstream."""
    from opentelemetry.sdk.trace import TracerProvider

    from rudriq.linker import register_object_identity
    from rudriq.processors import RudriQSpanProcessor
    from rudriq.processors.linking import set_recent_input_fallback

    upstream = ["doc-A"]
    register_object_identity(upstream, "upstream-A")

    set_recent_input_fallback(upstream)

    provider = TracerProvider()
    proc = RudriQSpanProcessor(run_id="scale-test-6")
    provider.add_span_processor(proc)
    tracer = provider.get_tracer("scale-test")

    with tracer.start_as_current_span("openai.embeddings.create") as span:
        span.set_attribute("gen_ai.system", "openai")

    graph = isolated_storage.load_run("scale-test-6")
    assert graph is not None
    lineage_edges = [e for e in graph.edges if e.kind.value == "lineage"]
    assert len(lineage_edges) == 1
    assert lineage_edges[0].parent_id == "upstream-A"


# ---------------------------------------------------------------------------
# Performance sanity (exporter remains fast at moderate scale).
# ---------------------------------------------------------------------------


def test_audit_export_fast_for_moderate_graph(isolated_storage):
    """Sanity check: 100-node graph exports JSON in under 200ms.
    Catches accidental quadratic blowups in the exporter."""
    import time
    from datetime import datetime, timedelta, timezone
    from rudriq.core.schema import (
        EdgeKind, LinkMethod, NodeKind, TraceEdge, TraceGraph, TraceNode,
    )
    from rudriq.export.audit import export_audit_json

    base = datetime(2026, 5, 5, tzinfo=timezone.utc)
    g = TraceGraph(run_id="perf-test", created_at=base)
    for i in range(100):
        g.add_node(TraceNode(
            node_id=f"n{i:03d}",
            kind=NodeKind.DATA_TRANSFORM if i % 2 else NodeKind.LLM_CHAT,
            library="pandas" if i % 2 else "openai",
            operation="op",
            started_at=base + timedelta(milliseconds=i),
        ))
        if i > 0:
            g.add_edge(TraceEdge(
                parent_id=f"n{i-1:03d}", child_id=f"n{i:03d}",
                kind=EdgeKind.DIRECT, confidence=1.0,
                link_method=LinkMethod.UNKNOWN,
            ))
    isolated_storage.replace_run(g)

    t0 = time.time()
    out = export_audit_json("perf-test")
    dt = time.time() - t0

    assert dt < 0.2, f"audit export took {dt*1000:.0f}ms for 100 nodes"
    assert len(out) > 0
