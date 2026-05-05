"""
Tests for the AutoLineage record mirroring path (v0.0.6+).

When AutoLineage emits a TransformationRecord, RudriQ's post-record
callback persists a TraceNode in DuckDB and DIRECT edges for each
parent_id. These tests exercise that callback directly with synthetic
records (not driving real pandas) to keep the test fast and isolated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


pytest.importorskip(
    "autolineage",
    reason="autolineage not installed; mirroring tests require it",
)


from autolineage.core import TransformationRecord
from autolineage.core.tracker import UnifiedTracker

import rudriq.auto as _rudriq_auto
from rudriq.core.schema import EdgeKind, NodeKind
from rudriq.linker import _object_registry, clear_object_registry


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    from rudriq.storage import duckdb_backend

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "mirror_test.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    clear_object_registry()
    yield db
    db.close()


@pytest.fixture
def reset_run_id(monkeypatch):
    """Each test starts with mirroring's run_id reset to None."""
    monkeypatch.setattr(_rudriq_auto, "_autolineage_run_id", {"value": None})
    yield
    monkeypatch.setattr(_rudriq_auto, "_autolineage_run_id", {"value": None})


def _build_record(**overrides) -> TransformationRecord:
    base = dict(
        library="pandas",
        category="transform",
        operation="filter",
        child_id="lid-child",
        parent_ids=["lid-parent"],
        duration_ms=12.5,
        rows_before=1000,
        rows_after=250,
    )
    base.update(overrides)
    return TransformationRecord(**base)


def test_mirroring_skipped_when_run_id_unset(isolated_storage, reset_run_id):
    """When no SpanProcessor has set the autolineage run_id, the
    mirror callback must be a graceful no-op — no nodes saved."""
    al_tracker = UnifiedTracker()
    cb = _rudriq_auto._make_mirror_callback(al_tracker)

    cb(_build_record())

    # No run, no nodes, no edges anywhere in storage.
    assert isolated_storage.list_runs() == []


def test_mirroring_creates_node_when_run_id_set(isolated_storage, reset_run_id):
    """With a run_id wired, calling the callback persists a TraceNode."""
    _rudriq_auto._set_autolineage_run_id("mirror-test-1")

    al_tracker = UnifiedTracker()
    cb = _rudriq_auto._make_mirror_callback(al_tracker)

    rec = _build_record(child_id="al-1", library="pandas", operation="read_csv",
                        category="io", parent_ids=[])
    cb(rec)

    graph = isolated_storage.load_run("mirror-test-1")
    assert graph is not None
    assert len(graph.nodes) == 1
    n = graph.nodes[0]
    assert n.node_id == "al-1"
    assert n.kind == NodeKind.DATA_READ  # category="io" + read_csv → DATA_READ
    assert n.library == "pandas"
    assert n.operation == "read_csv"


def test_mirroring_creates_direct_edges_for_parents(isolated_storage, reset_run_id):
    """Each parent_id must produce a DIRECT edge parent->child."""
    _rudriq_auto._set_autolineage_run_id("mirror-test-2")

    al_tracker = UnifiedTracker()
    cb = _rudriq_auto._make_mirror_callback(al_tracker)

    cb(_build_record(child_id="child-1", parent_ids=["parent-A", "parent-B"]))

    graph = isolated_storage.load_run("mirror-test-2")
    assert graph is not None
    direct_edges = [e for e in graph.edges if e.kind == EdgeKind.DIRECT]
    edge_pairs = {(e.parent_id, e.child_id) for e in direct_edges}
    assert ("parent-A", "child-1") in edge_pairs
    assert ("parent-B", "child-1") in edge_pairs


def test_mirroring_io_writes_classified_as_data_write(
    isolated_storage, reset_run_id,
):
    """category=io + a 'to_*' operation should map to DATA_WRITE, not DATA_READ."""
    _rudriq_auto._set_autolineage_run_id("mirror-test-3")
    cb = _rudriq_auto._make_mirror_callback(UnifiedTracker())

    cb(_build_record(child_id="write-1", category="io", operation="to_csv"))

    graph = isolated_storage.load_run("mirror-test-3")
    assert graph.nodes[0].kind == NodeKind.DATA_WRITE


def test_mirroring_unknown_category_falls_back_to_unknown(
    isolated_storage, reset_run_id,
):
    """An unrecognized category should not crash; map to UNKNOWN."""
    _rudriq_auto._set_autolineage_run_id("mirror-test-4")
    cb = _rudriq_auto._make_mirror_callback(UnifiedTracker())

    cb(_build_record(child_id="weird-1", category="<unrecognized>",
                     operation="weird_op"))

    graph = isolated_storage.load_run("mirror-test-4")
    assert graph.nodes[0].kind == NodeKind.UNKNOWN


def test_mirroring_enriches_metadata_from_tracker_nodes(
    isolated_storage, reset_run_id,
):
    """When the AL tracker has node-level metadata (shape, columns), mirroring
    should fold it into the TraceNode's metadata under autolineage.* keys."""
    _rudriq_auto._set_autolineage_run_id("mirror-test-5")

    al_tracker = UnifiedTracker()
    al_tracker.nodes["enriched-1"] = {
        "id": "enriched-1",
        "source": "pandas.read_csv",
        "shape": (1000, 5),
        "columns": ["a", "b"],
        "content_hash": "ch-XYZ",
        "filepath": "docs.csv",
    }

    cb = _rudriq_auto._make_mirror_callback(al_tracker)
    cb(_build_record(child_id="enriched-1"))

    graph = isolated_storage.load_run("mirror-test-5")
    n = graph.nodes[0]
    assert n.metadata.get("autolineage.shape") == [1000, 5] or \
           n.metadata.get("autolineage.shape") == (1000, 5)  # JSON may flatten
    assert n.metadata.get("autolineage.content_hash") == "ch-XYZ"
    assert n.metadata.get("autolineage.filepath") == "docs.csv"
    assert n.content_hash == "ch-XYZ"


def test_mirroring_buggy_record_does_not_crash(isolated_storage, reset_run_id):
    """A malformed record (missing fields) must not raise — graceful degrade."""
    _rudriq_auto._set_autolineage_run_id("mirror-test-6")

    cb = _rudriq_auto._make_mirror_callback(UnifiedTracker())

    class BrokenRecord:
        # Missing several attributes the callback reads
        child_id = "broken-1"

    # Must not raise.
    cb(BrokenRecord())


def test_spanprocessor_init_wires_autolineage_run_id():
    """RudriQSpanProcessor.__init__ must call _set_autolineage_run_id so
    subsequent records mirror under its run_id."""
    from rudriq.processors import RudriQSpanProcessor

    _rudriq_auto._set_autolineage_run_id(None)
    proc = RudriQSpanProcessor(run_id="proc-driven-run")
    assert _rudriq_auto._autolineage_run_id["value"] == "proc-driven-run"


def test_started_at_naive_timestamp_interpreted_as_local_time(
    isolated_storage, reset_run_id,
):
    """Bug 6 regression: AutoLineage's TransformationRecord.timestamp is
    a NAIVE local-time string. The mirror callback used to attach UTC
    tzinfo directly via replace(), producing a stored started_at that
    was off by the system's UTC offset (e.g., 5h for Central Time).
    The visible symptom in the demo notebook was a fake 18-million-ms
    'duration' between the data node and the LLM node.

    The fix interprets naive timestamps as local time and converts to
    UTC. This test pins that behavior."""
    _rudriq_auto._set_autolineage_run_id("tz-bug-run")
    cb = _rudriq_auto._make_mirror_callback(UnifiedTracker())

    # Construct a record at "now" using AutoLineage's own dataclass
    # default (datetime.now().isoformat()), which mimics what the real
    # tracker produces.
    rec = _build_record(child_id="tz-1")  # default factory fires
    cb(rec)

    graph = isolated_storage.load_run("tz-bug-run")
    n = graph.nodes[0]

    # Saved started_at should be within 5 seconds of "now in UTC",
    # not offset by hours. (5s slack absorbs test flakiness.)
    now_utc = datetime.now(timezone.utc)
    delta = abs((n.started_at - now_utc).total_seconds())
    assert delta < 5, (
        f"started_at off by {delta:.1f}s — likely a TZ-conversion "
        f"regression. saved={n.started_at}, now_utc={now_utc}"
    )


def test_end_to_end_real_callback_wiring(isolated_storage, reset_run_id):
    """Wire the callback through the actual rudriq.auto path (using a fresh
    UnifiedTracker), then call tracker.record() and verify mirroring fires."""
    _rudriq_auto._set_autolineage_run_id("e2e-run")

    tracker = UnifiedTracker()
    # This is what _activate_autolineage does internally:
    tracker.register_post_record_callback(_rudriq_auto._make_mirror_callback(tracker))

    rec = _build_record(child_id="e2e-1", parent_ids=["e2e-parent"])
    tracker.record(rec)

    graph = isolated_storage.load_run("e2e-run")
    assert graph is not None
    assert any(n.node_id == "e2e-1" for n in graph.nodes)
    assert any(
        e.parent_id == "e2e-parent" and e.child_id == "e2e-1"
        and e.kind == EdgeKind.DIRECT
        for e in graph.edges
    )
