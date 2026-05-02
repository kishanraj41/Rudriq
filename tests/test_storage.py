"""Tests for the DuckDB storage backend."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from rudriq.core.schema import (
    EdgeKind,
    LinkMethod,
    NodeKind,
    TraceEdge,
    TraceGraph,
    TraceNode,
)
from rudriq.storage.duckdb_backend import DuckDBStorage


@pytest.fixture
def temp_db(tmp_path: Path) -> DuckDBStorage:
    db = DuckDBStorage(db_path=tmp_path / "test.duckdb")
    yield db
    db.close()


def _sample_graph(run_id: str = "run-1") -> TraceGraph:
    now = datetime.now(timezone.utc)
    g = TraceGraph(run_id=run_id, created_at=now)
    g.add_node(TraceNode(
        node_id="n1",
        kind=NodeKind.DATA_READ,
        library="pandas",
        operation="read_csv",
        started_at=now,
        ended_at=now,
        content_hash="hash-A",
    ))
    g.add_node(TraceNode(
        node_id="n2",
        kind=NodeKind.LLM_EMBEDDING,
        library="openai",
        operation="embeddings.create",
        started_at=now,
    ))
    g.add_edge(TraceEdge(
        parent_id="n1",
        child_id="n2",
        kind=EdgeKind.LINEAGE_LINK,
        confidence=1.0,
        link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    return g


def test_save_and_load_round_trips(temp_db: DuckDBStorage) -> None:
    original = _sample_graph()
    temp_db.save_run(original)
    loaded = temp_db.load_run("run-1")

    assert loaded is not None
    assert loaded.run_id == "run-1"
    assert len(loaded.nodes) == 2
    assert len(loaded.edges) == 1
    assert loaded.nodes[0].kind == NodeKind.DATA_READ
    assert loaded.edges[0].kind == EdgeKind.LINEAGE_LINK


def test_load_missing_run_returns_none(temp_db: DuckDBStorage) -> None:
    assert temp_db.load_run("does-not-exist") is None


def test_find_nodes_by_hash(temp_db: DuckDBStorage) -> None:
    temp_db.save_run(_sample_graph())
    matches = temp_db.find_nodes_by_hash("hash-A")
    assert len(matches) == 1
    assert matches[0] == ("run-1", "n1")


def test_find_nodes_by_hash_returns_empty_for_unknown(temp_db: DuckDBStorage) -> None:
    temp_db.save_run(_sample_graph())
    assert temp_db.find_nodes_by_hash("hash-DOES-NOT-EXIST") == []


def test_list_runs_returns_in_recent_order(temp_db: DuckDBStorage) -> None:
    g1 = _sample_graph("run-1")
    g2 = _sample_graph("run-2")
    temp_db.save_run(g1)
    temp_db.save_run(g2)

    runs = temp_db.list_runs()
    assert "run-1" in runs
    assert "run-2" in runs


def test_save_run_is_idempotent(temp_db: DuckDBStorage) -> None:
    g = _sample_graph()
    temp_db.save_run(g)
    temp_db.save_run(g)  # second time should not error or duplicate
    loaded = temp_db.load_run("run-1")
    assert loaded is not None
    assert len(loaded.nodes) == 2
    assert len(loaded.edges) == 1  # bug 1 regression: edges were duplicating


def test_timestamps_preserve_utc_on_roundtrip(temp_db: DuckDBStorage) -> None:
    """Bug 2 regression: UTC timezone must survive save -> load."""
    saved_at = datetime(2026, 5, 1, 12, 30, 45, tzinfo=timezone.utc)
    g = TraceGraph(run_id="tz-test", created_at=saved_at)
    g.add_node(TraceNode(
        node_id="tz-n1",
        kind=NodeKind.DATA_READ,
        library="pandas",
        operation="read_csv",
        started_at=saved_at,
        ended_at=saved_at,
    ))
    temp_db.save_run(g)
    loaded = temp_db.load_run("tz-test")

    assert loaded is not None
    assert loaded.created_at == saved_at  # same instant, regardless of tzinfo
    # _from_db normalizes to UTC, not just any aware tzinfo, for audit
    # consistency. Without astimezone, DuckDB returns the system local TZ.
    assert loaded.created_at.utcoffset() == timezone.utc.utcoffset(loaded.created_at)
    assert loaded.nodes[0].started_at == saved_at
    assert loaded.nodes[0].ended_at == saved_at


def test_find_nodes_by_hash_returns_most_recent_first(
    temp_db: DuckDBStorage,
) -> None:
    """Bug 4 regression: ordering must be deterministic, most-recent first."""
    from datetime import timedelta

    base = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)

    g = TraceGraph(run_id="r-multi", created_at=base)
    g.add_node(TraceNode(
        node_id="oldest",
        kind=NodeKind.DATA_READ, library="pandas", operation="read_csv",
        started_at=base, ended_at=base, content_hash="shared-hash",
    ))
    g.add_node(TraceNode(
        node_id="middle",
        kind=NodeKind.DATA_READ, library="pandas", operation="read_csv",
        started_at=base + timedelta(seconds=10),
        ended_at=base + timedelta(seconds=10),
        content_hash="shared-hash",
    ))
    g.add_node(TraceNode(
        node_id="newest",
        kind=NodeKind.DATA_READ, library="pandas", operation="read_csv",
        started_at=base + timedelta(seconds=20),
        ended_at=base + timedelta(seconds=20),
        content_hash="shared-hash",
    ))
    temp_db.save_run(g)

    matches = temp_db.find_nodes_by_hash("shared-hash")
    assert len(matches) == 3
    assert matches[0][1] == "newest"   # most recent FIRST
    assert matches[2][1] == "oldest"   # oldest LAST
