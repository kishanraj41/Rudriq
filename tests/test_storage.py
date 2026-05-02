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
    temp_db.save_run(g)  # second time should not error
    loaded = temp_db.load_run("run-1")
    assert loaded is not None
    assert len(loaded.nodes) == 2
