"""
DuckDB-based local storage for RudriQ trace graphs.

Default location: ~/.rudriq/traces.duckdb

Schema:
    runs(run_id PK, created_at, metadata)
    nodes(node_id PK, run_id FK, kind, library, operation,
          started_at, ended_at, metadata, content_hash)
    edges(parent_id, child_id, run_id FK, kind, confidence,
          link_method, metadata)
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from rudriq.core.schema import (
    EdgeKind,
    LinkMethod,
    NodeKind,
    TraceEdge,
    TraceGraph,
    TraceNode,
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,
    created_at  TIMESTAMP NOT NULL,
    metadata    JSON
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id       VARCHAR PRIMARY KEY,
    run_id        VARCHAR NOT NULL,
    kind          VARCHAR NOT NULL,
    library       VARCHAR NOT NULL,
    operation     VARCHAR NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    ended_at      TIMESTAMP,
    metadata      JSON,
    content_hash  VARCHAR
);

CREATE TABLE IF NOT EXISTS edges (
    edge_id       BIGINT PRIMARY KEY,
    parent_id     VARCHAR NOT NULL,
    child_id      VARCHAR NOT NULL,
    run_id        VARCHAR NOT NULL,
    kind          VARCHAR NOT NULL,
    confidence    DOUBLE NOT NULL,
    link_method   VARCHAR NOT NULL,
    metadata      JSON
);

CREATE INDEX IF NOT EXISTS idx_nodes_run ON nodes(run_id);
CREATE INDEX IF NOT EXISTS idx_edges_run ON edges(run_id);
CREATE INDEX IF NOT EXISTS idx_edges_child ON edges(child_id);
CREATE INDEX IF NOT EXISTS idx_nodes_hash ON nodes(content_hash);

CREATE SEQUENCE IF NOT EXISTS edge_id_seq START 1;
"""


def _default_db_path() -> Path:
    home = Path(os.environ.get("RUDRIQ_HOME", Path.home() / ".rudriq"))
    home.mkdir(parents=True, exist_ok=True)
    return home / "traces.duckdb"


class DuckDBStorage:
    """
    Embedded DuckDB-backed trace storage.

    Thread-safety: methods are guarded by a per-instance lock. DuckDB
    itself supports concurrent readers but not concurrent writers, so
    we serialize writes.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self._conn = duckdb.connect(str(self.db_path))
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(_SCHEMA_SQL)

    # -- writes --------------------------------------------------------

    def save_run(self, graph: TraceGraph) -> None:
        """Persist a full TraceGraph to DuckDB."""
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                    [
                        graph.run_id,
                        graph.created_at,
                        json.dumps(graph.metadata),
                    ],
                )
                for node in graph.nodes:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            node.node_id,
                            graph.run_id,
                            node.kind.value,
                            node.library,
                            node.operation,
                            node.started_at,
                            node.ended_at,
                            json.dumps(node.metadata),
                            node.content_hash,
                        ],
                    )
                for edge in graph.edges:
                    self._conn.execute(
                        """
                        INSERT INTO edges
                        VALUES (nextval('edge_id_seq'), ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            edge.parent_id,
                            edge.child_id,
                            graph.run_id,
                            edge.kind.value,
                            edge.confidence,
                            edge.link_method.value,
                            json.dumps(edge.metadata),
                        ],
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def save_node(self, node: TraceNode, run_id: str) -> None:
        """Append a single node to an existing or implicit run."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?)",
                [run_id, datetime.now(), "{}"],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    node.node_id,
                    run_id,
                    node.kind.value,
                    node.library,
                    node.operation,
                    node.started_at,
                    node.ended_at,
                    json.dumps(node.metadata),
                    node.content_hash,
                ],
            )

    def save_edge(self, edge: TraceEdge, run_id: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO edges
                VALUES (nextval('edge_id_seq'), ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    edge.parent_id,
                    edge.child_id,
                    run_id,
                    edge.kind.value,
                    edge.confidence,
                    edge.link_method.value,
                    json.dumps(edge.metadata),
                ],
            )

    # -- reads ---------------------------------------------------------

    def load_run(self, run_id: str) -> TraceGraph | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_id, created_at, metadata FROM runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
            if not row:
                return None

            metadata = json.loads(row[2]) if row[2] else {}
            graph = TraceGraph(
                run_id=row[0],
                created_at=row[1],
                metadata=metadata,
            )

            node_rows = self._conn.execute(
                """
                SELECT node_id, kind, library, operation, started_at, ended_at,
                       metadata, content_hash
                FROM nodes WHERE run_id = ?
                """,
                [run_id],
            ).fetchall()
            for nr in node_rows:
                graph.add_node(
                    TraceNode(
                        node_id=nr[0],
                        kind=NodeKind(nr[1]),
                        library=nr[2],
                        operation=nr[3],
                        started_at=nr[4],
                        ended_at=nr[5],
                        metadata=json.loads(nr[6]) if nr[6] else {},
                        content_hash=nr[7],
                    )
                )

            edge_rows = self._conn.execute(
                """
                SELECT parent_id, child_id, kind, confidence, link_method, metadata
                FROM edges WHERE run_id = ?
                """,
                [run_id],
            ).fetchall()
            for er in edge_rows:
                graph.add_edge(
                    TraceEdge(
                        parent_id=er[0],
                        child_id=er[1],
                        kind=EdgeKind(er[2]),
                        confidence=er[3],
                        link_method=LinkMethod(er[4]),
                        metadata=json.loads(er[5]) if er[5] else {},
                    )
                )
            return graph

    def find_nodes_by_hash(self, content_hash: str) -> list[tuple[str, str]]:
        """Return [(run_id, node_id), ...] of nodes with this content hash."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id, node_id FROM nodes WHERE content_hash = ?",
                [content_hash],
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

    def list_runs(self, limit: int = 100) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?",
                [limit],
            ).fetchall()
            return [r[0] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# Default singleton
# ---------------------------------------------------------------------------

_default: DuckDBStorage | None = None
_default_lock = threading.Lock()


def get_default_storage() -> DuckDBStorage:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = DuckDBStorage()
    return _default


def reset_default_storage_for_tests(db_path: Path | str | None = None) -> DuckDBStorage:
    """Test-only: reset the default singleton to a fresh DB."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
        _default = DuckDBStorage(db_path=db_path)
    return _default
