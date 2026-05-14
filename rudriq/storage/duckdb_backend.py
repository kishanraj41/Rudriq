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
from datetime import datetime, timezone
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
    ensure_utc,
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      VARCHAR PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL,
    metadata    JSON
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id       VARCHAR PRIMARY KEY,
    run_id        VARCHAR NOT NULL,
    kind          VARCHAR NOT NULL,
    library       VARCHAR NOT NULL,
    operation     VARCHAR NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL,
    ended_at      TIMESTAMPTZ,
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


def _from_db(dt: datetime | None) -> datetime | None:
    """Ensure datetimes returned from DuckDB are UTC-aware.

    DuckDB TIMESTAMPTZ values are returned in the system local timezone
    (the instant is preserved, but the tzinfo is local). We normalize to
    UTC so downstream code never has to think about it. Naive datetimes
    are also caught — those would only appear if the schema regressed
    back to TIMESTAMP."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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

    def ensure_run(
        self,
        run_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Ensure a run row exists, but never touch its nodes or edges.

        Idempotent. Calling ensure_run on an already-existing run_id is a
        no-op — the metadata argument is only honored on the FIRST call
        that creates the row. To update metadata on an existing run, use
        update_run_metadata (not yet implemented; v0.1).

        Called by long-running components (the SpanProcessor, AutoLineage
        hooks) that incrementally append nodes and edges over the lifetime
        of a single run. Unlike replace_run, this is purely additive:
        existing nodes and edges under run_id are untouched.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?)",
                [
                    run_id,
                    datetime.now(timezone.utc),
                    json.dumps(metadata or {}),
                ],
            )

    def replace_run(self, graph: TraceGraph) -> None:
        """
        Authoritatively replace all state for graph.run_id.

        DESTRUCTIVE: deletes any existing edges under this run_id before
        inserting the new ones. Nodes are upserted (INSERT OR REPLACE) so
        nodes from prior writes that aren't in the new graph are NOT
        deleted — they remain orphaned. Use this when you have a complete
        graph in hand and want to make storage match it; use ensure_run
        + save_node + save_edge for incremental writes.
        """
        with self._lock:
            self._conn.execute("BEGIN TRANSACTION")
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                    [
                        graph.run_id,
                        ensure_utc(graph.created_at, source="TraceGraph.created_at"),
                        json.dumps(graph.metadata),
                    ],
                )
                # Idempotency: clear edges for this run before re-inserting.
                # Nodes use INSERT OR REPLACE (PK is node_id) so they're already
                # idempotent. Edges have no natural primary key (a node can have
                # multiple edges to the same parent with different metadata),
                # so we delete-then-insert.
                self._conn.execute(
                    "DELETE FROM edges WHERE run_id = ?",
                    [graph.run_id],
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
                            ensure_utc(node.started_at, source="TraceNode.started_at (replace_run)"),
                            ensure_utc(node.ended_at, source="TraceNode.ended_at (replace_run)"),
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

    def save_run(self, graph: TraceGraph) -> None:
        """
        DEPRECATED: use replace_run for destructive writes, or
        ensure_run + save_node + save_edge for incremental writes.

        Retained as a thin alias for backward compatibility. Will be
        removed in v0.1.
        """
        import warnings
        warnings.warn(
            "save_run is deprecated; use replace_run for destructive "
            "writes or ensure_run + save_node + save_edge for incremental.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.replace_run(graph)

    def save_node(self, node: TraceNode, run_id: str) -> None:
        """Append a single node to an existing or implicit run."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO runs VALUES (?, ?, ?)",
                [run_id, datetime.now(timezone.utc), "{}"],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    node.node_id,
                    run_id,
                    node.kind.value,
                    node.library,
                    node.operation,
                    ensure_utc(node.started_at, source="TraceNode.started_at (save_node)"),
                    ensure_utc(node.ended_at, source="TraceNode.ended_at (save_node)"),
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
                created_at=_from_db(row[1]),
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
                        started_at=_from_db(nr[4]),
                        ended_at=_from_db(nr[5]),
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
        """
        Return [(run_id, node_id), ...] of nodes with this content hash,
        ordered most-recent-first.

        The linker takes the latest match when correlating an LLM call's
        input to upstream data. Without explicit ordering, DuckDB makes
        no guarantee, so we sort by started_at descending.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT run_id, node_id
                FROM nodes
                WHERE content_hash = ?
                ORDER BY started_at DESC
                """,
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
