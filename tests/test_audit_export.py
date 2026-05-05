"""Tests for audit JSON and Markdown exporters."""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
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
def temp_db(tmp_path: Path, monkeypatch):
    """Each test gets its own DuckDB and a fresh default singleton."""
    from rudriq.storage import duckdb_backend

    db = DuckDBStorage(db_path=tmp_path / "audit_test.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    yield db
    db.close()


def _build_rag_graph(run_id: str = "audit-rag-1") -> TraceGraph:
    """A small but realistic RAG-like graph for export testing."""
    base = datetime(2026, 5, 2, 14, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id=run_id, created_at=base)

    g.add_node(TraceNode(
        node_id="data-read",
        kind=NodeKind.DATA_READ, library="pandas", operation="read_csv",
        started_at=base, ended_at=base + timedelta(milliseconds=10),
        metadata={"path": "docs.csv", "rows": 1000},
    ))
    g.add_node(TraceNode(
        node_id="data-filter",
        kind=NodeKind.DATA_TRANSFORM, library="pandas", operation="filter",
        started_at=base + timedelta(milliseconds=11),
        ended_at=base + timedelta(milliseconds=15),
        metadata={"rows_in": 1000, "rows_out": 250},
    ))
    g.add_node(TraceNode(
        node_id="llm-embed",
        kind=NodeKind.LLM_EMBEDDING, library="openai",
        operation="embeddings.create",
        started_at=base + timedelta(milliseconds=20),
        ended_at=base + timedelta(milliseconds=120),
        metadata={"gen_ai.request.model": "text-embedding-3-small"},
    ))
    g.add_node(TraceNode(
        node_id="llm-chat",
        kind=NodeKind.LLM_CHAT, library="openai",
        operation="chat.completions.create",
        started_at=base + timedelta(milliseconds=130),
        ended_at=base + timedelta(milliseconds=2130),
        metadata={"gen_ai.request.model": "gpt-4"},
    ))

    g.add_edge(TraceEdge(
        parent_id="data-read", child_id="data-filter",
        kind=EdgeKind.DIRECT, confidence=1.0,
        link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    g.add_edge(TraceEdge(
        parent_id="data-filter", child_id="llm-embed",
        kind=EdgeKind.LINEAGE_LINK, confidence=1.0,
        link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    g.add_edge(TraceEdge(
        parent_id="data-filter", child_id="llm-chat",
        kind=EdgeKind.LINEAGE_LINK, confidence=0.8,
        link_method=LinkMethod.CONTENT_HASH,
    ))
    return g


def test_export_json_returns_schema_versioned_string(temp_db):
    from rudriq.export.audit import export_audit_json, AUDIT_SCHEMA_VERSION

    temp_db.replace_run(_build_rag_graph())
    out = export_audit_json("audit-rag-1")
    parsed = json.loads(out)

    assert parsed["schema_version"] == AUDIT_SCHEMA_VERSION
    assert "summary" in parsed
    assert "run" in parsed
    assert "lineage_chains" in parsed


def test_export_json_summary_counts_match_graph(temp_db):
    from rudriq.export.audit import export_audit_json

    temp_db.replace_run(_build_rag_graph())
    parsed = json.loads(export_audit_json("audit-rag-1"))
    s = parsed["summary"]

    assert s["total_nodes"] == 4
    assert s["data_nodes"] == 2
    assert s["llm_nodes"] == 2
    assert s["other_nodes"] == 0
    assert s["total_edges"] == 3
    assert s["direct_edges"] == 1
    assert s["lineage_links"] == 2
    assert s["linked_llm_calls"] == 2
    assert s["unlinked_llm_calls"] == 0
    assert s["libraries_seen"] == ["openai", "pandas"]


def test_export_json_is_deterministic(temp_db, monkeypatch):
    """Same graph + pinned generated_at → byte-identical output."""
    from rudriq.export import audit

    monkeypatch.setattr(
        audit, "_now_utc_iso", lambda: "2026-05-02T15:00:00+00:00"
    )

    temp_db.replace_run(_build_rag_graph())
    out1 = audit.export_audit_json("audit-rag-1")
    out2 = audit.export_audit_json("audit-rag-1")

    assert out1 == out2


def test_export_json_is_deterministic_across_db_instances(tmp_path, monkeypatch):
    """Same graph saved to two different DuckDB files → identical output.

    This is the stronger determinism claim: not just stable within one
    process, but stable across DuckDB instances regardless of internal
    row return order.
    """
    from rudriq.export import audit
    from rudriq.storage import duckdb_backend

    monkeypatch.setattr(
        audit, "_now_utc_iso", lambda: "2026-05-02T15:00:00+00:00"
    )

    db1 = duckdb_backend.DuckDBStorage(db_path=tmp_path / "db1.duckdb")
    db2 = duckdb_backend.DuckDBStorage(db_path=tmp_path / "db2.duckdb")
    try:
        graph = _build_rag_graph()
        db1.replace_run(graph)
        db2.replace_run(graph)

        monkeypatch.setattr(duckdb_backend, "_default", db1)
        out1 = audit.export_audit_json("audit-rag-1")

        monkeypatch.setattr(duckdb_backend, "_default", db2)
        out2 = audit.export_audit_json("audit-rag-1")

        assert out1 == out2, "JSON output differs across DuckDB instances"
    finally:
        db1.close()
        db2.close()


def test_export_json_lineage_chains_walk_full_upstream(temp_db):
    from rudriq.export.audit import export_audit_json

    temp_db.replace_run(_build_rag_graph())
    parsed = json.loads(export_audit_json("audit-rag-1"))

    chains = parsed["lineage_chains"]
    assert len(chains) == 2  # two LLM calls

    # The embedding call's chain: data-filter <- data-read
    embed_chain = next(c for c in chains if c["llm_node_id"] == "llm-embed")
    chain_ids = [step["node_id"] for step in embed_chain["chain"]]
    assert "data-filter" in chain_ids
    assert "data-read" in chain_ids
    assert embed_chain["chain_length"] == 2


def test_export_json_raises_on_unknown_run(temp_db):
    from rudriq.export.audit import export_audit_json

    with pytest.raises(ValueError, match="Run not found"):
        export_audit_json("does-not-exist")


def test_export_markdown_starts_with_header(temp_db):
    from rudriq.export.audit import export_audit_markdown

    temp_db.replace_run(_build_rag_graph())
    md = export_audit_markdown("audit-rag-1")

    assert md.startswith("# RudriQ Audit Report")
    assert "## Summary" in md
    assert "## LLM Lineage Chains" in md
    assert "## Full Operations Appendix" in md


def test_export_markdown_contains_each_node(temp_db):
    from rudriq.export.audit import export_audit_markdown

    temp_db.replace_run(_build_rag_graph())
    md = export_audit_markdown("audit-rag-1")

    assert "data-read" in md
    assert "data-filter" in md
    assert "llm-embed" in md
    assert "llm-chat" in md


def test_export_markdown_handles_empty_run(temp_db):
    from rudriq.export.audit import export_audit_markdown

    empty = TraceGraph(run_id="empty", created_at=datetime.now(timezone.utc))
    temp_db.replace_run(empty)
    md = export_audit_markdown("empty")

    assert "# RudriQ Audit Report" in md
    assert "_No LLM operations recorded in this run._" in md


def test_export_markdown_handles_unlinked_llm_call(temp_db):
    """An LLM call with no upstream chain should render gracefully."""
    from rudriq.export.audit import export_audit_markdown

    base = datetime(2026, 5, 2, 14, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id="orphan-llm", created_at=base)
    g.add_node(TraceNode(
        node_id="orphan",
        kind=NodeKind.LLM_CHAT, library="openai",
        operation="chat.completions.create",
        started_at=base,
    ))
    temp_db.replace_run(g)

    md = export_audit_markdown("orphan-llm")
    assert "_No upstream operations linked to this LLM call._" in md


def test_summary_counts_unlinked_llm_calls_correctly(temp_db):
    from rudriq.export.audit import export_audit_json

    base = datetime(2026, 5, 2, 14, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id="mixed", created_at=base)

    g.add_node(TraceNode(
        node_id="data-1",
        kind=NodeKind.DATA_READ, library="pandas", operation="read_csv",
        started_at=base,
    ))
    g.add_node(TraceNode(
        node_id="llm-linked",
        kind=NodeKind.LLM_CHAT, library="openai", operation="chat",
        started_at=base + timedelta(milliseconds=10),
    ))
    g.add_node(TraceNode(
        node_id="llm-orphan",
        kind=NodeKind.LLM_CHAT, library="openai", operation="chat",
        started_at=base + timedelta(milliseconds=20),
    ))
    g.add_edge(TraceEdge(
        parent_id="data-1", child_id="llm-linked",
        kind=EdgeKind.LINEAGE_LINK, confidence=1.0,
        link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    temp_db.replace_run(g)

    parsed = json.loads(export_audit_json("mixed"))
    assert parsed["summary"]["linked_llm_calls"] == 1
    assert parsed["summary"]["unlinked_llm_calls"] == 1


def test_export_via_legacy_api_with_run_id(temp_db):
    """generate_audit_report still works for v0.0.x callers."""
    from rudriq.export.audit import generate_audit_report

    temp_db.replace_run(_build_rag_graph())
    out = generate_audit_report(run_id="audit-rag-1", output_format="json")

    # Legacy returns dict for json
    assert isinstance(out, dict)
    assert "summary" in out


def test_legacy_api_pdf_raises_not_implemented(temp_db):
    from rudriq.export.audit import generate_audit_report

    temp_db.replace_run(_build_rag_graph())
    with pytest.raises(NotImplementedError, match="PDF"):
        generate_audit_report(run_id="audit-rag-1", output_format="pdf")


def test_lineage_chain_handles_external_parent_id(temp_db):
    """A LINEAGE_LINK edge whose parent_id is NOT a node in this run
    (e.g., AutoLineage lid not yet mirrored) must surface in the chain
    with an 'unknown' kind rather than disappear silently."""
    from rudriq.export.audit import export_audit_json

    base = datetime(2026, 5, 2, 14, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id="external-parent", created_at=base)
    g.add_node(TraceNode(
        node_id="llm-1",
        kind=NodeKind.LLM_EMBEDDING, library="openai",
        operation="embeddings.create",
        started_at=base,
    ))
    g.add_edge(TraceEdge(
        parent_id="autolineage-lid-XYZ",  # not present as a node
        child_id="llm-1",
        kind=EdgeKind.LINEAGE_LINK, confidence=1.0,
        link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    temp_db.replace_run(g)

    parsed = json.loads(export_audit_json("external-parent"))
    chain = parsed["lineage_chains"][0]["chain"]
    assert len(chain) == 1
    assert chain[0]["node_id"] == "autolineage-lid-XYZ"
    assert chain[0]["library"] == "external"
