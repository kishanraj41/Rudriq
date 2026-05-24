"""Tests for the Day 1/9 ``--include-rca`` audit-report embedding.

The schema-1.2 contract: a new top-level ``root_cause_analysis`` key,
``null`` when ``--include-rca`` is not passed, otherwise an object with
``target_node_id`` / ``candidates`` / ``note``. The list of candidates
must be deterministic (same input → byte-identical output) for the
audit-grade hashing claim to keep holding.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
from rudriq.evaluate.embeddings import reset_embedder_for_tests
from rudriq.export.audit import (
    _auto_select_failure_target,
    _run_audit_rca,
    export_audit_json,
    export_audit_markdown,
)
from rudriq.storage.duckdb_backend import DuckDBStorage


@pytest.fixture(autouse=True)
def _reset_embedder():
    reset_embedder_for_tests()
    yield
    reset_embedder_for_tests()


@pytest.fixture
def storage_with_chain(tmp_path: Path, monkeypatch):
    """A small read→filter→embed→chat graph persisted in a temp DuckDB."""
    from rudriq.storage import duckdb_backend

    db = DuckDBStorage(db_path=tmp_path / "rca_audit.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)

    base = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id="rca-audit-run", created_at=base)
    g.add_node(TraceNode(
        node_id="read", kind=NodeKind.DATA_READ,
        library="pandas", operation="read_csv",
        started_at=base, ended_at=base + timedelta(milliseconds=5),
        metadata={"autolineage.shape": [1000, 5]},
    ))
    g.add_node(TraceNode(
        node_id="filter", kind=NodeKind.DATA_TRANSFORM,
        library="pandas", operation="filter",
        started_at=base + timedelta(milliseconds=6),
        ended_at=base + timedelta(milliseconds=10),
        metadata={"autolineage.shape": [250, 5]},
    ))
    g.add_node(TraceNode(
        node_id="embed", kind=NodeKind.LLM_EMBEDDING,
        library="openai", operation="embeddings",
        started_at=base + timedelta(milliseconds=12), ended_at=None,
        metadata={"rudriq.prompt_preview": "embed these"},
    ))
    g.add_node(TraceNode(
        node_id="chat", kind=NodeKind.LLM_CHAT,
        library="openai", operation="chat",
        started_at=base + timedelta(milliseconds=20), ended_at=None,
        metadata={
            "rudriq.prompt_preview": "What is the capital of France?",
            "rudriq.completion_preview": "Paris is the capital of France.",
        },
    ))
    g.add_edge(TraceEdge(
        parent_id="read", child_id="filter", kind=EdgeKind.DIRECT,
        confidence=1.0, link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    g.add_edge(TraceEdge(
        parent_id="filter", child_id="embed", kind=EdgeKind.LINEAGE_LINK,
        confidence=1.0, link_method=LinkMethod.OBJECT_IDENTITY,
    ))
    g.add_edge(TraceEdge(
        parent_id="embed", child_id="chat", kind=EdgeKind.LINEAGE_LINK,
        confidence=0.7, link_method=LinkMethod.SUBSTRING,
    ))
    db.replace_run(g)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Null-shape contract when --include-rca is OFF
# ---------------------------------------------------------------------------


def test_json_without_rca_has_null_root_cause_analysis(storage_with_chain):
    out = export_audit_json("rca-audit-run", include_rca=False)
    report = json.loads(out)
    assert report["schema_version"] == "rudriq.audit/1.2"
    assert report["root_cause_analysis"] is None


def test_markdown_without_rca_omits_section(storage_with_chain):
    out = export_audit_markdown("rca-audit-run", include_rca=False)
    assert "## Root Cause Analysis" not in out


# ---------------------------------------------------------------------------
# Embed shape: structure + ordering when --include-rca is ON
# ---------------------------------------------------------------------------


def test_json_with_rca_embeds_ranked_candidates(storage_with_chain):
    """RCA payload populates: target picked, candidates ranked
    descending, honesty note present. Doesn't pin the *specific*
    auto-selected target — the picker has its own dedicated unit
    test and this fixture's metadata is intentionally thin enough
    that the picker falls back to lex-last (covers a real path)."""
    out = export_audit_json("rca-audit-run", include_rca=True)
    report = json.loads(out)
    rca = report["root_cause_analysis"]
    assert rca is not None
    assert rca["target_node_id"] in {"chat", "embed"}
    # Candidates must be ranked descending by score.
    scores = [c["score"] for c in rca["candidates"]]
    assert scores == sorted(scores, reverse=True)
    # At least one candidate from the upstream chain.
    assert len(rca["candidates"]) >= 1
    # Honest framing is in the note.
    assert "NOT a causal proof" in rca["note"]


def test_markdown_with_rca_has_section_and_table(storage_with_chain):
    out = export_audit_markdown("rca-audit-run", include_rca=True)
    assert "## Root Cause Analysis" in out
    # Diagnoses *some* LLM node; we don't pin which one (see above).
    assert "**Diagnosing:** node `" in out
    assert "| Rank | Operation | Score | Hops upstream | Why flagged |" in out
    assert "NOT a causal proof" in out


def test_explicit_rca_target_overrides_auto_select(storage_with_chain):
    """Passing --rca-target uses that node_id, ignoring auto-select."""
    out = export_audit_json(
        "rca-audit-run", include_rca=True, rca_target="embed",
    )
    rca = json.loads(out)["root_cause_analysis"]
    assert rca["target_node_id"] == "embed"
    # embed has only one upstream (filter via lineage edge in this fixture)
    # — and filter has read as DIRECT parent → 2 ancestors total
    ids = [c["node_id"] for c in rca["candidates"]]
    assert set(ids) == {"filter", "read"}


# ---------------------------------------------------------------------------
# Determinism — the audit-grade load-bearing claim, extended to 1.2
# ---------------------------------------------------------------------------


def test_json_with_rca_is_deterministic(storage_with_chain):
    """Two RCA-augmented exports of the same run must be byte-identical."""
    out1 = export_audit_json("rca-audit-run", include_rca=True)
    out2 = export_audit_json("rca-audit-run", include_rca=True)
    assert out1 == out2, "RCA-augmented JSON is not byte-deterministic"
    report = json.loads(out1)
    assert "generated_at" not in report


def test_markdown_with_rca_is_deterministic(storage_with_chain):
    out1 = export_audit_markdown("rca-audit-run", include_rca=True)
    out2 = export_audit_markdown("rca-audit-run", include_rca=True)
    assert out1 == out2, "RCA-augmented Markdown is not byte-deterministic"


# ---------------------------------------------------------------------------
# Auto-select behavior — direct unit test on the picker
# ---------------------------------------------------------------------------


def test_auto_select_picks_lowest_groundedness_when_available(monkeypatch):
    """With multiple LLM nodes, the auto-selector should prefer the
    lowest-scoring one (the most plausibly failing call), not a
    deterministic fallback."""
    base = datetime(2026, 5, 24, tzinfo=timezone.utc)
    g = TraceGraph(run_id="picker", created_at=base)
    # Two chat nodes, one with retrieval the response is grounded in,
    # one with retrieval that's orthogonal.
    g.add_node(TraceNode(
        node_id="doc_good", kind=NodeKind.DATA_READ,
        library="pandas", operation="read",
        started_at=base, ended_at=base,
        metadata={"rudriq.content_preview": "Paris is the capital of France."},
    ))
    g.add_node(TraceNode(
        node_id="doc_bad", kind=NodeKind.DATA_READ,
        library="pandas", operation="read",
        started_at=base, ended_at=base,
        metadata={"rudriq.content_preview": "totally unrelated context about cuisine"},
    ))
    g.add_node(TraceNode(
        node_id="chat_grounded", kind=NodeKind.LLM_CHAT,
        library="openai", operation="chat",
        started_at=base, ended_at=base,
        metadata={
            "rudriq.prompt_preview": "France capital",
            "rudriq.completion_preview": "Paris is the capital of France.",
        },
    ))
    g.add_node(TraceNode(
        node_id="chat_hallucinated", kind=NodeKind.LLM_CHAT,
        library="openai", operation="chat",
        started_at=base, ended_at=base,
        metadata={
            "rudriq.prompt_preview": "France capital",
            "rudriq.completion_preview": "The capital is purple and weighs 12 kilograms.",
        },
    ))
    g.add_edge(TraceEdge(
        parent_id="doc_good", child_id="chat_grounded",
        kind=EdgeKind.LINEAGE_LINK, confidence=0.7,
        link_method=LinkMethod.SUBSTRING,
    ))
    g.add_edge(TraceEdge(
        parent_id="doc_bad", child_id="chat_hallucinated",
        kind=EdgeKind.LINEAGE_LINK, confidence=0.7,
        link_method=LinkMethod.SUBSTRING,
    ))

    # Stub embed_texts so the hallucinated chat scores low and the
    # grounded chat scores high. We route by content: if the text
    # contains "purple" it's the hallucinated response, embed as a
    # vector orthogonal to its docs.
    def fake_embed(texts):
        out = []
        for t in texts:
            if "purple" in t:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return out
    monkeypatch.setattr(
        "rudriq.evaluate.groundedness.embed_texts", fake_embed
    )

    target = _auto_select_failure_target(g)
    assert target == "chat_hallucinated"


def test_auto_select_falls_back_to_last_llm_node_id_when_no_scores():
    """When groundedness can't produce a scored result (no eval),
    the picker falls back to the lexicographically last LLM node_id —
    deterministic and never raises."""
    base = datetime(2026, 5, 24, tzinfo=timezone.utc)
    # LLM nodes with no linked upstream context → groundedness SKIPs.
    g = TraceGraph(run_id="fallback", created_at=base, nodes=[
        TraceNode(
            node_id="llm_a", kind=NodeKind.LLM_CHAT, library="x", operation="y",
            started_at=base, ended_at=base, metadata={},
        ),
        TraceNode(
            node_id="llm_z", kind=NodeKind.LLM_CHAT, library="x", operation="y",
            started_at=base, ended_at=base, metadata={},
        ),
    ])
    target = _auto_select_failure_target(g)
    # Lexicographic last → llm_z
    assert target == "llm_z"


def test_auto_select_returns_none_for_graph_without_llm_nodes():
    """A pure data-side graph has nothing to diagnose; picker → None,
    and the RCA helper emits a clean 'nothing to diagnose' shape."""
    base = datetime(2026, 5, 24, tzinfo=timezone.utc)
    g = TraceGraph(run_id="empty-llm", created_at=base, nodes=[
        TraceNode(
            node_id="data_only", kind=NodeKind.DATA_READ,
            library="pandas", operation="read",
            started_at=base, ended_at=base, metadata={},
        ),
    ])
    assert _auto_select_failure_target(g) is None

    rca = _run_audit_rca(g)
    assert rca["target_node_id"] is None
    assert rca["candidates"] == []
