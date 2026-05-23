"""Tests for eval-augmented audit reports (Day 16 ``--include-evals``)."""

from __future__ import annotations

import json
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
from rudriq.evaluate.embeddings import reset_embedder_for_tests
from rudriq.export.audit import (
    _run_audit_evaluations,
    _summarize_evaluations,
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
def storage_with_run(tmp_path: Path, monkeypatch):
    """A DuckDB with a single small RAG run containing previews."""
    from rudriq.storage import duckdb_backend

    db = DuckDBStorage(db_path=tmp_path / "audit_eval.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)

    base = datetime(2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id="audit-eval-run", created_at=base)
    g.add_node(TraceNode(
        node_id="doc1", kind=NodeKind.DATA_READ,
        library="pandas", operation="read_csv",
        started_at=base, ended_at=base,
        metadata={"rudriq.content_preview": "France's capital city is Paris."},
    ))
    g.add_node(TraceNode(
        node_id="llm1", kind=NodeKind.LLM_CHAT,
        library="openai", operation="chat",
        started_at=base, ended_at=base,
        metadata={
            "rudriq.prompt_preview": "What is the capital of France?",
            "rudriq.completion_preview": "Paris is the capital of France.",
        },
    ))
    g.add_edge(TraceEdge(
        parent_id="doc1", child_id="llm1", kind=EdgeKind.LINEAGE_LINK,
        confidence=0.7, link_method=LinkMethod.SUBSTRING,
    ))
    db.replace_run(g)
    yield db
    db.close()


def _stub_embeddings_for_all(monkeypatch):
    """Make every evaluator's embed_texts return identical [1, 0] vectors.

    Stubbing per-module (rather than mocking ``fastembed``) keeps the
    test independent of whether fastembed is installed locally.
    """
    for mod in (
        "retrieval_relevance", "groundedness", "coherence", "consistency",
    ):
        monkeypatch.setattr(
            f"rudriq.evaluate.{mod}.embed_texts",
            lambda t: [[1.0, 0.0] for _ in t],
        )


# ---------------------------------------------------------------------------
# Schema-version + null shape when --include-evals is OFF
# ---------------------------------------------------------------------------


def test_json_without_evals_has_null_evaluations(storage_with_run):
    out = export_audit_json("audit-eval-run", include_evals=False)
    report = json.loads(out)
    assert report["schema_version"] == "rudriq.audit/1.1"
    assert report["evaluations"] is None
    assert report["evaluation_summary"] is None


def test_markdown_without_evals_omits_quality_section(storage_with_run):
    out = export_audit_markdown("audit-eval-run", include_evals=False)
    assert "## Quality Evaluation" not in out


# ---------------------------------------------------------------------------
# Determinism (load-bearing): byte-identical output across two exports.
# Freezes ``_now_utc_iso`` because the top-level ``generated_at`` is
# documented as the one non-deterministic field.
# ---------------------------------------------------------------------------


def test_json_with_evals_is_deterministic(storage_with_run, monkeypatch):
    _stub_embeddings_for_all(monkeypatch)
    monkeypatch.setattr(
        "rudriq.export.audit._now_utc_iso",
        lambda: "2026-05-23T12:00:00+00:00",
    )

    out1 = export_audit_json("audit-eval-run", include_evals=True)
    out2 = export_audit_json("audit-eval-run", include_evals=True)
    assert out1 == out2


# ---------------------------------------------------------------------------
# Embed shape: keys present, summary populated
# ---------------------------------------------------------------------------


def test_json_with_evals_embeds_results(storage_with_run, monkeypatch):
    _stub_embeddings_for_all(monkeypatch)
    out = export_audit_json("audit-eval-run", include_evals=True)
    report = json.loads(out)

    assert report["evaluations"] is not None
    assert report["evaluation_summary"] is not None
    # The four default content-aware evaluators should all have produced
    # *some* result against our linked-LLM fixture.
    summary = report["evaluation_summary"]
    assert "retrieval_relevance" in summary
    # Identical-vec stub → high mean → green for the content-aware metrics.
    assert summary["retrieval_relevance"]["traffic_light"] == "green"


def test_markdown_with_evals_has_quality_section(storage_with_run, monkeypatch):
    _stub_embeddings_for_all(monkeypatch)
    out = export_audit_markdown("audit-eval-run", include_evals=True)
    assert "## Quality Evaluation" in out
    assert "Mean Score" in out
    # The summary table heading should be present.
    assert "| Metric | Status | Mean Score | Evaluated |" in out


# ---------------------------------------------------------------------------
# Traffic-light bands — pure-data unit test on the summarizer
# ---------------------------------------------------------------------------


def test_summarize_traffic_lights():
    eval_dicts = [
        {"metric": "groundedness", "status": "ok", "score": 0.9, "node_id": "n1"},
        {"metric": "groundedness", "status": "ok", "score": 0.8, "node_id": "n2"},
        {"metric": "coherence", "status": "ok", "score": 0.5, "node_id": "n1"},
        {"metric": "retrieval_relevance", "status": "ok", "score": 0.2, "node_id": "n1"},
        {"metric": "consistency", "status": "skipped", "score": None, "node_id": None},
    ]
    summary = _summarize_evaluations(eval_dicts)
    assert summary["groundedness"]["traffic_light"] == "green"  # mean 0.85
    assert summary["groundedness"]["evaluated"] == 2
    assert summary["coherence"]["traffic_light"] == "yellow"  # 0.5
    assert summary["retrieval_relevance"]["traffic_light"] == "red"  # 0.2
    assert summary["consistency"]["traffic_light"] == "gray"  # no scored ok
    assert summary["consistency"]["evaluated"] == 0
    assert summary["consistency"]["status_counts"] == {"skipped": 1}


# ---------------------------------------------------------------------------
# Eval-result ordering inside the JSON list is deterministic
# (the JSON's outer ``sort_keys`` doesn't sort lists; this is the
# guarantee that audit diffs across runs stay clean.)
# ---------------------------------------------------------------------------


def test_run_audit_evaluations_ordering_is_stable(storage_with_run, monkeypatch):
    _stub_embeddings_for_all(monkeypatch)
    storage = storage_with_run
    graph = storage.load_run("audit-eval-run")

    out1 = _run_audit_evaluations(graph)
    out2 = _run_audit_evaluations(graph)
    assert out1 == out2

    keys = [(d["metric"], d.get("node_id") or "") for d in out1]
    assert keys == sorted(keys), (
        "audit eval list must be sorted by (metric, node_id) for "
        "deterministic byte-diffs across runs"
    )


# ---------------------------------------------------------------------------
# Notable-findings list surfaces non-OK and sub-green results in Markdown
# ---------------------------------------------------------------------------


def test_markdown_lists_notable_findings(storage_with_run, monkeypatch):
    """A red metric should appear under '### Notable findings'."""

    def stub_low(texts):
        # Orthogonal vecs → low similarity → sub-green retrieval_relevance.
        # Returns alternating [1,0] / [0,1] for any input length.
        return [
            [1.0, 0.0] if i % 2 == 0 else [0.0, 1.0]
            for i in range(len(texts))
        ]

    for mod in ("retrieval_relevance", "groundedness", "coherence", "consistency"):
        monkeypatch.setattr(f"rudriq.evaluate.{mod}.embed_texts", stub_low)

    out = export_audit_markdown("audit-eval-run", include_evals=True)
    assert "### Notable findings" in out
