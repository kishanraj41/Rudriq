"""Tests for the fpdf2-based PDF audit exporter."""

from __future__ import annotations

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
from rudriq.storage.duckdb_backend import DuckDBStorage


@pytest.fixture
def storage_with_run(tmp_path: Path, monkeypatch):
    """A modest graph (data → LLM) so the PDF has real sections to render."""
    from rudriq.storage import duckdb_backend

    db = DuckDBStorage(db_path=tmp_path / "pdf.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)

    base = datetime(2026, 5, 25, 12, 0, 0, tzinfo=timezone.utc)
    g = TraceGraph(run_id="pdf-run", created_at=base)
    g.add_node(TraceNode(
        node_id="doc1", kind=NodeKind.DATA_READ,
        library="pandas", operation="read_csv",
        started_at=base, ended_at=base + timedelta(milliseconds=5),
        metadata={"rudriq.content_preview": "France's capital is Paris."},
    ))
    g.add_node(TraceNode(
        node_id="llm1", kind=NodeKind.LLM_CHAT,
        library="openai", operation="chat",
        started_at=base + timedelta(milliseconds=10),
        ended_at=base + timedelta(milliseconds=20),
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


# ---------------------------------------------------------------------------
# Output shape — file written, valid PDF magic bytes
# ---------------------------------------------------------------------------


def test_pdf_export_writes_valid_pdf(storage_with_run, tmp_path: Path):
    """End-to-end smoke: the file exists and is recognizable as PDF."""
    pytest.importorskip("fpdf")
    from rudriq.export.pdf import export_audit_pdf

    out = tmp_path / "report.pdf"
    export_audit_pdf("pdf-run", str(out))
    assert out.exists()
    # All PDF files start with the four-byte ``%PDF`` magic.
    assert out.read_bytes()[:4] == b"%PDF"
    # And carry a non-trivial body — a single-byte "PDF" header would
    # be a sign rendering bailed out.
    assert out.stat().st_size > 1000


def test_pdf_export_with_evals_and_rca(storage_with_run, tmp_path: Path, monkeypatch):
    """All three optional sections render together without raising."""
    pytest.importorskip("fpdf")
    # Stub embed_texts so we don't depend on fastembed in CI.
    for mod in (
        "retrieval_relevance", "groundedness", "coherence", "consistency",
    ):
        monkeypatch.setattr(
            f"rudriq.evaluate.{mod}.embed_texts",
            lambda t: [[1.0, 0.0] for _ in t],
        )
    from rudriq.export.pdf import export_audit_pdf

    out = tmp_path / "full.pdf"
    export_audit_pdf(
        "pdf-run", str(out),
        include_evals=True, include_rca=True,
    )
    assert out.read_bytes()[:4] == b"%PDF"
    assert out.stat().st_size > 2000  # carries the extra sections


# ---------------------------------------------------------------------------
# Determinism — the audit-grade load-bearing claim, extended to PDF.
# ---------------------------------------------------------------------------


def test_pdf_export_is_deterministic(storage_with_run, tmp_path: Path):
    """Two consecutive PDF exports of the same run must be byte-identical.

    fpdf2 embeds a creation-date timestamp in the PDF metadata by
    default; the exporter pins it to the run's ``created_at`` so the
    output is reproducible — same audit-grade hashability the JSON
    and Markdown forms already have.
    """
    pytest.importorskip("fpdf")
    from rudriq.export.pdf import export_audit_pdf

    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    export_audit_pdf("pdf-run", str(a))
    export_audit_pdf("pdf-run", str(b))
    assert a.read_bytes() == b.read_bytes(), (
        "PDF output must be byte-deterministic for hash-as-evidence to work"
    )


# ---------------------------------------------------------------------------
# Graceful degradation when the [pdf] extra is missing.
# ---------------------------------------------------------------------------


def test_pdf_export_raises_runtime_error_without_fpdf(
    storage_with_run, tmp_path: Path, monkeypatch,
):
    """When fpdf2 is unavailable, raise a clean RuntimeError with the
    install hint — never a bare ImportError up to the user."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fpdf" or name.startswith("fpdf."):
            raise ImportError("simulated: fpdf2 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from rudriq.export.pdf import export_audit_pdf

    with pytest.raises(RuntimeError) as exc:
        export_audit_pdf("pdf-run", str(tmp_path / "x.pdf"))
    msg = str(exc.value).lower()
    assert "pdf" in msg
    assert "install" in msg or "rudriq[pdf]" in msg


# ---------------------------------------------------------------------------
# Missing run — clean ValueError, not a 500-line traceback.
# ---------------------------------------------------------------------------


def test_pdf_missing_run_raises_value_error(storage_with_run, tmp_path: Path):
    pytest.importorskip("fpdf")
    from rudriq.export.pdf import export_audit_pdf

    with pytest.raises(ValueError):
        export_audit_pdf("nonexistent-run", str(tmp_path / "x.pdf"))


# ---------------------------------------------------------------------------
# Shared-source-of-truth: the JSON and PDF render from identical dicts.
# ---------------------------------------------------------------------------


def test_build_audit_report_dict_matches_json_exporter(storage_with_run):
    """The JSON exporter must round-trip through the shared dict builder
    — guards against the two paths drifting if one is edited."""
    import json
    from rudriq.export.audit import build_audit_report_dict, export_audit_json

    d = build_audit_report_dict("pdf-run")
    parsed = json.loads(export_audit_json("pdf-run"))
    # Same top-level keys.
    assert set(d.keys()) == set(parsed.keys())
    # Same schema_version + run_id.
    assert d["schema_version"] == parsed["schema_version"]
    assert d["run"]["run_id"] == parsed["run"]["run_id"]
