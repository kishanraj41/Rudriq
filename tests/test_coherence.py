"""Tests for the coherence evaluator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rudriq.core.schema import (
    EdgeKind,
    LinkMethod,
    NodeKind,
    TraceEdge,
    TraceGraph,
    TraceNode,
)
from rudriq.evaluate.base import EvalStatus
from rudriq.evaluate.coherence import CoherenceEvaluator
from rudriq.evaluate.embeddings import reset_embedder_for_tests


@pytest.fixture(autouse=True)
def reset_embedder():
    reset_embedder_for_tests()
    yield
    reset_embedder_for_tests()


def _node(nid: str, kind: NodeKind, md: dict) -> TraceNode:
    return TraceNode(
        node_id=nid, kind=kind, library="t", operation="o",
        started_at=datetime.now(timezone.utc), ended_at=None, metadata=md,
    )


def _graph(llm_md: dict, doc_md: dict) -> TraceGraph:
    llm = _node("llm1", NodeKind.LLM_CHAT, llm_md)
    doc = _node("doc1", NodeKind.DATA_READ, doc_md)
    edge = TraceEdge(
        parent_id="doc1", child_id="llm1", kind=EdgeKind.LINEAGE_LINK,
        confidence=0.7, link_method=LinkMethod.SUBSTRING, metadata={},
    )
    return TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc), metadata={},
        nodes=[llm, doc], edges=[edge],
    )


def test_skips_without_linked_context():
    g = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc), metadata={},
        nodes=[_node("llm1", NodeKind.LLM_CHAT, {})], edges=[],
    )
    results = CoherenceEvaluator().evaluate(g)
    assert len(results) == 1
    assert results[0].status == EvalStatus.SKIPPED


def test_degrades_without_fastembed(monkeypatch):
    monkeypatch.setattr(
        "rudriq.evaluate.coherence.embed_texts", lambda t: None
    )
    g = _graph(
        {"rudriq.completion_preview": "response text here"},
        {"rudriq.content_preview": "context text here"},
    )
    results = CoherenceEvaluator().evaluate(g)
    assert any(r.status == EvalStatus.DEGRADED for r in results)


def test_high_coherence_when_aligned(monkeypatch):
    """Identical vectors → coherence 1.0."""
    monkeypatch.setattr(
        "rudriq.evaluate.coherence.embed_texts",
        lambda t: [[1.0, 0.0], [1.0, 0.0]],
    )
    g = _graph(
        {"rudriq.completion_preview": "Paris is the capital."},
        {"rudriq.content_preview": "France capital Paris."},
    )
    ok = [r for r in CoherenceEvaluator().evaluate(g) if r.status == EvalStatus.OK]
    assert len(ok) == 1
    assert ok[0].score == pytest.approx(1.0)
    assert "Response-to-context coherence" in ok[0].explanation


def test_low_coherence_flags_decorative_retrieval(monkeypatch):
    """Orthogonal vectors → low coherence → decorative-retrieval warning."""
    monkeypatch.setattr(
        "rudriq.evaluate.coherence.embed_texts",
        lambda t: [[1.0, 0.0], [0.0, 1.0]],
    )
    g = _graph(
        {"rudriq.completion_preview": "Unrelated answer."},
        {"rudriq.content_preview": "Context about something else."},
    )
    ok = [r for r in CoherenceEvaluator().evaluate(g) if r.status == EvalStatus.OK]
    assert len(ok) == 1
    assert "ignoring retrieval" in ok[0].explanation
    assert ok[0].details["coherence_similarity"] == pytest.approx(0.0)
