"""Tests for the retrieval relevance evaluator."""

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
from rudriq.evaluate.embeddings import reset_embedder_for_tests
from rudriq.evaluate.retrieval_relevance import RetrievalRelevanceEvaluator


@pytest.fixture(autouse=True)
def reset_embedder():
    reset_embedder_for_tests()
    yield
    reset_embedder_for_tests()


def _node(node_id: str, kind: NodeKind, metadata: dict) -> TraceNode:
    return TraceNode(
        node_id=node_id,
        kind=kind,
        library="test",
        operation="op",
        started_at=datetime.now(timezone.utc),
        ended_at=None,
        metadata=metadata,
    )


def _linked_graph(llm_md: dict, doc_md: dict) -> TraceGraph:
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


def test_skips_when_no_linked_llm_nodes():
    graph = TraceGraph(
        run_id="r",
        created_at=datetime.now(timezone.utc),
        metadata={},
        nodes=[_node("llm1", NodeKind.LLM_CHAT, {})],
        edges=[],
    )
    results = RetrievalRelevanceEvaluator().evaluate(graph)
    assert len(results) == 1
    assert results[0].status == EvalStatus.SKIPPED


def test_degrades_gracefully_without_fastembed(monkeypatch):
    """When fastembed is unavailable, returns DEGRADED, not ERROR."""
    monkeypatch.setattr(
        "rudriq.evaluate.retrieval_relevance.embed_texts",
        lambda texts: None,
    )

    graph = _linked_graph(
        {"gen_ai.prompt": "What is the capital?"},
        {"output_preview": "Paris is the capital of France."},
    )
    results = RetrievalRelevanceEvaluator().evaluate(graph)
    assert len(results) == 1
    assert results[0].status == EvalStatus.DEGRADED
    assert results[0].score is None
    assert "fastembed" in results[0].explanation


def test_scores_relevant_docs_high(monkeypatch):
    """With a stub embedder, vectors pointing the same direction score 1.0."""

    def fake_embed(texts):
        # All texts identical → cosine 1.0 → score 1.0
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "rudriq.evaluate.retrieval_relevance.embed_texts", fake_embed
    )

    graph = _linked_graph(
        {"gen_ai.prompt": "capital of France?"},
        {"output_preview": "Paris is the capital."},
    )
    results = RetrievalRelevanceEvaluator().evaluate(graph)
    ok = [r for r in results if r.status == EvalStatus.OK]
    assert len(ok) == 1
    assert ok[0].score == pytest.approx(1.0, abs=0.01)
    assert ok[0].details["doc_count"] == 1


def test_flags_low_relevance_docs(monkeypatch):
    """Orthogonal vectors → low similarity → flagged in explanation."""

    def fake_embed(texts):
        # query = [1,0,0]; every doc = [0,1,0] → cosine 0 → below threshold
        vecs = [[1.0, 0.0, 0.0]]
        vecs += [[0.0, 1.0, 0.0] for _ in texts[1:]]
        return vecs

    monkeypatch.setattr(
        "rudriq.evaluate.retrieval_relevance.embed_texts", fake_embed
    )

    graph = _linked_graph(
        {"gen_ai.prompt": "capital of France?"},
        {"output_preview": "Unrelated content about cooking."},
    )
    results = RetrievalRelevanceEvaluator().evaluate(graph)
    ok = [r for r in results if r.status == EvalStatus.OK]
    assert len(ok) == 1
    assert "below relevance threshold" in ok[0].explanation
    assert ok[0].details["low_relevance_count"] == 1
