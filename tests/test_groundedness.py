"""Tests for the groundedness (hallucination) evaluator."""

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
from rudriq.evaluate.groundedness import GroundednessEvaluator, _split_sentences


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


def _graph_with_one_llm(llm_md: dict, doc_md: dict) -> TraceGraph:
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


def test_split_sentences_basic():
    s = _split_sentences("First sentence here. Second one follows! Third one is here?")
    assert len(s) == 3


def test_split_sentences_drops_short_fragments():
    s = _split_sentences("Hi. This is a longer real sentence here.")
    # "Hi." is too short to score; only the longer fragment is kept.
    assert len(s) == 1
    assert s[0].startswith("This is")


def test_skips_when_no_response_text():
    graph = _graph_with_one_llm({}, {"output_preview": "some context here"})
    results = GroundednessEvaluator().evaluate(graph)
    assert any(r.status == EvalStatus.SKIPPED for r in results)


def test_degrades_without_fastembed(monkeypatch):
    monkeypatch.setattr(
        "rudriq.evaluate.groundedness.embed_texts", lambda texts: None
    )
    graph = _graph_with_one_llm(
        {"gen_ai.completion": "Paris is the capital of France. It is a large city."},
        {"output_preview": "France's capital is Paris, a major European city."},
    )
    results = GroundednessEvaluator().evaluate(graph)
    assert any(r.status == EvalStatus.DEGRADED for r in results)


def test_high_groundedness_when_response_matches_context(monkeypatch):
    """All vectors identical → every sentence grounded → score 1.0."""
    monkeypatch.setattr(
        "rudriq.evaluate.groundedness.embed_texts",
        lambda texts: [[1.0, 0.0] for _ in texts],
    )
    graph = _graph_with_one_llm(
        {"gen_ai.completion": "Paris is the capital here. It is a large city now."},
        {"output_preview": "Paris is the capital of France."},
    )
    results = GroundednessEvaluator().evaluate(graph)
    ok = [r for r in results if r.status == EvalStatus.OK]
    assert len(ok) == 1
    assert ok[0].score == pytest.approx(1.0)
    assert ok[0].details["grounded_sentences"] == ok[0].details["total_sentences"]


def test_embedding_call_skipped_as_not_applicable():
    """Embedding spans have no response by spec — groundedness flags
    not_applicable so the audit report can suppress the noise.

    The SKIP is correct (no response to score). What was wrong before
    Day 17 Thread A was that this SKIP appeared in the audit report's
    Notable findings as if it were a gap. The ``not_applicable`` flag
    in details lets the renderer distinguish "doesn't apply here" from
    "missing data we should have."
    """
    emb = _node("emb1", NodeKind.LLM_EMBEDDING, {"rudriq.prompt_preview": "x"})
    doc = _node("doc1", NodeKind.DATA_READ, {"rudriq.content_preview": "context"})
    edge = TraceEdge(
        parent_id="doc1", child_id="emb1", kind=EdgeKind.LINEAGE_LINK,
        confidence=0.7, link_method=LinkMethod.SUBSTRING, metadata={},
    )
    graph = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc), metadata={},
        nodes=[emb, doc], edges=[edge],
    )

    results = GroundednessEvaluator().evaluate(graph)
    emb_results = [r for r in results if r.node_id == "emb1"]
    assert len(emb_results) == 1
    assert emb_results[0].status == EvalStatus.SKIPPED
    assert emb_results[0].details.get("not_applicable") is True
    assert "Not applicable" in emb_results[0].explanation


def test_low_groundedness_flags_ungrounded(monkeypatch):
    """Response sentences orthogonal to context → score 0."""

    def fake_embed(texts):
        # Evaluator orders response sentences first, then docs.
        # The completion below yields 2 sentences; 1 doc → 3 vectors total.
        # First 2 (sentences) = [1,0]; last (doc) = [0,1] → cosine 0.
        return [[1.0, 0.0]] * (len(texts) - 1) + [[0.0, 1.0]]

    monkeypatch.setattr("rudriq.evaluate.groundedness.embed_texts", fake_embed)
    graph = _graph_with_one_llm(
        {"gen_ai.completion": "Completely unrelated claim one here. Another wild claim now."},
        {"output_preview": "The actual context is about something else entirely."},
    )
    results = GroundednessEvaluator().evaluate(graph)
    ok = [r for r in results if r.status == EvalStatus.OK]
    assert len(ok) == 1
    assert ok[0].score == pytest.approx(0.0)
    assert "Ungrounded:" in ok[0].explanation
    assert "semantic-similarity heuristic" in ok[0].explanation
