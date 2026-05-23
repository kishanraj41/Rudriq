"""Tests for the within-run consistency evaluator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rudriq.core.schema import NodeKind, TraceGraph, TraceNode
from rudriq.evaluate.base import EvalStatus
from rudriq.evaluate.consistency import ConsistencyEvaluator
from rudriq.evaluate.embeddings import reset_embedder_for_tests


@pytest.fixture(autouse=True)
def reset_embedder():
    reset_embedder_for_tests()
    yield
    reset_embedder_for_tests()


def _llm(nid: str, prompt: str, response: str) -> TraceNode:
    return TraceNode(
        node_id=nid, kind=NodeKind.LLM_CHAT, library="openai", operation="chat",
        started_at=datetime.now(timezone.utc), ended_at=None,
        metadata={
            "rudriq.prompt_preview": prompt,
            "rudriq.completion_preview": response,
        },
    )


def _graph(nodes: list[TraceNode]) -> TraceGraph:
    return TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc), metadata={},
        nodes=nodes, edges=[],
    )


def test_skips_with_fewer_than_two_calls():
    g = _graph([_llm("1", "q", "a")])
    results = ConsistencyEvaluator().evaluate(g)
    assert len(results) == 1
    assert results[0].status == EvalStatus.SKIPPED


def test_degrades_without_fastembed(monkeypatch):
    monkeypatch.setattr(
        "rudriq.evaluate.consistency.embed_texts", lambda t: None
    )
    g = _graph([_llm("1", "q", "a"), _llm("2", "q", "a")])
    results = ConsistencyEvaluator().evaluate(g)
    assert results[0].status == EvalStatus.DEGRADED


def test_skips_when_all_prompts_distinct(monkeypatch):
    """Distinct prompt embeddings → all singleton groups → SKIPPED."""

    def fake_embed(texts):
        # Each text gets a distinct one-hot vector — orthogonal pairwise.
        return [
            [1.0 if i == j else 0.0 for j in range(len(texts))]
            for i in range(len(texts))
        ]

    monkeypatch.setattr(
        "rudriq.evaluate.consistency.embed_texts", fake_embed
    )
    g = _graph([
        _llm("1", "topic A", "ans A"),
        _llm("2", "topic B", "ans B"),
    ])
    results = ConsistencyEvaluator().evaluate(g)
    assert len(results) == 1
    assert results[0].status == EvalStatus.SKIPPED
    assert "distinct" in results[0].explanation


def test_high_consistency_for_repeated_prompt(monkeypatch):
    """Same prompt, same response → grouped, score 1.0."""

    monkeypatch.setattr(
        "rudriq.evaluate.consistency.embed_texts",
        lambda t: [[1.0, 0.0] for _ in t],
    )
    g = _graph([
        _llm("1", "same query", "same answer"),
        _llm("2", "same query", "same answer"),
    ])
    ok = [
        r for r in ConsistencyEvaluator().evaluate(g)
        if r.status == EvalStatus.OK
    ]
    assert len(ok) == 1
    assert ok[0].score == pytest.approx(1.0)
    assert ok[0].details["group_size"] == 2


def test_groups_by_user_message_not_full_prompt(monkeypatch):
    """The Day 17 Thread B fix: ConsistencyEvaluator must group on the
    user message, not on the assembled prompt. Two RAG calls with
    DIFFERENT user questions but SIMILAR retrieval context must NOT
    be grouped together (or the metric reports a false 1.0).

    Stub strategy: distinct user messages → orthogonal vectors; the
    full prompts (sharing retrieval context) would be near-identical
    but the evaluator should never look at them when user_message is
    present.
    """

    def fake_embed(texts):
        # Each text gets a one-hot vector — orthogonal pairwise, so
        # nothing groups regardless of shared content.
        return [
            [1.0 if i == j else 0.0 for j in range(len(texts))]
            for i in range(len(texts))
        ]

    monkeypatch.setattr(
        "rudriq.evaluate.consistency.embed_texts", fake_embed
    )

    shared_context = "Context: " + ("doc text " * 50)

    def _llm_with_user_msg(nid, user_msg, full_prompt, response):
        return TraceNode(
            node_id=nid, kind=NodeKind.LLM_CHAT,
            library="openai", operation="chat",
            started_at=datetime.now(timezone.utc), ended_at=None,
            metadata={
                "rudriq.user_message_preview": user_msg,
                "rudriq.prompt_preview": full_prompt,
                "rudriq.completion_preview": response,
            },
        )

    g = _graph([
        _llm_with_user_msg(
            "1", "What is topic five?", shared_context + " topic five", "answer A",
        ),
        _llm_with_user_msg(
            "2", "Explain topic twelve",
            shared_context + " topic twelve", "answer B",
        ),
    ])
    results = ConsistencyEvaluator().evaluate(g)
    # Distinct user messages → no grouping → SKIPPED (the correct
    # answer for distinct queries), NOT a false 1.0 OK.
    assert len(results) == 1
    assert results[0].status == EvalStatus.SKIPPED
    assert "distinct" in results[0].explanation


def test_falls_back_to_full_prompt_for_legacy_traces(monkeypatch):
    """For legacy traces without rudriq.user_message_preview, the
    evaluator still works against the full prompt — same as before
    Day 17 Thread B. The fix is additive, not breaking."""

    monkeypatch.setattr(
        "rudriq.evaluate.consistency.embed_texts",
        lambda t: [[1.0, 0.0] for _ in t],
    )

    # No user_message_preview — only the legacy prompt_preview
    def _legacy_llm(nid, prompt, response):
        return TraceNode(
            node_id=nid, kind=NodeKind.LLM_CHAT,
            library="openai", operation="chat",
            started_at=datetime.now(timezone.utc), ended_at=None,
            metadata={
                "rudriq.prompt_preview": prompt,
                "rudriq.completion_preview": response,
            },
        )

    g = _graph([
        _legacy_llm("1", "shared prompt text", "answer A"),
        _legacy_llm("2", "shared prompt text", "answer B"),
    ])
    ok = [
        r for r in ConsistencyEvaluator().evaluate(g)
        if r.status == EvalStatus.OK
    ]
    assert len(ok) == 1
    assert ok[0].details["group_size"] == 2


def test_low_consistency_flags_divergent_responses(monkeypatch):
    """Prompts grouped, responses divergent → low-consistency warning."""

    def fake_embed(texts):
        # Heuristic: prompts contain "query"; responses don't. The evaluator
        # calls embed_texts(prompts) first, then embed_texts(responses) per
        # group, so we can route by content.
        if all("query" in t for t in texts):
            return [[1.0, 0.0] for _ in texts]  # identical prompts
        # Divergent responses (must match the actual call shape: 2 responses)
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr(
        "rudriq.evaluate.consistency.embed_texts", fake_embed
    )
    g = _graph([
        _llm("1", "same query", "answer one"),
        _llm("2", "same query", "totally different"),
    ])
    ok = [
        r for r in ConsistencyEvaluator().evaluate(g)
        if r.status == EvalStatus.OK
    ]
    assert len(ok) == 1
    assert "Low consistency" in ok[0].explanation
    assert ok[0].score < 0.7
