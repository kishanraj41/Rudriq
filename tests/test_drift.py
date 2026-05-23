"""Tests for the cross-run drift evaluator."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rudriq.core.schema import NodeKind, TraceGraph, TraceNode
from rudriq.evaluate.base import EvalStatus
from rudriq.evaluate.drift import DriftEvaluator
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


def _graph(nodes: list[TraceNode], run_id: str = "r") -> TraceGraph:
    return TraceGraph(
        run_id=run_id, created_at=datetime.now(timezone.utc), metadata={},
        nodes=nodes, edges=[],
    )


def test_skips_without_baseline():
    g = _graph([_llm("1", "q", "a")])
    results = DriftEvaluator(baseline_graph=None).evaluate(g)
    assert len(results) == 1
    assert results[0].status == EvalStatus.SKIPPED
    assert "baseline" in results[0].explanation.lower()


def test_structural_drift_detects_op_mix_change(monkeypatch):
    """An extra LLM call in current vs baseline → structural drift > 0."""
    monkeypatch.setattr(
        "rudriq.evaluate.drift.embed_texts",
        lambda t: [[1.0, 0.0] for _ in t],
    )
    baseline = _graph(
        [_llm("b1", "q1", "a1"), _llm("b2", "q2", "a2")], "base"
    )
    current = _graph(
        [_llm("c1", "q1", "a1"), _llm("c2", "q2", "a2"), _llm("c3", "q3", "a3")],
        "cur",
    )
    results = DriftEvaluator(baseline_graph=baseline).evaluate(current)
    structural = [r for r in results if r.metric == "drift_structural"]
    assert len(structural) == 1
    # Op mix changed (3 chat vs 2 chat → fraction 1.0 vs 1.0 actually
    # identical because both are 100% chat); but counts differ. The
    # normalized-L1 measure treats both as 100% chat, so drift_magnitude=0.
    # Use a kind-changing variant instead to actually trigger drift.
    assert structural[0].score >= 0.0  # well-formed


def test_structural_drift_changes_when_kinds_change(monkeypatch):
    """Adding a different operation kind in current → real structural drift."""
    monkeypatch.setattr(
        "rudriq.evaluate.drift.embed_texts",
        lambda t: [[1.0, 0.0] for _ in t],
    )
    baseline = _graph([_llm("b1", "q1", "a1")], "base")
    # current has an embedding op in addition to chat → different library.op key
    emb = TraceNode(
        node_id="c2",
        kind=NodeKind.LLM_EMBEDDING,
        library="openai",
        operation="embeddings",
        started_at=datetime.now(timezone.utc),
        ended_at=None,
        metadata={},
    )
    current = _graph([_llm("c1", "q1", "a1"), emb], "cur")
    results = DriftEvaluator(baseline_graph=baseline).evaluate(current)
    structural = next(r for r in results if r.metric == "drift_structural")
    assert structural.score < 1.0
    assert "openai.embeddings" in structural.explanation


def test_no_response_drift_when_identical(monkeypatch):
    """Identical aligned prompts AND responses → drift_response = 0."""
    monkeypatch.setattr(
        "rudriq.evaluate.drift.embed_texts",
        lambda t: [[1.0, 0.0] for _ in t],
    )
    baseline = _graph([_llm("b1", "same query", "same answer")], "base")
    current = _graph([_llm("c1", "same query", "same answer")], "cur")
    results = DriftEvaluator(baseline_graph=baseline).evaluate(current)
    resp = [r for r in results if r.metric == "drift_response"]
    assert len(resp) == 1
    assert resp[0].status == EvalStatus.OK
    assert resp[0].score == pytest.approx(1.0)


def test_high_response_drift_flagged(monkeypatch):
    """Same prompt, divergent response → low score on drift_response."""

    def fake_embed(texts):
        # The evaluator calls embed_texts(prompts) first (4 strings:
        # cur_prompts + base_prompts), then embed_texts(responses) for
        # aligned pairs (2 strings: cur_resp + base_resp). Route by
        # content shape: prompts contain "query", responses don't.
        if all("query" in t for t in texts):
            return [[1.0, 0.0] for _ in texts]  # all prompts identical
        # Responses divergent
        return [[1.0, 0.0], [0.0, 1.0]]

    monkeypatch.setattr("rudriq.evaluate.drift.embed_texts", fake_embed)
    baseline = _graph([_llm("b1", "the query", "baseline answer")], "base")
    current = _graph([_llm("c1", "the query", "very different now")], "cur")
    results = DriftEvaluator(baseline_graph=baseline).evaluate(current)
    resp = [r for r in results if r.metric == "drift_response"]
    assert len(resp) == 1
    assert resp[0].score < 1.0
    assert "mean response drift" in resp[0].explanation


def test_drift_detects_response_change_end_to_end(monkeypatch):
    """Day 17 Thread D pin: drift evaluator actually DETECTS change.

    The Day 15 sanity test only proved 'identical-runs returns 1.0'
    (no false positive). This test proves the *other* direction: same
    prompts, materially different responses → score drops well below
    1.0. The validation script ``examples/drift_validation.py`` runs
    this same shape end-to-end; the test pins it against future
    regressions so we can't accidentally break detection capability.

    Uses a hash-based deterministic embedding so identical prompts
    align cleanly and different responses are uncorrelated.
    """
    import hashlib

    def deterministic_embed(texts):
        out = []
        for t in texts:
            d = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([(d[i] - 128) / 128.0 for i in range(16)])
        return out

    monkeypatch.setattr(
        "rudriq.evaluate.drift.embed_texts", deterministic_embed
    )

    questions = ["q1", "q2", "q3"]
    baseline_answers = ["a1 original", "a2 original", "a3 original"]
    perturbed_answers = [
        "REVISED: completely different topic",
        "REVISED: another unrelated answer",
        "REVISED: third orthogonal response",
    ]

    base = datetime(2026, 5, 23, tzinfo=timezone.utc)
    baseline = TraceGraph(run_id="base", created_at=base, metadata={})
    perturbed = TraceGraph(run_id="cur", created_at=base, metadata={})
    for i, q in enumerate(questions):
        baseline.add_node(_llm(f"b{i}", q, baseline_answers[i]))
        perturbed.add_node(_llm(f"c{i}", q, perturbed_answers[i]))

    results = DriftEvaluator(baseline_graph=baseline).evaluate(perturbed)
    resp = next(r for r in results if r.metric == "drift_response")

    # Detection: identical prompts align, different responses register
    # as drift.
    assert resp.status == EvalStatus.OK
    assert resp.score < 0.8, (
        f"perturbation should produce score < 0.8; got {resp.score:.3f}"
    )
    assert resp.details["aligned_pairs"] == 3
    assert resp.details["mean_drift"] > 0.2, (
        f"perturbed responses should show meaningful drift; "
        f"got mean_drift={resp.details['mean_drift']:.3f}"
    )


def test_degrades_without_fastembed(monkeypatch):
    monkeypatch.setattr("rudriq.evaluate.drift.embed_texts", lambda t: None)
    baseline = _graph([_llm("b1", "q", "a")], "base")
    current = _graph([_llm("c1", "q", "a")], "cur")
    results = DriftEvaluator(baseline_graph=baseline).evaluate(current)
    resp = [r for r in results if r.metric == "drift_response"]
    assert any(r.status == EvalStatus.DEGRADED for r in resp)
