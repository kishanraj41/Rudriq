"""Tests for the evaluation framework (base abstractions + isolation)."""

from __future__ import annotations

from datetime import datetime, timezone

from rudriq.core.schema import TraceGraph
from rudriq.evaluate.base import (
    EvalResult,
    EvalStatus,
    Evaluator,
    run_evaluators,
)


def _empty_graph() -> TraceGraph:
    return TraceGraph(
        run_id="eval-test-run",
        created_at=datetime.now(timezone.utc),
        metadata={},
        nodes=[],
        edges=[],
    )


def test_eval_result_to_dict():
    r = EvalResult(
        metric="test", status=EvalStatus.OK, score=0.8,
        explanation="all good",
    )
    d = r.to_dict()
    assert d["metric"] == "test"
    assert d["status"] == "ok"
    assert d["score"] == 0.8
    assert d["node_id"] is None
    assert d["details"] == {}


def test_run_evaluators_collects_all():
    class FakeEval:
        metric = "fake"

        def evaluate(self, graph):
            return [EvalResult("fake", EvalStatus.OK, 1.0, "fine")]

    results = run_evaluators(_empty_graph(), [FakeEval(), FakeEval()])
    assert len(results) == 2
    assert all(r.status == EvalStatus.OK for r in results)


def test_run_evaluators_isolates_failures():
    """One evaluator raising must not abort the others."""

    class GoodEval:
        metric = "good"

        def evaluate(self, graph):
            return [EvalResult("good", EvalStatus.OK, 1.0, "fine")]

    class BadEval:
        metric = "bad"

        def evaluate(self, graph):
            raise RuntimeError("intentional failure")

    results = run_evaluators(_empty_graph(), [GoodEval(), BadEval(), GoodEval()])
    assert len(results) == 3
    statuses = [r.status for r in results]
    assert statuses.count(EvalStatus.OK) == 2
    assert statuses.count(EvalStatus.ERROR) == 1
    error_result = next(r for r in results if r.status == EvalStatus.ERROR)
    assert "intentional failure" in error_result.explanation
    assert error_result.metric == "bad"


def test_evaluator_protocol_structural():
    """Anything with ``metric`` + ``evaluate`` satisfies the Protocol."""

    class Duck:
        metric = "duck"

        def evaluate(self, graph):
            return []

    assert isinstance(Duck(), Evaluator)
