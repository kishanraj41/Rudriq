"""Base abstractions for the evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

from rudriq.core.schema import TraceGraph


class EvalStatus(str, Enum):
    """Outcome status of an evaluation.

    OK         — evaluator ran end-to-end; score is meaningful.
    SKIPPED    — evaluator could not apply (e.g. no LLM nodes linked
                 to upstream context). Not an error; benign no-op.
    DEGRADED   — evaluator ran but an optional dependency (fastembed)
                 was unavailable. score is None; explanation says so.
    ERROR      — evaluator raised an unexpected exception. The
                 explanation carries the exception type and message.
    """

    OK = "ok"
    SKIPPED = "skipped"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass
class EvalResult:
    """The result of one evaluator on one trace (or one node).

    An evaluator that scores each LLM node separately emits one
    ``EvalResult`` per node (with ``node_id`` set). An evaluator that
    produces a single trace-level result emits one ``EvalResult`` with
    ``node_id=None``.
    """

    metric: str
    status: EvalStatus
    score: Optional[float]
    explanation: str
    node_id: Optional[str] = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "status": self.status.value,
            "score": self.score,
            "explanation": self.explanation,
            "node_id": self.node_id,
            "details": self.details,
        }


@runtime_checkable
class Evaluator(Protocol):
    """Structural interface for an evaluator.

    An evaluator is any object with a ``metric`` name and an
    ``evaluate(graph)`` method that returns a list of ``EvalResult``.
    Returning a list (not a single result) lets an evaluator score
    each LLM node separately or produce one trace-level result.
    """

    metric: str

    def evaluate(self, graph: TraceGraph) -> list[EvalResult]:
        ...


def run_evaluators(
    graph: TraceGraph,
    evaluators: list[Evaluator],
) -> list[EvalResult]:
    """Run a list of evaluators against a trace, collecting all results.

    Each evaluator is isolated: if one raises, its failure becomes an
    ERROR ``EvalResult`` rather than aborting the whole evaluation.
    Mirrors the SpanProcessor's defensive philosophy — one broken
    evaluator must not break the others.
    """
    results: list[EvalResult] = []
    for ev in evaluators:
        try:
            results.extend(ev.evaluate(graph))
        except Exception as exc:  # noqa: BLE001
            results.append(
                EvalResult(
                    metric=getattr(ev, "metric", "unknown"),
                    status=EvalStatus.ERROR,
                    score=None,
                    explanation=f"Evaluator raised: {type(exc).__name__}: {exc}",
                )
            )
    return results
