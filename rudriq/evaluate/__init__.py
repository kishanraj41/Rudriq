"""
RudriQ evaluation engine.

Evaluators score the quality of LLM operations in a trace: groundedness,
retrieval relevance, drift, consistency, coherence. Each evaluator
produces a numeric score in [0, 1] plus a human-readable explanation.

All evaluators run locally. Those needing sentence embeddings use
fastembed (an optional ``evaluate`` extra); they degrade gracefully
when the extra is not installed, returning an ``EvalResult`` with
``status=DEGRADED`` and ``score=None`` whose explanation names the
missing dependency.

Design: evaluators operate on a ``TraceGraph`` (the same canonical
schema the linker and exporter use). They read LLM nodes, their
linked upstream context (retrieved documents), and the response, then
score.
"""

from rudriq.evaluate.base import (
    EvalResult,
    EvalStatus,
    Evaluator,
    run_evaluators,
)

__all__ = ["Evaluator", "EvalResult", "EvalStatus", "run_evaluators"]
