"""
The linker — RudriQ's core technical contribution.

The linker watches LLM spans (emitted by OpenLLMetry as OpenTelemetry
``gen_ai.*`` spans) and correlates their inputs back to data lineage
records (emitted by AutoLineage). When a match is found, it annotates
the OTel span with rudriq.* attributes describing the cross-domain link.

Three matching strategies, in order of confidence
-------------------------------------------------

1. **Object identity** (confidence 1.0): the LLM call's input is the
   *same Python object* (matched by ``id()``) as a known AutoLineage
   output. Highest confidence; works in-process for inputs that are
   passed directly without reconstruction.

2. **Content hash** (confidence 0.8): the LLM call's input has the same
   SHA-256 content hash as a known AutoLineage output. Survives object
   identity loss (e.g., DataFrame.copy(), list reconstruction). Costs
   one hash per LLM call.

3. **Name match** (confidence 0.5): heuristic correlation by variable
   name or column name when the above fail. Useful for traceability,
   not authoritative.

In v0.0.1, ``install_linker`` registers an OTel SpanProcessor that will
implement these strategies in v0.1. This file ships the API contract
and tests so v0.1 implementation has a target.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

_LOG = logging.getLogger("rudriq.linker")

# OpenTelemetry GenAI semantic convention span name prefixes.
# https://opentelemetry.io/docs/specs/semconv/gen-ai/
_GENAI_SPAN_PREFIXES = ("gen_ai.", "openai.", "anthropic.", "traceloop.")


class LinkerHook(Protocol):
    """Callable signature for a linker correlation hook."""

    def __call__(
        self,
        llm_input: Any,
        span_attributes: dict[str, Any],
    ) -> tuple[str | None, str, float]:
        """
        Attempt to find a lineage parent for ``llm_input``.

        Returns a tuple of (parent_node_id, link_method, confidence).
        If no match is found, parent_node_id is None.
        """
        ...


def link_by_object_identity(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """
    Match by Python ``id()``.

    v0.0.1: stub. v0.1 queries AutoLineage's UnifiedTracker for the
    object id and returns the AutoLineage node_id if found.
    """
    return None, "object_identity", 1.0


def link_by_content_hash(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """
    Match by SHA-256 content hash.

    v0.0.1: stub. v0.1 hashes ``llm_input`` (or its serialized
    representation) and checks AutoLineage's recorded hashes.
    """
    return None, "content_hash", 0.8


def link_by_name_match(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """
    Match by variable name or column name heuristic.

    v0.0.1: stub. v0.1 compares column names from the LLM input
    (e.g., a DataFrame slice) against AutoLineage-tracked DataFrames.
    """
    return None, "name_match", 0.5


_DEFAULT_STRATEGIES: tuple[LinkerHook, ...] = (
    link_by_object_identity,
    link_by_content_hash,
    link_by_name_match,
)


def correlate(
    llm_input: Any,
    span_attributes: dict[str, Any],
    strategies: tuple[LinkerHook, ...] = _DEFAULT_STRATEGIES,
) -> tuple[str | None, str, float]:
    """
    Apply each strategy in order; return the first match (highest
    confidence first), or (None, "", 0.0) if no strategy matches.
    """
    for strategy in strategies:
        parent_id, method, confidence = strategy(llm_input, span_attributes)
        if parent_id is not None:
            return parent_id, method, confidence
    return None, "", 0.0


def install_linker(
    *,
    lineage_enabled: bool,
    llm_enabled: bool,
) -> None:
    """
    Install the linker as an OTel SpanProcessor.

    In v0.0.1 this is a no-op; v0.1 registers a SpanProcessor that
    intercepts ``gen_ai.*`` spans and applies ``correlate`` to their
    inputs, attaching ``rudriq.lineage_parent`` to matched spans.

    Parameters
    ----------
    lineage_enabled:
        True iff AutoLineage was successfully imported in rudriq.auto.
    llm_enabled:
        True iff Traceloop (OpenLLMetry) was successfully initialized.
    """
    if not (lineage_enabled and llm_enabled):
        _LOG.debug(
            "RudriQ linker: not installed (lineage=%s, llm=%s). "
            "Both subsystems must be active for cross-domain linking.",
            lineage_enabled,
            llm_enabled,
        )
        return

    # v0.1: register OTel SpanProcessor here.
    _LOG.info("RudriQ linker: installed (v0.0.1 stub; full impl in v0.1).")


def is_genai_span(span_name: str) -> bool:
    """True iff a span name indicates an LLM operation per OTel GenAI conventions."""
    return any(span_name.startswith(p) for p in _GENAI_SPAN_PREFIXES)