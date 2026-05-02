"""
Convert OpenTelemetry spans into RudriQ canonical TraceNode objects.

OpenLLMetry emits spans following the OpenTelemetry GenAI semantic
conventions (gen_ai.* attributes). Other instrumentations (OpenInference,
Datadog APM, custom user spans) follow similar but slightly different
conventions. This adapter handles the GenAI conventions plus a few
common variants.

Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rudriq.core.schema import NodeKind, TraceNode


# ---------------------------------------------------------------------------
# Span name -> NodeKind classification
# ---------------------------------------------------------------------------


def classify_span_kind(span_name: str, attributes: dict[str, Any]) -> NodeKind:
    """
    Heuristically determine the canonical NodeKind for an OTel span.

    Examines the span name first (fast path), then falls back to
    attributes (gen_ai.operation.name, gen_ai.system) for ambiguous
    cases.
    """
    name_lower = span_name.lower()

    # GenAI semantic conventions: span names like "chat <model>"
    # or "embeddings <model>" or vendor-prefixed like "openai.chat".
    if "embedding" in name_lower:
        return NodeKind.LLM_EMBEDDING
    if "chat" in name_lower or "completion" in name_lower or "message" in name_lower:
        return NodeKind.LLM_CHAT
    if "retriev" in name_lower or "search" in name_lower:
        return NodeKind.RETRIEVAL
    if "prompt" in name_lower:
        return NodeKind.PROMPT_ASSEMBLY

    # Fallback: gen_ai.operation.name attribute (per spec).
    op = attributes.get("gen_ai.operation.name", "").lower()
    if op == "embeddings":
        return NodeKind.LLM_EMBEDDING
    if op in ("chat", "text_completion"):
        return NodeKind.LLM_CHAT

    # System-level fallback: presence of gen_ai.system suggests an LLM op
    # even when the span name is generic ("openai.client.request").
    if "gen_ai.system" in attributes:
        return NodeKind.LLM_COMPLETION

    return NodeKind.UNKNOWN


# ---------------------------------------------------------------------------
# Span -> TraceNode conversion
# ---------------------------------------------------------------------------


def _ns_to_datetime(timestamp_ns: int) -> datetime:
    """OTel spans use nanoseconds-since-epoch; convert to UTC datetime."""
    return datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)


def _extract_library(span_name: str, attributes: dict[str, Any]) -> str:
    """
    Determine which framework/SDK emitted this span.

    Returns a normalized library name. The library is conceptually the
    "system" (openai, anthropic, pandas, ...), distinct from the
    operation (chat, embeddings, read_csv).
    """
    if "gen_ai.system" in attributes:
        return str(attributes["gen_ai.system"]).lower()
    name_lower = span_name.lower()
    for prefix in ("openai", "anthropic", "google", "cohere", "mistral"):
        if prefix in name_lower:
            return prefix
    return "unknown"


def _normalize_operation(span_name: str, library: str) -> str:
    """
    Strip a redundant library prefix from the span name when present.

    OpenLLMetry sometimes emits "openai.embeddings.create" with
    gen_ai.system="openai", producing a doubled rendering. Other
    instrumentations emit "gen_ai.embeddings.create" or just
    "embeddings.create". We normalize so that the rendered
    f"{library}.{operation}" string is always sensible AND so that the
    analyzer's cross-run fingerprinting (which uses operation as part
    of the key) is stable across instrumentation upgrades.

    Examples:
        ("openai.embeddings.create", "openai") -> "embeddings.create"
        ("gen_ai.embeddings.create", "openai") -> "embeddings.create"
        ("embeddings.create", "openai")        -> "embeddings.create"
        ("openai.chat", "openai")              -> "chat"
        ("custom.span.name", "unknown")        -> "custom.span.name"
    """
    op = span_name
    # Strip library-name prefix if present.
    lib_prefix = f"{library}."
    if op.startswith(lib_prefix):
        op = op[len(lib_prefix):]
    # Strip "gen_ai." prefix if present (semantic-conventions style).
    if op.startswith("gen_ai."):
        op = op[len("gen_ai."):]
    return op


def _extract_metadata(attributes: dict[str, Any]) -> dict[str, Any]:
    """
    Pull a useful subset of GenAI semantic attributes into our metadata
    field. We intentionally don't copy everything — that bloats storage
    and most attributes are debug-only.
    """
    keep_keys = (
        "gen_ai.system",
        "gen_ai.operation.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.request.temperature",
        "gen_ai.request.top_p",
        "gen_ai.request.max_tokens",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.prompt_tokens",
        "gen_ai.usage.completion_tokens",
        "gen_ai.usage.total_tokens",
        "gen_ai.response.finish_reasons",
        "gen_ai.response.id",
        # OpenLLMetry-specific (some duplicate the standard above)
        "llm.request.type",
        "llm.request.model",
        "llm.usage.total_tokens",
        "traceloop.workflow.name",
    )
    out: dict[str, Any] = {}
    for k in keep_keys:
        if k in attributes:
            out[k] = attributes[k]
    return out


def otel_span_to_node(
    span_name: str,
    attributes: dict[str, Any],
    start_time_ns: int,
    end_time_ns: int | None = None,
    span_id: str | None = None,
) -> TraceNode:
    """
    Convert one OTel span into a canonical TraceNode.

    Parameters
    ----------
    span_name:
        The name of the OTel span (e.g. "openai.chat", "gen_ai.embeddings").
    attributes:
        The flat attributes dict from the span. RudriQ-specific attrs
        (rudriq.lineage_parent etc.) are preserved into metadata as well.
    start_time_ns:
        Span start timestamp, in nanoseconds since Unix epoch.
    end_time_ns:
        Span end timestamp, or None if the span is still open.
    span_id:
        OTel span_id, used as the canonical node_id when present. If
        None, caller is expected to provide a uuid.

    Returns
    -------
    A TraceNode populated from the span. Caller is responsible for
    persisting it via storage.save_node and adding edges as needed.
    """
    if span_id is None:
        import uuid
        span_id = uuid.uuid4().hex

    kind = classify_span_kind(span_name, attributes)
    library = _extract_library(span_name, attributes)
    operation = _normalize_operation(span_name, library)
    metadata = _extract_metadata(attributes)

    # Preserve any rudriq.* attributes the linker added.
    for k, v in attributes.items():
        if k.startswith("rudriq."):
            metadata[k] = v

    return TraceNode(
        node_id=span_id,
        kind=kind,
        library=library,
        operation=operation,
        started_at=_ns_to_datetime(start_time_ns),
        ended_at=_ns_to_datetime(end_time_ns) if end_time_ns else None,
        metadata=metadata,
    )
