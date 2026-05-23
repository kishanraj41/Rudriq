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

from datetime import datetime
from typing import Any

from rudriq.core.schema import NodeKind, TraceNode, otel_nanos_to_utc


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
    """OTel spans use nanoseconds-since-epoch; convert to UTC datetime.

    Thin wrapper around the canonical helper in rudriq.core.schema so the
    timezone boundary at OTel ingestion is documented in one place.
    """
    return otel_nanos_to_utc(timestamp_ns)


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


def _parse_genai_messages(messages_json: str) -> dict[str, str]:
    """Parse OpenLLMetry's gen_ai.input.messages JSON into role-separated text.

    openllmetry-openai 0.60+ (matching the OTel GenAI semconv) emits
    ``gen_ai.input.messages`` / ``gen_ai.output.messages`` as JSON
    strings of the form::

        [{"role": "user", "parts": [{"type": "text", "content": "..."}]}, ...]

    Returns a dict with three keys:

    * ``full``   — every text chunk, in document order, newline-joined
    * ``user``   — text from ``role == "user"`` messages only
    * ``system`` — text from ``role == "system"`` messages only

    The user message is the actual question / instruction, distinct
    from the system prompt and any assembled retrieval context.
    Day 15 surfaced that the assembled prompt is dominated by shared
    retrieval context across distinct RAG queries, which makes
    similarity-based grouping (consistency) collapse them spuriously.
    Day 17 Thread B fixes that by exposing the user-role text
    separately for evaluators that want the bare question.

    Non-JSON or unexpected shapes return all-empty so callers can
    SKIP gracefully without distinguishing parse failures.
    """
    import json

    result = {"full": "", "user": "", "system": ""}
    try:
        messages = json.loads(messages_json)
    except (TypeError, ValueError):
        return result
    if not isinstance(messages, list):
        return result

    full_parts: list[str] = []
    user_parts: list[str] = []
    system_parts: list[str] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")

        # Extract this message's text content (handling both the new
        # parts-based shape and the older content-string shape).
        text = ""
        parts = msg.get("parts")
        if isinstance(parts, list):
            chunks: list[str] = []
            for part in parts:
                if isinstance(part, dict):
                    if part.get("type", "text") == "text":
                        content = part.get("content")
                        if isinstance(content, str) and content.strip():
                            chunks.append(content)
            text = "\n".join(chunks)
        else:
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                text = content

        if not text:
            continue

        full_parts.append(text)
        if role == "user":
            user_parts.append(text)
        elif role == "system":
            system_parts.append(text)

    result["full"] = "\n".join(full_parts)
    result["user"] = "\n".join(user_parts)
    result["system"] = "\n".join(system_parts)
    return result


def _extract_messages_text(messages_json: str) -> str:
    """Backward-compatible wrapper — returns the ``full`` blob only.

    Retained so existing call sites (``_extract_input_content``,
    ``_extract_output_content``) keep working unchanged. New call
    sites that need role separation should use ``_parse_genai_messages``
    directly.
    """
    return _parse_genai_messages(messages_json).get("full", "")


def _gather_indexed_content(attributes: dict[str, Any], prefix: str) -> str:
    """Legacy: collect content from indexed gen_ai attributes.

    openllmetry < 0.60 emitted ``gen_ai.prompt.0.content``,
    ``gen_ai.prompt.1.content``, etc. Newer versions use the
    JSON-encoded ``gen_ai.input.messages`` shape handled by
    ``_extract_messages_text``. Keep this helper as a fallback so we
    work across instrumentor versions.
    """
    items: list[tuple[int, str]] = []
    suffix = ".content"
    for key, val in attributes.items():
        if key.startswith(prefix) and key.endswith(suffix):
            if not isinstance(val, str):
                continue
            parts = key.split(".")
            try:
                idx = int(parts[-2])
            except (ValueError, IndexError):
                idx = 0
            items.append((idx, val))
    items.sort(key=lambda t: t[0])
    return "\n".join(v for _, v in items)


def _extract_input_content(attributes: dict[str, Any]) -> str:
    """Pull prompt/input text from a span's attributes across semconv versions."""
    new = attributes.get("gen_ai.input.messages")
    if isinstance(new, str):
        text = _extract_messages_text(new)
        if text:
            return text
    return _gather_indexed_content(attributes, prefix="gen_ai.prompt")


def _extract_output_content(attributes: dict[str, Any]) -> str:
    """Pull completion/output text from a span's attributes across semconv versions."""
    new = attributes.get("gen_ai.output.messages")
    if isinstance(new, str):
        text = _extract_messages_text(new)
        if text:
            return text
    return _gather_indexed_content(attributes, prefix="gen_ai.completion")


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

    # Content previews — ONLY when capture is explicitly enabled. OpenLLMetry
    # stores prompt/completion as gen_ai.prompt.N.content / gen_ai.completion.N.content
    # when its own content-capture env var is set. We mirror that into a
    # RudriQ-namespaced, truncated preview so it's unambiguously OUR stored
    # copy (not raw OTel) and downstream consumers know it's bounded.
    from rudriq.core.config import content_capture_enabled, truncate_preview

    if content_capture_enabled():
        # Preferred path: parse the new GenAI semconv messages JSON so
        # we can also isolate the user-role text (Day 17 Thread B —
        # the assembled prompt is dominated by shared retrieval
        # context across distinct queries, which confounds
        # similarity-based grouping like ConsistencyEvaluator).
        input_messages_json = attributes.get("gen_ai.input.messages")
        if isinstance(input_messages_json, str) and input_messages_json:
            parsed = _parse_genai_messages(input_messages_json)
            if parsed["full"]:
                metadata["rudriq.prompt_preview"] = truncate_preview(parsed["full"])
            if parsed["user"]:
                metadata["rudriq.user_message_preview"] = truncate_preview(parsed["user"])
        else:
            # Legacy openllmetry (<0.60): only the indexed shape is
            # available; we can't reliably separate roles from there,
            # so user_message_preview is omitted in this branch.
            prompt_text = _gather_indexed_content(attributes, prefix="gen_ai.prompt")
            if prompt_text:
                metadata["rudriq.prompt_preview"] = truncate_preview(prompt_text)

        completion_text = _extract_output_content(attributes)
        if completion_text:
            metadata["rudriq.completion_preview"] = truncate_preview(completion_text)

    return TraceNode(
        node_id=span_id,
        kind=kind,
        library=library,
        operation=operation,
        started_at=_ns_to_datetime(start_time_ns),
        ended_at=_ns_to_datetime(end_time_ns) if end_time_ns else None,
        metadata=metadata,
    )
