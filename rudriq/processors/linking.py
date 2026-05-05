"""
RudriQSpanProcessor — the OTel SpanProcessor that runs the linker.

This is where RudriQ becomes more than a pile of primitives. The
SpanProcessor:

1. Watches every span the OpenTelemetry SDK creates.
2. When a span is a GenAI/LLM operation, extracts its input from the
   span attributes (or from a side-channel).
3. Runs the linker (rudriq.linker.correlate) to find the upstream
   data-lineage parent.
4. Annotates the span with rudriq.lineage_parent / rudriq.link_method
   / rudriq.link_confidence attributes.
5. Persists the span as a canonical TraceNode in DuckDB, with the
   appropriate cross-domain TraceEdge if a link was found.

Implementation note
-------------------
OTel SpanProcessor has two hooks that matter for us: on_start (called
when a span begins) and on_end (called when it completes). We do
linking on on_end because by then the input attributes are stable —
on_start, OpenLLMetry hasn't always finished setting attributes.

The trade-off: rudriq.lineage_parent is set after the span ends, so
real-time consumers of the span (a Datadog exporter, etc.) won't see
it on_end-1 but will see it on_end+0 because attributes are mutable
during on_end processing.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from rudriq.adapters import otel_span_to_node
from rudriq.core.schema import (
    EdgeKind,
    LinkMethod,
    TraceEdge,
)
from rudriq.core.span_attributes import (
    RUDRIQ_DOMAIN,
    RUDRIQ_LINEAGE_PARENT,
    RUDRIQ_LINK_CONFIDENCE,
    RUDRIQ_LINK_METHOD,
)
from rudriq.linker import correlate, is_genai_span
from rudriq.storage import get_default_storage

_LOG = logging.getLogger("rudriq.processor")


# ---------------------------------------------------------------------------
# Side-channel: stash LLM inputs by span_id so on_end can correlate them.
# ---------------------------------------------------------------------------
#
# Why this exists: OpenTelemetry spans don't natively carry the raw
# Python objects passed to openai.chat.completions.create(). They carry
# string/numeric *attributes* (model name, token counts, etc.). The
# raw input list is usually JSON-serialized into gen_ai.prompt.* attrs,
# which is too lossy for object_identity matching.
#
# In v0.0.3, we expose a helper (record_llm_input) that user-side
# instrumentation calls to register the actual Python input object
# alongside the span_id. Then on_end can look it up and run the linker
# against the real object, not against the serialized form.
#
# In v0.1, we'll add automatic instrumentation hooks that invoke
# record_llm_input transparently from inside the OpenAI SDK call site.
# Until then, the demo notebook shows the manual pattern.

_inputs_by_span_id: dict[str, Any] = {}
_inputs_lock = threading.Lock()


def record_llm_input(span_id: str, llm_input: Any) -> None:
    """
    Stash the raw Python input for a span so on_end can link it.

    Called by user-side instrumentation (or by future RudriQ auto-hooks)
    immediately before invoking the LLM SDK. Idempotent and best-effort:
    if span_id is already registered, the new value overwrites the old.
    """
    with _inputs_lock:
        _inputs_by_span_id[span_id] = llm_input


def _consume_llm_input(span_id: str) -> Any | None:
    with _inputs_lock:
        return _inputs_by_span_id.pop(span_id, None)


def clear_input_registry() -> None:
    """Test-only helper."""
    with _inputs_lock:
        _inputs_by_span_id.clear()


# ---------------------------------------------------------------------------
# The SpanProcessor itself
# ---------------------------------------------------------------------------


class RudriQSpanProcessor(SpanProcessor):
    """
    OTel SpanProcessor that links GenAI spans to upstream data lineage.

    Register this with the OTel TracerProvider once at startup:

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from rudriq.processors import RudriQSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(RudriQSpanProcessor())
        trace.set_tracer_provider(provider)
    """

    def __init__(self, run_id: str | None = None) -> None:
        self._run_id = run_id or uuid.uuid4().hex
        self._storage = get_default_storage()
        self._run_initialized = False
        self._lock = threading.Lock()

        # Wire our run_id into the AutoLineage mirroring callback so
        # subsequently-captured AutoLineage records land in the same run
        # as our LLM spans. Audit reports for in-process pipelines then
        # show full upstream chains without 'external' placeholders.
        # If rudriq.auto isn't imported yet (or autolineage isn't
        # installed), this is a graceful no-op.
        try:
            import rudriq.auto as _auto
            if hasattr(_auto, "_set_autolineage_run_id"):
                _auto._set_autolineage_run_id(self._run_id)
        except Exception:  # noqa: BLE001
            pass

    @property
    def run_id(self) -> str:
        return self._run_id

    def _ensure_run_exists(self) -> None:
        """Lazily register the run row on first span we see.

        We use storage.ensure_run rather than save_run because save_run
        treats the input graph as authoritative and DELETEs any
        pre-existing edges for the run_id. If a data-lineage adapter
        wrote nodes+edges under the same run_id before the first LLM
        span fired, calling save_run with an empty graph would wipe
        those edges. ensure_run only inserts the runs row.
        """
        if self._run_initialized:
            return
        with self._lock:
            if self._run_initialized:
                return
            self._storage.ensure_run(self._run_id)
            self._run_initialized = True

    def on_start(self, span, parent_context=None) -> None:  # noqa: D401
        # We do all work in on_end where attributes are stable.
        return

    def on_end(self, span: ReadableSpan) -> None:
        # Wrap the entire on_end body in a defensive try/except. on_end is
        # called by OpenTelemetry's MultiSpanProcessor, and a raise here
        # can prevent SUBSEQUENT processors from firing (and in some OTel
        # versions, crash the export pipeline). Observability must never
        # break the user's code, and one stale SpanProcessor with a
        # closed DuckDB handle (e.g., across pytest sessions where a
        # prior test's fixture closed its DB) must not poison sibling
        # processors. We log at DEBUG and move on.
        try:
            self._on_end_impl(span)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug(
                "RudriQSpanProcessor.on_end suppressed exception: %s", exc,
            )

    def _on_end_impl(self, span: ReadableSpan) -> None:
        self._ensure_run_exists()

        attrs = dict(span.attributes or {})
        span_name = span.name

        # Convert to canonical node.
        span_id_hex = format(span.context.span_id, "016x") if span.context else None
        node = otel_span_to_node(
            span_name=span_name,
            attributes=attrs,
            start_time_ns=span.start_time,
            end_time_ns=span.end_time,
            span_id=span_id_hex,
        )

        # If this is an LLM span, run the linker.
        if is_genai_span(span_name):
            llm_input = _consume_llm_input(span_id_hex) if span_id_hex else None
            parent_id, method, confidence = correlate(llm_input, attrs)

            if parent_id is not None:
                # Annotate node metadata for downstream consumers.
                node.metadata[RUDRIQ_LINEAGE_PARENT] = parent_id
                node.metadata[RUDRIQ_LINK_METHOD] = method
                node.metadata[RUDRIQ_LINK_CONFIDENCE] = confidence
                node.metadata[RUDRIQ_DOMAIN] = "linked"

                self._storage.save_node(node, run_id=self._run_id)
                self._storage.save_edge(
                    TraceEdge(
                        parent_id=parent_id,
                        child_id=node.node_id,
                        kind=EdgeKind.LINEAGE_LINK,
                        confidence=confidence,
                        link_method=LinkMethod(method),
                    ),
                    run_id=self._run_id,
                )
                _LOG.debug(
                    "Linked span %s to upstream node %s (method=%s, confidence=%.2f)",
                    span_id_hex, parent_id, method, confidence,
                )
                return

            node.metadata[RUDRIQ_DOMAIN] = "llm"

        # No link found, or not a GenAI span: persist as standalone node.
        self._storage.save_node(node, run_id=self._run_id)

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
