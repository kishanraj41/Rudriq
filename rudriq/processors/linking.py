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
from collections import OrderedDict
from typing import Any

from opentelemetry import context as otel_context
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

# Peek semantics with bounded LRU (Day 12 Phase C polish)
# ------------------------------------------------------
# Originally _consume_llm_input popped. That broke when the user wires
# MULTIPLE RudriQSpanProcessors against the same TracerProvider — e.g.,
# rudriq.auto adds one at import time AND user code adds a second to
# scope a run_id. The first processor's pop emptied the channel before
# the second could read it, so only one run got linked edges.
#
# Empirically also needed because Traceloop's openllmetry-openai
# *embeddings* wrapper detaches the OTel context before span.end()
# fires on_end, so the context-keyed fallback (below) returns None for
# embedding spans (it still works for chat spans). With pop semantics
# AND ctx-MISS, the second processor saw nothing.
#
# Switching to peek + bounded LRU lets every processor see the same
# input; old entries evict naturally as new spans flow in, so memory
# stays bounded even for long-running processes that never explicitly
# clear the channel.
_INPUT_REGISTRY_MAX = 1000
_inputs_by_span_id: "OrderedDict[str, Any]" = OrderedDict()
_inputs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# OTel context-keyed input fallback (Day 12 Phase C)
# ---------------------------------------------------------------------------
#
# Replaces the Day 8 thread-local "most recent input" fallback. The
# thread-local worked single-threaded but degraded under concurrency:
# one thread's input could mask another's, and nested LLM calls
# overwrote each other before on_end fired.
#
# The new mechanism uses OTel's context module, which is implemented
# on top of Python's ``contextvars.ContextVar``. ContextVars propagate
# correctly across:
#   * threads (each thread has an independent copy of the context)
#   * asyncio tasks (each task inherits but writes its own copy)
#   * OTel's automatic span-context propagation
#
# Lifecycle (subtle): the auto_capture wrapper attaches the input on
# the OTel context before calling the SDK method. It does NOT detach
# after the call returns — see the auto_capture docstring for why
# (on_end fires AFTER Traceloop's outer span context detach, so a
# wrapper-side detach would clear the value before on_end can read it).
# Subsequent calls' attach stacks new layers on top; old ones are
# bounded by the LRU on the by-span-id channel and by natural unwinding
# when the asyncio task / thread exits.
#
# Note: this fallback works for chat spans (ctx is still visible at
# on_end) but NOT for openllmetry-openai embedding spans (their wrapper
# detaches the context before span.end). The peek-based by-span-id
# channel above is what carries embeddings through.
_LLM_INPUT_CONTEXT_KEY = otel_context.create_key("rudriq.llm_input")


def stash_input_on_context(raw_input: Any):
    """Attach an LLM input to the current OTel context.

    Returns a token that MUST be passed to ``detach_input_from_context``
    when the LLM call completes. Concurrency-safe via Python's
    contextvars: each thread / asyncio task / OTel context sees its
    own value, even with many concurrent calls in flight.
    """
    return otel_context.attach(
        otel_context.set_value(_LLM_INPUT_CONTEXT_KEY, raw_input)
    )


def retrieve_input_from_context() -> Any | None:
    """Read the LLM input from the current OTel context, or None if absent.

    Non-destructive: multiple reads in the same context return the same
    value. The wrapper-managed detach is what eventually clears it.
    """
    return otel_context.get_value(_LLM_INPUT_CONTEXT_KEY)


def detach_input_from_context(token) -> None:
    """Restore the OTel context to its state before ``stash_input_on_context``."""
    otel_context.detach(token)


def record_llm_input(span_id: str, llm_input: Any) -> None:
    """
    Stash the raw Python input for a span so on_end can link it.

    Called by user-side instrumentation (or by future RudriQ auto-hooks)
    immediately before invoking the LLM SDK. Idempotent and best-effort:
    if span_id is already registered, the new value overwrites the old.

    LRU-bounded: when the registry exceeds ``_INPUT_REGISTRY_MAX`` entries,
    the oldest entry is evicted. New writes refresh LRU position.
    """
    with _inputs_lock:
        if span_id in _inputs_by_span_id:
            _inputs_by_span_id.move_to_end(span_id)
        _inputs_by_span_id[span_id] = llm_input
        while len(_inputs_by_span_id) > _INPUT_REGISTRY_MAX:
            _inputs_by_span_id.popitem(last=False)


def _consume_llm_input(span_id: str) -> Any | None:
    """Peek (non-destructive) the input stashed for ``span_id``.

    Despite the name, this does NOT pop — see the registry comment above
    for why peek semantics are required (multiple SpanProcessors). The
    entry remains until LRU-evicted by subsequent inserts.
    """
    with _inputs_lock:
        value = _inputs_by_span_id.get(span_id)
        if value is not None:
            _inputs_by_span_id.move_to_end(span_id)
        return value


def clear_input_registry() -> None:
    """Test-only helper: clears the by-span_id channel. The OTel
    context-keyed fallback is per-context, so each test naturally
    starts fresh — no explicit clearing needed."""
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
            if llm_input is None:
                # Fallback: OTel context-keyed input (Day 12 Phase C).
                # The auto_capture wrapper attached the input to the
                # current context before calling the SDK method; we
                # read it back here. Concurrency-safe because
                # contextvars give each thread / asyncio task its own
                # value.
                llm_input = retrieve_input_from_context()
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
