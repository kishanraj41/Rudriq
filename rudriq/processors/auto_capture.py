"""
Automatic LLM input capture.

OpenLLMetry's instrumentors wrap LLM SDK methods to emit OTel spans.
This module adds a thin layer that, when an OpenLLMetry-instrumented
method is called, captures the raw input object and registers it with
RudriQ's record_llm_input side-channel by span_id.

Layer ordering (load-bearing)
-----------------------------
We install our wrappers BEFORE Traceloop.init() runs, NOT after. This
is the opposite of the obvious instinct (and an earlier draft of the
design). The reasoning:

* Traceloop's instrumentor uses ``wrapt.wrap_function_wrapper`` /
  class-attribute reassignment. Whichever patcher runs LAST becomes
  the OUTERMOST wrapper at call time.
* If we wrap last, we wrap on top of Traceloop. At call time our
  wrapper runs first — but Traceloop has not yet entered its span
  context, so ``trace.get_current_span()`` returns the no-op invalid
  span and our recording silently no-ops.
* If we wrap first, Traceloop wraps on top. At call time Traceloop's
  wrapper enters its span context, then calls the inner chain (us),
  and we read the active span_id correctly.

Practical consequence: ``rudriq.auto._activate_traceloop`` calls
``install_all()`` BEFORE ``Traceloop.init()``.

Best-effort, never breaks user code
-----------------------------------
Every wrapper is in a try/except that falls through to the original
method on any failure. Auto-capture is observability — if it breaks,
the user's actual LLM call must still succeed.

Monkey-patching is fragile. Pin OpenAI SDK to a tested range
(``openai>=1.0,<2.0`` per pyproject.toml). If a future OpenAI minor
release reshuffles ``openai.resources.embeddings``, our patch logs
and degrades; user's code still runs.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from opentelemetry import trace

_LOG = logging.getLogger("rudriq.auto_capture")
_INSTALLED = False


def _wrap_method_for_input_capture(original_method, input_param_name: str):
    """
    Wrap a method to record its input parameter to RudriQ's side-channel.

    Two-channel input stashing
    --------------------------
    The wrapper stashes the input in TWO ways, by design:

    1. **By span_id** when an active recording span is available. This is
       the precise correlation channel: SpanProcessor.on_end looks up by
       the same span_id and gets the exact input.
    2. **On the OTel context** unconditionally. This is the fallback for
       the common case where our wrapper runs OUTSIDE an active recording
       span — for example when OpenLLMetry's openai instrumentor uses
       suppression flags or its wrapt wrapping layer doesn't establish a
       current span context at the moment our (inner) wrapper executes.
       Empirically observed in opentelemetry-instrumentation-openai 0.60+
       where ``trace.get_current_span()`` at the wrapper entry returns
       the no-op NonRecordingSpan despite the instrumentor having
       ``with tracer.start_as_current_span(...)`` immediately above.

    Day 12 Phase C: context-keyed (replaces Day 8's thread-local)
    ------------------------------------------------------------
    Switched to OTel's ``context`` module, which is built on Python's
    ``contextvars``. ContextVars propagate correctly across threads,
    across asyncio tasks, and across OTel's own automatic span-context
    propagation. Each concurrent LLM call sees its own input, even with
    many calls in flight simultaneously. The thread-local was
    last-writer-wins per thread; the context-keyed approach is
    per-context.

    Lifecycle, important caveat
    ---------------------------
    The wrapper does NOT detach the context token after the call.
    Reason: ``on_end`` fires AFTER our wrapper returns (during
    Traceloop's ``__exit__`` of its ``start_as_current_span``), so
    detaching here makes the value disappear before ``on_end`` can
    retrieve it. Instead, each subsequent call's ``attach`` creates a
    new context layer; ``retrieve_input_from_context`` always returns
    the latest. Memory grows linearly with the number of LLM calls
    per context (~one small dict per call); bounded in practice and
    cleared when the asyncio task or thread exits. Tests that need
    strict lifecycle can call ``detach_input_from_context`` directly.
    """
    @functools.wraps(original_method)
    def wrapped(*args, **kwargs):
        try:
            from rudriq.processors.linking import (
                record_llm_input,
                stash_input_on_context,
            )

            input_value = kwargs.get(input_param_name)
            if input_value is None and len(args) > 1:
                input_value = args[1]

            if input_value is not None:
                # Primary fallback: attach the input on the OTel context
                # so on_end can read it back regardless of whether a
                # recording span is active here. We do NOT detach (see
                # docstring above — on_end fires after our wrapper
                # returns, so detaching would clear the value too early).
                stash_input_on_context(input_value)

                # Precise channel: ALSO stash by span_id when an active
                # recording span exists. Handles cooperating layering.
                current_span = trace.get_current_span()
                if current_span is not None and current_span.is_recording():
                    span_id_int = current_span.get_span_context().span_id
                    if span_id_int:
                        span_id_hex = format(span_id_int, "016x")
                        record_llm_input(span_id_hex, input_value)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug(
                "Auto-capture failed for %s: %s",
                getattr(original_method, "__name__", "<unknown>"), exc,
            )

        return original_method(*args, **kwargs)

    return wrapped


def install_openai_auto_capture() -> bool:
    """
    Install auto-capture wrappers on OpenAI SDK methods.

    Returns True if both wrappers were installed; False otherwise (SDK
    not importable, or class layout has changed in an incompatible way).
    Idempotent: subsequent calls re-wrap the already-wrapped attribute,
    which is harmless because functools.wraps preserves the inner.
    Production callers should rely on install_all's _INSTALLED guard
    instead of calling this directly.
    """
    try:
        from openai.resources.embeddings import Embeddings
        from openai.resources.chat.completions import Completions
    except ImportError:
        _LOG.debug("RudriQ: openai SDK not installed; auto-capture skipped.")
        return False

    try:
        Embeddings.create = _wrap_method_for_input_capture(
            Embeddings.create, input_param_name="input"
        )
        _LOG.info("RudriQ: auto-capture installed for openai.embeddings.create")

        Completions.create = _wrap_method_for_input_capture(
            Completions.create, input_param_name="messages"
        )
        _LOG.info(
            "RudriQ: auto-capture installed for openai.chat.completions.create"
        )
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("RudriQ: openai auto-capture install failed: %s", exc)
        return False


def install_anthropic_auto_capture() -> bool:
    """Install auto-capture wrappers on Anthropic SDK methods."""
    try:
        from anthropic.resources.messages import Messages
    except ImportError:
        _LOG.debug("RudriQ: anthropic SDK not installed; auto-capture skipped.")
        return False

    try:
        Messages.create = _wrap_method_for_input_capture(
            Messages.create, input_param_name="messages"
        )
        _LOG.info("RudriQ: auto-capture installed for anthropic.messages.create")
        return True
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("RudriQ: anthropic auto-capture install failed: %s", exc)
        return False


def install_pandas_lid_propagation() -> bool:
    """
    Wrap ``pd.DataFrame.__getitem__`` and ``pd.Series.tolist`` so that
    derived objects inherit their parent's RudriQ node_id.

    Why this exists: AutoLineage tracks DataFrame-level operations
    (read_csv, filter, merge, ...) but not column access (``df['col']``)
    or Python-list materialization (``series.tolist()``). When a user
    passes ``df['text'].tolist()`` to ``openai.embeddings.create``, the
    resulting list is a fresh object the linker has never seen, so
    object-identity matching fails even though the lineage is obvious.

    This patch propagates the parent's node_id through those two common
    pandas accessors. The result chain:
      df (registered by AutoLineage callback)
      -> df['col'] (registered here against df's node_id)
      -> series.tolist() (registered here against series's node_id == df's)

    We register through RudriQ's ``_object_registry`` directly (not
    through AutoLineage) because the propagated node_id is a derivation
    marker, not a new lineage operation. AutoLineage's record of the
    upstream operation is what the linker ultimately points to.
    """
    try:
        import pandas as pd
    except ImportError:
        _LOG.debug("RudriQ: pandas not installed; lid propagation skipped.")
        return False

    try:
        from rudriq.linker import _object_registry, register_object_identity
    except ImportError:
        return False

    orig_getitem = pd.DataFrame.__getitem__

    @functools.wraps(orig_getitem)
    def wrapped_getitem(self, key):
        result = orig_getitem(self, key)
        try:
            parent_id = _object_registry.get(id(self))
            if parent_id is not None and id(result) not in _object_registry:
                # Don't clobber: AutoLineage may already have assigned
                # a fresh lid (e.g. for boolean-mask filtering, which
                # AL tracks as a transformation). Only register results
                # that have no existing entry.
                register_object_identity(result, parent_id)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("getitem propagation failed: %s", exc)
        return result

    pd.DataFrame.__getitem__ = wrapped_getitem

    orig_tolist = pd.Series.tolist

    @functools.wraps(orig_tolist)
    def wrapped_tolist(self):
        result = orig_tolist(self)
        try:
            parent_id = _object_registry.get(id(self))
            if parent_id is not None:
                register_object_identity(result, parent_id)
                # Also register each element so list slicing preserves
                # linkage. Without this, `texts[batch_idx:batch_idx+N]`
                # produces a fresh list whose elements have no registry
                # entry; the linker's element-walk strategy then fails.
                # Cost: O(len) entries in the in-process registry, which
                # is acceptable for typical batch sizes (10-10k). For
                # very large lists (>=100k elements), we skip per-element
                # registration to avoid memory pressure — the whole-list
                # registration still works for the no-slice case.
                if len(result) < 100_000:
                    for elem in result:
                        if id(elem) not in _object_registry:
                            register_object_identity(elem, parent_id)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("tolist propagation failed: %s", exc)
        return result

    pd.Series.tolist = wrapped_tolist

    _LOG.info(
        "RudriQ: pandas lid propagation installed "
        "(DataFrame.__getitem__, Series.tolist)."
    )
    return True


def install_all() -> None:
    """Install all available auto-capture wrappers. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return

    install_openai_auto_capture()
    install_anthropic_auto_capture()
    install_pandas_lid_propagation()
    _INSTALLED = True


def _reset_for_tests() -> None:
    """Test-only: reset the install flag so tests can drive install_all
    deterministically. Does NOT undo class-attribute patches — those are
    persistent module-level side effects."""
    global _INSTALLED
    _INSTALLED = False
