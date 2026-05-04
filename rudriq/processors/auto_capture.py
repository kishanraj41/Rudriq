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

    The wrapper:
    1. Resolves the input value from kwargs (preferred) or positional args
       (fallback — last positional after self, best-effort).
    2. Reads the current OTel span's span_id from the active span context.
    3. Calls record_llm_input(span_id_hex, input_value).
    4. Delegates to the original method unchanged.

    If anything fails, the wrapper logs at DEBUG and falls through to
    the original method. Auto-capture is best-effort and never breaks
    the user's call.
    """
    @functools.wraps(original_method)
    def wrapped(*args, **kwargs):
        try:
            from rudriq.processors.linking import record_llm_input

            input_value = kwargs.get(input_param_name)
            # Fallback: when the user passed input positionally. args[0]
            # is self for bound methods; the LLM input (if positional)
            # is conventionally args[1] for create-style APIs.
            if input_value is None and len(args) > 1:
                input_value = args[1]

            if input_value is not None:
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
