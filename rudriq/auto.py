"""
rudriq.auto — One-import activation.

    import rudriq.auto

This module:

1. Activates AutoLineage hooks if the ``[lineage]`` extra is installed,
   so data pipeline operations are captured.
2. Initializes Traceloop (OpenLLMetry) if the ``[llm]`` extra is installed,
   so LLM calls and RAG operations are captured.
3. Installs the RudriQ linker, which correlates spans from both domains
   into a unified OpenTelemetry trace.

Importing this module is idempotent — calling it twice has no extra effect.
"""

from __future__ import annotations

import logging

from rudriq.linker import install_linker

_LOG = logging.getLogger("rudriq")
_INSTALLED = False


def _activate_autolineage() -> bool:
    """Activate AutoLineage if available and wire RudriQ's auto-registration."""
    try:
        import autolineage.auto  # noqa: F401
        _LOG.info("RudriQ: AutoLineage activated for data pipeline tracing.")

        # Wire automatic object-identity registration. When AutoLineage
        # assigns a lineage ID to an output (in tracker.assign_id), it
        # fires our callback with (obj, lid). We mirror id(obj) -> lid
        # into RudriQ's linker registry so subsequent LLM calls can
        # correlate against it via object_identity.
        #
        # Why assign_id and not the more obvious "post-record" hook:
        # TransformationRecord carries only lineage IDs (strings), not
        # the underlying Python object. assign_id is the only place
        # AutoLineage holds both at the same time.
        try:
            from autolineage.auto import get_tracker as _get_al_tracker
            from rudriq.linker import register_object_identity

            tracker = _get_al_tracker()
            if tracker is None:
                _LOG.debug(
                    "RudriQ: AutoLineage tracker not yet initialized; "
                    "auto-registration callback not wired."
                )
            elif hasattr(tracker, "register_assign_id_callback"):
                tracker.register_assign_id_callback(register_object_identity)
                _LOG.info(
                    "RudriQ: auto-registration callback wired into AutoLineage."
                )
            else:
                _LOG.warning(
                    "RudriQ: AutoLineage version does not expose "
                    "register_assign_id_callback. Auto-registration disabled. "
                    "Upgrade AutoLineage to >=0.5 for automatic linking; "
                    "manual register_object_identity calls still work."
                )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "RudriQ: Failed to wire auto-registration callback: %s. "
                "Manual register_object_identity calls still work.",
                exc,
            )

        return True
    except ImportError:
        _LOG.debug(
            "RudriQ: AutoLineage not installed; data lineage capture disabled. "
            "Install with: pip install 'rudriq[lineage]'"
        )
        return False


def _activate_traceloop() -> bool:
    """Initialize Traceloop (OpenLLMetry) if available. Returns True iff activated."""
    try:
        from traceloop.sdk import Traceloop

        # Note: rudriq.processors.auto_capture.install_all() has already
        # run from _install() below, BEFORE we get here. That ordering
        # is load-bearing — it makes our patch innermost when Traceloop
        # wraps on top, so by the time our wrapper executes, the active
        # span context Traceloop established is readable.

        Traceloop.init(
            app_name="rudriq-app",
            disable_batch=True,
        )
        _LOG.info("RudriQ: Traceloop (OpenLLMetry) initialized for LLM tracing.")
        return True
    except ImportError:
        _LOG.debug(
            "RudriQ: Traceloop not installed; LLM call capture disabled. "
            "Install with: pip install 'rudriq[llm]'"
        )
        return False
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("RudriQ: Traceloop init failed: %s", exc)
        return False


def _install() -> None:
    """One-time activation of all available subsystems."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Install SDK auto-capture FIRST and unconditionally. This patches
    # openai/anthropic SDK methods (no-op if those SDKs aren't installed)
    # so that when Traceloop later wraps them, our patch is innermost
    # and runs inside the span context Traceloop creates. It also lets
    # users without Traceloop benefit by manually starting OTel spans.
    try:
        from rudriq.processors.auto_capture import install_all as _install_capture
        _install_capture()
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("Auto-capture install failed: %s", exc)

    lineage_active = _activate_autolineage()
    llm_active = _activate_traceloop()
    install_linker(lineage_enabled=lineage_active, llm_enabled=llm_active)

    # Register our SpanProcessor on the global TracerProvider so that
    # every OTel span (whether emitted by Traceloop, OpenInference, or
    # user code) flows through our linker.
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            from rudriq.processors import RudriQSpanProcessor
            provider.add_span_processor(RudriQSpanProcessor())
            _LOG.info("RudriQ SpanProcessor registered with global TracerProvider.")
        else:
            _LOG.warning(
                "Global TracerProvider is not an SDK TracerProvider; "
                "RudriQSpanProcessor not registered. "
                "Call rudriq.auto.install_processor(provider) manually."
            )
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("Failed to install RudriQ SpanProcessor: %s", exc)

    _INSTALLED = True

    if not lineage_active and not llm_active:
        _LOG.warning(
            "RudriQ: Neither AutoLineage nor Traceloop is installed. "
            "Install with: pip install 'rudriq[all]'"
        )


def install_processor(provider) -> None:
    """
    Public helper for users with a custom TracerProvider setup.

    Call this after creating your own TracerProvider but before
    importing rudriq.auto, or after if you set up your provider later.
    """
    from rudriq.processors import RudriQSpanProcessor
    provider.add_span_processor(RudriQSpanProcessor())


_install()