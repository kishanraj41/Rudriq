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
    """Activate AutoLineage if available. Returns True iff activated."""
    try:
        import autolineage.auto  # noqa: F401
        _LOG.info("RudriQ: AutoLineage activated for data pipeline tracing.")
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