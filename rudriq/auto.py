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
from datetime import datetime, timezone
from typing import Any

from rudriq.linker import install_linker

_LOG = logging.getLogger("rudriq")
_INSTALLED = False

# Holds the run_id under which AutoLineage records get mirrored. Set by
# RudriQSpanProcessor.__init__ so subsequently-captured AutoLineage records
# land in the same run as the LLM spans. Module-level dict so the closure
# in the callback can read the current value (rather than a stale captured
# binding).
_autolineage_run_id: dict[str, str | None] = {"value": None}


def _set_autolineage_run_id(run_id: str | None) -> None:
    """Wire the run_id under which AutoLineage records will be mirrored.

    Called by RudriQSpanProcessor.__init__ so subsequently-captured
    AutoLineage records land in the same run as our LLM spans. Audit
    reports for in-process pipelines then show full upstream chains
    without 'external' placeholders. Pass None to disable mirroring.
    """
    _autolineage_run_id["value"] = run_id


# AutoLineage uses category strings ("io", "transform"); RudriQ uses
# NodeKind enum. This mapping decides what each AL record becomes.
_AL_READ_OPS = (
    "read_csv", "read_parquet", "read_json", "read_excel", "read_sql",
    "read_feather", "read_orc", "read_pickle", "read_table", "read_hdf",
    "load", "from_pandas", "from_dict",
)
_AL_WRITE_OPS = (
    "to_csv", "to_parquet", "to_json", "to_excel", "to_sql", "to_pickle",
    "to_feather", "to_hdf", "to_orc", "save",
)


def _autolineage_record_to_node_kind(record: Any) -> Any:
    """Map an AutoLineage TransformationRecord to a NodeKind."""
    from rudriq.core.schema import NodeKind

    category = getattr(record, "category", "") or ""
    operation = getattr(record, "operation", "") or ""

    if category == "io":
        op_lower = operation.lower()
        if any(w in op_lower for w in _AL_WRITE_OPS):
            return NodeKind.DATA_WRITE
        return NodeKind.DATA_READ
    if category == "transform":
        return NodeKind.DATA_TRANSFORM
    if category == "model_train":
        return NodeKind.MODEL_TRAIN
    if category == "model_predict":
        return NodeKind.MODEL_PREDICT
    return NodeKind.UNKNOWN


def _autolineage_record_to_started_at(record: Any) -> datetime:
    """Derive a UTC datetime from the record's timestamp string.

    AutoLineage's TransformationRecord.timestamp is set by
    ``datetime.now().isoformat()`` — a NAIVE local-time string. Day 7's
    bug was labeling it as UTC via ``.replace(tzinfo=timezone.utc)``,
    producing 5-hour errors on non-UTC systems. The canonical conversion
    lives in rudriq.core.schema.autolineage_timestamp_to_utc.
    """
    from rudriq.core.schema import autolineage_timestamp_to_utc

    ts = getattr(record, "timestamp", None)
    if ts:
        try:
            return autolineage_timestamp_to_utc(datetime.fromisoformat(ts))
        except (ValueError, TypeError):
            pass
    return datetime.now(timezone.utc)


def _make_mirror_callback(al_tracker):
    """Build a post-record callback that mirrors each AutoLineage record
    as a TraceNode + parent DIRECT edges in RudriQ's DuckDB.

    Closes over ``al_tracker`` so the callback can read tracker.nodes
    metadata (shape, columns, content_hash) at fire time.
    """
    from rudriq.core.schema import EdgeKind, LinkMethod, TraceEdge, TraceNode
    from rudriq.linker import register_object_identity

    def _on_record(record):
        run_id = _autolineage_run_id["value"]
        if run_id is None:
            return

        try:
            from rudriq.storage import get_default_storage
            storage = get_default_storage()

            child_id = getattr(record, "child_id", None)
            if not child_id:
                return

            # Pull richer metadata from the AL tracker's nodes dict.
            al_node = (al_tracker.nodes.get(child_id) or {}) if al_tracker else {}
            metadata = dict(getattr(record, "metadata", {}) or {})
            for key in ("shape", "columns", "content_hash", "source", "filepath"):
                v = al_node.get(key)
                if v is not None:
                    metadata.setdefault(f"autolineage.{key}", v)

            duration_ms = getattr(record, "duration_ms", None)
            if duration_ms is not None:
                metadata.setdefault("autolineage.duration_ms", duration_ms)

            started_at = _autolineage_record_to_started_at(record)
            if duration_ms:
                from datetime import timedelta
                ended_at = started_at + timedelta(milliseconds=float(duration_ms))
            else:
                ended_at = started_at

            node = TraceNode(
                node_id=child_id,
                kind=_autolineage_record_to_node_kind(record),
                library=getattr(record, "library", "autolineage") or "autolineage",
                operation=getattr(record, "operation", "unknown") or "unknown",
                started_at=started_at,
                ended_at=ended_at,
                metadata=metadata,
                content_hash=al_node.get("content_hash"),
            )

            storage.ensure_run(run_id)
            storage.save_node(node, run_id=run_id)

            # Mirror parent_ids as DIRECT edges so the chain
            # parent_op -> child_op is visible in audit reports.
            for parent_id in (getattr(record, "parent_ids", None) or []):
                if not parent_id:
                    continue
                storage.save_edge(
                    TraceEdge(
                        parent_id=parent_id,
                        child_id=child_id,
                        kind=EdgeKind.DIRECT,
                        confidence=1.0,
                        link_method=LinkMethod.UNKNOWN,
                    ),
                    run_id=run_id,
                )

            # ALSO register object identity for the record's child if we
            # can resolve it to a live Python object via AL's _lid_to_obj
            # weak-value dict. Many AL hooks call record() WITHOUT firing
            # assign_id for the result (sort_values, drop_duplicates,
            # head, reset_index, ...) — without this lookup, those
            # outputs never enter rudriq's _object_registry and the
            # downstream linker's identity match silently fails.
            try:
                lid_to_obj = getattr(al_tracker, "_lid_to_obj", None)
                if lid_to_obj is not None:
                    obj = lid_to_obj.get(child_id)
                    if obj is not None:
                        register_object_identity(obj, child_id)
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            _LOG.debug(
                "RudriQ mirror callback failed for record %s: %s",
                getattr(record, "child_id", "<unknown>"), exc,
            )

    return _on_record


def _make_assign_id_stub_callback(al_tracker):
    """Build an assign_id callback that registers identity AND saves a
    stub TraceNode for the assigned object.

    Why this exists: AutoLineage's read_csv (and similar source hooks)
    fire ``assign_id`` for the resulting DataFrame but never call
    ``record()``. Without this stub, the read_csv output gets identity-
    registered (so the linker can match it) but never appears as a
    TraceNode in DuckDB — audit reports show 'unmirrored' placeholders
    when the chain walks back to a read_csv.

    The post_record callback's mirror later upserts with fuller info
    (library, operation, parent_ids, duration_ms) when a record fires
    for the same child_id. INSERT OR REPLACE in save_node makes this
    safe — the latest write wins.
    """
    from rudriq.core.schema import NodeKind, TraceNode
    from rudriq.linker import register_object_identity

    def _on_assign_id(obj, lid):
        # 1. Object identity registration — unchanged.
        register_object_identity(obj, lid)

        # 2. Save a stub TraceNode if mirroring is wired.
        run_id = _autolineage_run_id["value"]
        if run_id is None:
            return

        try:
            from rudriq.storage import get_default_storage
            storage = get_default_storage()

            al_node = (al_tracker.nodes.get(lid) or {}) if al_tracker else {}
            source = str(al_node.get("source") or "unknown")

            # Heuristically classify by source string. Most read_csv-
            # style hooks pass source like "pandas.read_csv"; fall back
            # to UNKNOWN for unfamiliar sources. The post_record callback
            # will overwrite with a more specific kind when it fires.
            source_lower = source.lower()
            if any(w in source_lower for w in ("read_", "load", "from_")):
                kind = NodeKind.DATA_READ
            elif any(w in source_lower for w in ("write", "to_", "save")):
                kind = NodeKind.DATA_WRITE
            elif source_lower in ("untracked", "unknown"):
                # No info from AL; conservative default. Often this is
                # actually a read_csv (the AL hook didn't pass source);
                # mark as DATA_READ.
                kind = NodeKind.DATA_READ
            else:
                kind = NodeKind.DATA_TRANSFORM

            metadata: dict[str, Any] = {}
            for key in ("shape", "columns", "filepath", "source"):
                v = al_node.get(key)
                if v is not None:
                    metadata[f"autolineage.{key}"] = v

            # Try to extract a clean operation name from the source.
            # "pandas.read_csv" → operation="read_csv"; otherwise pass
            # the raw source through.
            if "." in source:
                library, _, operation = source.partition(".")
            else:
                library = "autolineage"
                operation = source

            node = TraceNode(
                node_id=lid,
                kind=kind,
                library=library or "autolineage",
                operation=operation or "tracked",
                started_at=datetime.now(timezone.utc),
                ended_at=datetime.now(timezone.utc),
                metadata=metadata,
                content_hash=al_node.get("content_hash"),
            )

            storage.ensure_run(run_id)
            storage.save_node(node, run_id=run_id)
        except Exception as exc:  # noqa: BLE001
            _LOG.debug(
                "RudriQ assign_id stub mirror failed for lid %s: %s",
                lid, exc,
            )

    return _on_assign_id


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
                # Combined assign_id callback: registers object identity
                # AND saves a stub TraceNode (covers read_csv-style hooks
                # that fire assign_id without calling record()).
                tracker.register_assign_id_callback(
                    _make_assign_id_stub_callback(tracker)
                )
                _LOG.info(
                    "RudriQ: assign_id callback wired (object identity + "
                    "stub TraceNode mirroring)."
                )

                # ALSO wire post-record mirroring (autolineage v0.6.0+).
                # This persists each AutoLineage record as a TraceNode +
                # parent->child DIRECT edges in RudriQ's DuckDB so audit
                # reports show full upstream chains. The post_record
                # callback ALSO registers identity for the record's child
                # via AL's _lid_to_obj (covers transforms like sort_values
                # that record() but skip assign_id for the result).
                if hasattr(tracker, "register_post_record_callback"):
                    tracker.register_post_record_callback(
                        _make_mirror_callback(tracker)
                    )
                    _LOG.info(
                        "RudriQ: post-record mirroring callback wired "
                        "(autolineage>=0.6)."
                    )
                else:
                    _LOG.info(
                        "RudriQ: AutoLineage<0.6 detected; mirroring disabled. "
                        "Audit reports will show 'external' placeholders for "
                        "data-side nodes. Upgrade autolineage to >=0.6 to "
                        "enable in-DuckDB mirroring."
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