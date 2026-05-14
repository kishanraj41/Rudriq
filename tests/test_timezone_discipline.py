"""Tests for timezone discipline at every ingestion boundary.

The audit on May 13, 2026 (Day 12) reviewed every place RudriQ accepts
a timestamp from an external source. This test file pins the canonical
behavior at each boundary so regressions are caught immediately.

Two prior timezone bugs (Day 2 DuckDB roundtrip, Day 7 AutoLineage
naive interpretation) motivated this audit. These tests prevent a
third.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from rudriq.core.schema import (
    NodeKind,
    TraceNode,
    TimezoneViolationError,
    autolineage_timestamp_to_utc,
    ensure_utc,
    otel_nanos_to_utc,
)


# --- ensure_utc -----------------------------------------------------------


def test_ensure_utc_accepts_utc_aware():
    """UTC-aware datetimes pass through unchanged."""
    dt = datetime(2026, 5, 13, 18, 30, tzinfo=timezone.utc)
    result = ensure_utc(dt, source="test")
    assert result == dt
    assert result.tzinfo is timezone.utc


def test_ensure_utc_converts_other_tz_to_utc():
    """Non-UTC aware datetimes are converted to UTC."""
    plus5 = timezone(timedelta(hours=5))
    dt = datetime(2026, 5, 13, 18, 30, tzinfo=plus5)
    result = ensure_utc(dt, source="test")
    assert result.tzinfo is timezone.utc
    # 18:30 in UTC+5 == 13:30 UTC
    assert result.hour == 13
    assert result.minute == 30


def test_ensure_utc_rejects_naive_with_source_info():
    """Naive datetimes raise with a helpful, source-tagged message."""
    dt = datetime(2026, 5, 13, 18, 30)
    with pytest.raises(TimezoneViolationError) as exc_info:
        ensure_utc(dt, source="hypothetical-test-source")
    msg = str(exc_info.value)
    assert "hypothetical-test-source" in msg
    assert "Naive datetime" in msg


def test_ensure_utc_passes_none_through():
    """None is a valid value (optional datetime columns); pass through."""
    assert ensure_utc(None) is None
    assert ensure_utc(None, source="x") is None


def test_timezone_violation_is_a_value_error():
    """TimezoneViolationError is a ValueError subclass so callers can
    catch generic ValueError if they prefer."""
    assert issubclass(TimezoneViolationError, ValueError)


# --- autolineage_timestamp_to_utc -----------------------------------------


def test_autolineage_naive_round_trips_to_local():
    """AutoLineage's naive-local input converts to aware UTC.

    The canonical conversion uses .astimezone(), which interprets naive
    as local. The round-trip (UTC back to local) must recover the
    original wall-clock hour/minute. The Day 7 bug pattern
    (.replace(tzinfo=timezone.utc)) would not — it would claim the
    naive wall-clock value WAS already UTC, breaking the round-trip on
    non-UTC systems.
    """
    naive = datetime(2026, 5, 13, 18, 30, 0)
    result = autolineage_timestamp_to_utc(naive)
    assert result.tzinfo is not None
    # Round-trip back to local: wall clock matches the original.
    local_back = result.astimezone()
    assert local_back.hour == naive.hour
    assert local_back.minute == naive.minute


def test_autolineage_aware_normalizes_to_utc():
    """If AutoLineage ever starts emitting aware datetimes, normalize."""
    plus5 = timezone(timedelta(hours=5))
    dt = datetime(2026, 5, 13, 18, 30, tzinfo=plus5)
    result = autolineage_timestamp_to_utc(dt)
    assert result.tzinfo is timezone.utc
    assert result.hour == 13  # 18:30 +5 -> 13:30 UTC


def test_autolineage_naive_now_matches_aware_now():
    """Day 7 regression guard.

    On a non-UTC machine, the Day 7 bug (.replace(tzinfo=timezone.utc))
    would produce a UTC timestamp hours off from datetime.now(timezone.utc).
    With .astimezone(), naive-local interpreted as local converts to the
    same instant.

    On a UTC machine, both behaviors produce identical results — this
    test passes either way. Its value is on developer/CI machines in
    non-UTC zones where the Day 7 bug would fail loudly.
    """
    naive_now = datetime.now()
    aware_now = datetime.now(timezone.utc)
    converted = autolineage_timestamp_to_utc(naive_now)
    delta = abs((aware_now - converted).total_seconds())
    # Should be well under a second; the Day 7 bug would yield hours.
    assert delta < 60, (
        f"AutoLineage naive->UTC conversion off by {delta}s — likely the "
        f"Day 7 .replace(tzinfo=timezone.utc) regression. "
        f"aware_now={aware_now}, converted={converted}"
    )


# --- otel_nanos_to_utc ----------------------------------------------------


def test_otel_nanos_to_utc_basic():
    """OTel nanos -> aware UTC datetime with correct wall-clock value."""
    dt = datetime(2026, 5, 13, 18, 30, tzinfo=timezone.utc)
    nanos = int(dt.timestamp() * 1_000_000_000)
    result = otel_nanos_to_utc(nanos)
    assert result.tzinfo is timezone.utc
    assert result.year == 2026
    assert result.month == 5
    assert result.day == 13
    assert result.hour == 18
    assert result.minute == 30


def test_otel_nanos_to_utc_handles_zero():
    """Zero nanos = UNIX epoch in UTC."""
    result = otel_nanos_to_utc(0)
    assert result.tzinfo is timezone.utc
    assert result.year == 1970
    assert result.month == 1
    assert result.day == 1


# --- Integration: storage refuses naive at the write boundary -------------


def test_storage_rejects_naive_started_at(tmp_path: Path):
    """End-to-end: a naive datetime persisted via save_node is refused.

    The strict ensure_utc helper at the storage write boundary turns
    what was previously a silent .replace(tzinfo=timezone.utc) coercion
    into a loud TimezoneViolationError. This is the regression guard
    for the latent corruption the audit identified at
    docs/timezone_audit_notes.md entry #6.
    """
    from rudriq.storage.duckdb_backend import DuckDBStorage

    db = DuckDBStorage(db_path=tmp_path / "tz_test.duckdb")
    try:
        node = TraceNode(
            node_id="naive-test",
            kind=NodeKind.DATA_TRANSFORM,
            library="test",
            operation="test_op",
            started_at=datetime(2026, 5, 13, 18, 30, 0),  # NAIVE
            ended_at=None,
            metadata={},
        )
        db.ensure_run("tz-test-run")
        with pytest.raises(TimezoneViolationError):
            db.save_node(node, run_id="tz-test-run")
    finally:
        db.close()


def test_storage_accepts_aware_started_at(tmp_path: Path):
    """Positive control for the strict storage guard: aware datetimes
    persist normally."""
    from rudriq.storage.duckdb_backend import DuckDBStorage

    db = DuckDBStorage(db_path=tmp_path / "tz_test.duckdb")
    try:
        now = datetime.now(timezone.utc)
        node = TraceNode(
            node_id="aware-test",
            kind=NodeKind.DATA_TRANSFORM,
            library="test",
            operation="test_op",
            started_at=now,
            ended_at=now,
            metadata={},
        )
        db.ensure_run("tz-test-run")
        db.save_node(node, run_id="tz-test-run")  # must not raise
    finally:
        db.close()
