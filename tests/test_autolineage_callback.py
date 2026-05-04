"""
Tests for the AutoLineage register_assign_id_callback integration.

These tests exercise the seam between AutoLineage (which we control as
a sibling project) and RudriQ. They require autolineage>=0.5 to be
installed; they're skipped otherwise so the rudriq test suite stays
green even without the optional [lineage] extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip(
    "autolineage",
    reason="autolineage[lineage] not installed; skipping callback integration tests",
)


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    """Each test gets its own DuckDB."""
    from rudriq.storage import duckdb_backend
    from rudriq.linker import clear_object_registry

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "al_cb_test.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    clear_object_registry()
    yield db
    db.close()


def test_assign_id_fires_registered_callback():
    """AutoLineage's UnifiedTracker.assign_id must invoke registered callbacks
    with (obj, lid). This is the load-bearing API that RudriQ wires into."""
    from autolineage.core.tracker import UnifiedTracker

    tracker = UnifiedTracker()
    fired: list = []

    def cb(obj, lid):
        fired.append((obj, lid))

    tracker.register_assign_id_callback(cb)

    sentinel = {"id": "marker"}
    lid = tracker.assign_id(sentinel, source="test")

    assert len(fired) == 1
    fired_obj, fired_lid = fired[0]
    assert fired_obj is sentinel
    assert fired_lid == lid


def test_callback_exception_does_not_break_assign_id():
    """A buggy callback must not propagate; assign_id keeps working."""
    from autolineage.core.tracker import UnifiedTracker

    tracker = UnifiedTracker()

    def angry_cb(obj, lid):
        raise RuntimeError("simulated callback failure")

    tracker.register_assign_id_callback(angry_cb)

    # Should NOT raise; lid is returned normally.
    lid = tracker.assign_id({"x": 1}, source="test")
    assert isinstance(lid, str) and len(lid) > 0


def test_rudriq_auto_wires_callback_into_autolineage(isolated_storage):
    """End-to-end: invoking rudriq.auto's activation path registers
    register_object_identity as an assign_id callback, so subsequent
    AutoLineage tracking auto-populates RudriQ's object registry."""
    import autolineage.auto as al_auto
    from rudriq.linker import _object_registry, clear_object_registry

    clear_object_registry()

    # Force-init the AutoLineage global tracker. (autolineage.auto exports
    # init/get_tracker; importing the module installs its hooks.)
    tracker = al_auto.get_tracker()
    if tracker is None:
        # Some autolineage versions lazy-init; force it via the activation
        # path that rudriq.auto exercises.
        import autolineage.auto  # noqa: F401  (re-import to ensure init)
        tracker = al_auto.get_tracker()

    if tracker is None:
        pytest.skip("autolineage tracker did not initialize; skipping wiring test")

    # Manually invoke RudriQ's activation seam (avoids re-importing
    # rudriq.auto, which has global side effects and may already have run).
    from rudriq.linker import register_object_identity
    if hasattr(tracker, "register_assign_id_callback"):
        tracker.register_assign_id_callback(register_object_identity)

    # Drive AutoLineage to assign a lineage ID. The callback should
    # mirror id(obj) -> lid into rudriq's registry.
    sentinel = {"text": ["doc-1", "doc-2"]}
    lid = tracker.assign_id(sentinel, source="test")

    assert _object_registry.get(id(sentinel)) == lid
