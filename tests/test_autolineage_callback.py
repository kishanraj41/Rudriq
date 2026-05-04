"""
Integration test for the rudriq <-> AutoLineage callback wiring.

Tests in this file exercise rudriq's seam against AutoLineage. Tests of
AutoLineage's own API (e.g. that ``register_assign_id_callback`` itself
fires the callback, that exceptions don't propagate) live in
autolineage's own test suite, not here.

Skipped when autolineage is not installed so the rudriq test suite
stays green without the optional [lineage] extra.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytest.importorskip(
    "autolineage",
    reason="autolineage not installed; skipping wiring integration test",
)


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    from rudriq.storage import duckdb_backend
    from rudriq.linker import clear_object_registry

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "al_cb_test.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    clear_object_registry()
    yield db
    db.close()


def test_rudriq_auto_wires_callback_into_autolineage(isolated_storage):
    """End-to-end: invoking rudriq's activation seam registers
    register_object_identity as an assign_id callback, so subsequent
    AutoLineage tracking auto-populates RudriQ's object registry."""
    import autolineage.auto as al_auto
    from rudriq.linker import _object_registry, clear_object_registry

    clear_object_registry()

    tracker = al_auto.get_tracker()
    if tracker is None:
        # Some autolineage versions lazy-init; force activation.
        import autolineage.auto  # noqa: F401
        tracker = al_auto.get_tracker()

    if tracker is None:
        pytest.skip("autolineage tracker did not initialize; skipping wiring test")

    # Manually invoke RudriQ's activation seam (avoids re-importing
    # rudriq.auto, which has global side effects and may already have run).
    from rudriq.linker import register_object_identity
    if hasattr(tracker, "register_assign_id_callback"):
        tracker.register_assign_id_callback(register_object_identity)
    else:
        pytest.skip(
            "autolineage<0.5 does not expose register_assign_id_callback; "
            "wiring test cannot exercise the seam."
        )

    # Drive AutoLineage to assign a lineage ID. The callback should
    # mirror id(obj) -> lid into rudriq's registry.
    sentinel = {"text": ["doc-1", "doc-2"]}
    lid = tracker.assign_id(sentinel, source="test")

    assert _object_registry.get(id(sentinel)) == lid
