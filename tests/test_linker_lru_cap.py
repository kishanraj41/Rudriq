"""Tests for LRU-bounded linker registries.

Day 11 — closes the BACKLOG critical-priority item from Day 8 about
unbounded registry growth. Verifies that registries enforce a cap,
evict in LRU order, refresh on access, and degrade gracefully when
the cap is reached during an active pipeline.
"""

from __future__ import annotations

import threading

import pytest

from rudriq.linker import (
    DEFAULT_CACHE_SIZE,
    _content_registry,
    _get_cache_size,
    _object_registry,
    _set_cache_size_for_tests,
    clear_object_registry,
    link_by_object_identity,
    link_by_substring,
    register_object_identity,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Each test starts clean."""
    _set_cache_size_for_tests(DEFAULT_CACHE_SIZE)
    clear_object_registry()
    yield
    _set_cache_size_for_tests(DEFAULT_CACHE_SIZE)
    clear_object_registry()


def test_default_cache_size_is_reasonable():
    """50K is large enough for real pipelines, small enough to bound memory."""
    assert DEFAULT_CACHE_SIZE == 50_000


def test_env_var_overrides_cache_size(monkeypatch):
    """RUDRIQ_LINKER_CACHE_SIZE should override the default."""
    monkeypatch.setenv("RUDRIQ_LINKER_CACHE_SIZE", "1000")
    assert _get_cache_size() == 1000


def test_env_var_invalid_falls_back_to_default(monkeypatch):
    """Garbage values should not crash; fall back to default."""
    monkeypatch.setenv("RUDRIQ_LINKER_CACHE_SIZE", "not-a-number")
    assert _get_cache_size() == DEFAULT_CACHE_SIZE


def test_env_var_too_small_clamped_to_minimum(monkeypatch):
    """Pathologically small values should be clamped, not used as-is."""
    monkeypatch.setenv("RUDRIQ_LINKER_CACHE_SIZE", "5")
    assert _get_cache_size() == 100


def test_object_registry_evicts_oldest_when_over_cap():
    """When the cap is reached, oldest entries should be evicted FIFO."""
    _set_cache_size_for_tests(3)

    objs = [object() for _ in range(5)]
    for i, obj in enumerate(objs):
        register_object_identity(obj, f"node-{i}", extract_content=False)

    # Cap is 3, registered 5. Oldest two (node-0, node-1) should be gone.
    assert len(_object_registry) == 3
    values = list(_object_registry.values())
    assert "node-0" not in values
    assert "node-1" not in values
    assert "node-2" in values
    assert "node-3" in values
    assert "node-4" in values


def test_object_registry_refresh_on_re_register():
    """Re-registering an existing key should refresh its LRU position."""
    _set_cache_size_for_tests(3)

    obj_a = object()
    obj_b = object()
    obj_c = object()
    obj_d = object()

    register_object_identity(obj_a, "node-A", extract_content=False)
    register_object_identity(obj_b, "node-B", extract_content=False)
    register_object_identity(obj_c, "node-C", extract_content=False)

    # Re-register A — should move to end, making B oldest
    register_object_identity(obj_a, "node-A", extract_content=False)

    # Now register D — should evict B (oldest), not A
    register_object_identity(obj_d, "node-D", extract_content=False)

    values = list(_object_registry.values())
    assert "node-A" in values
    assert "node-C" in values
    assert "node-D" in values
    assert "node-B" not in values


def test_lookup_refreshes_lru_position():
    """Successful linker lookups should mark the matched entry as recently used."""
    _set_cache_size_for_tests(3)

    obj_a = object()
    obj_b = object()
    obj_c = object()
    obj_d = object()

    register_object_identity(obj_a, "node-A", extract_content=False)
    register_object_identity(obj_b, "node-B", extract_content=False)
    register_object_identity(obj_c, "node-C", extract_content=False)

    # Look up A — should refresh it
    parent_id, _, _ = link_by_object_identity(obj_a, {})
    assert parent_id == "node-A"

    # Now register D — should evict B (oldest after A's refresh)
    register_object_identity(obj_d, "node-D", extract_content=False)

    values = list(_object_registry.values())
    assert "node-A" in values
    assert "node-B" not in values
    assert "node-C" in values
    assert "node-D" in values


def test_content_registry_evicts_oldest_when_over_cap():
    """Content registry should also enforce LRU eviction."""
    _set_cache_size_for_tests(2)

    for i in range(4):
        text = f"document number {i} with substantial content for substring matching"
        register_object_identity(text, f"node-{i}")

    assert len(_content_registry) == 2
    keys = list(_content_registry.keys())
    assert "node-0" not in keys
    assert "node-1" not in keys
    assert "node-2" in keys
    assert "node-3" in keys


def test_substring_lookup_refreshes_content_lru():
    """A substring match should refresh the matched content entry."""
    _set_cache_size_for_tests(2)

    text_a = "alpha document with substantial unique content for matching alpha"
    text_b = "beta document with substantial unique content for matching beta"
    text_c = "gamma document with substantial unique content for matching gamma"

    register_object_identity(text_a, "node-A")
    register_object_identity(text_b, "node-B")

    # Match against A — should refresh
    llm_input = f"Reference: {text_a} — analyze."
    parent_id, _, _ = link_by_substring(llm_input, {})
    assert parent_id == "node-A"

    # Register C — should evict B (oldest after A's refresh)
    register_object_identity(text_c, "node-C")

    keys = list(_content_registry.keys())
    assert "node-A" in keys
    assert "node-B" not in keys
    assert "node-C" in keys


def test_concurrent_registration_does_not_corrupt():
    """Concurrent register calls should not produce inconsistent state.

    Smoke test: Python's GIL plus our lock should prevent corruption,
    but worth verifying we don't get a key error or oversized cache.

    Each thread holds references to its own objects to prevent CPython
    id reuse: an unreferenced object()'s id can be reused immediately
    by the next object() call, which would make our registrations look
    like LRU refreshes instead of new inserts.
    """
    _set_cache_size_for_tests(100)

    held: list = []  # global list to keep objects alive
    held_lock = threading.Lock()

    def worker(start_idx):
        local_objs = [object() for _ in range(50)]
        with held_lock:
            held.extend(local_objs)
        for i, obj in enumerate(local_objs):
            register_object_identity(obj, f"node-{start_idx + i}", extract_content=False)

    threads = [threading.Thread(target=worker, args=(i * 50,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 200 registrations across 4 threads, cap 100. Should have exactly 100.
    assert len(_object_registry) == 100


def test_set_cache_size_for_tests_trims_existing():
    """Setting a smaller cache size should immediately trim oversized registries.

    Holds references to created objects so CPython doesn't reuse ids
    (an unreferenced object()'s id is freed immediately and the next
    object() can take it, which would make our 50 registrations
    collapse into 1 LRU-refreshed slot)."""
    _set_cache_size_for_tests(100)

    objs = [object() for _ in range(50)]
    for i, obj in enumerate(objs):
        register_object_identity(obj, f"node-{i}", extract_content=False)

    assert len(_object_registry) == 50

    # Shrink the cap below the current size
    _set_cache_size_for_tests(10)

    assert len(_object_registry) == 10
