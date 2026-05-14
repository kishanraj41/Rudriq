"""Tests for concurrency-safe LLM input capture.

Day 12 Phase C — replaces the Day 8 thread-local fallback with OTel
context-keyed storage. The thread-local fallback worked single-threaded
but degraded under concurrency: one thread's input could mask another's,
and nested calls would overwrite each other before on_end fired.

These tests verify the new mechanism works correctly across threads
and asyncio tasks. The mechanism uses ``opentelemetry.context``, which
is built on Python's ``contextvars`` — propagation across threads and
asyncio tasks is what makes per-context isolation work.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from rudriq.processors.linking import (
    detach_input_from_context,
    retrieve_input_from_context,
    stash_input_on_context,
)


def test_context_stash_basic():
    """Stash and retrieve in the same context."""
    token = stash_input_on_context("test-input-value")
    try:
        assert retrieve_input_from_context() == "test-input-value"
    finally:
        detach_input_from_context(token)


def test_context_stash_cleared_after_detach():
    """After detach, the context reverts to its pre-stash state.

    We don't assert None here: when other code paths (auto_capture
    wrappers, prior tests) have attached without detaching, the
    pre-stash state may itself carry a value. The contract is
    'detach reverts', not 'detach clears to None'.
    """
    before = retrieve_input_from_context()
    token = stash_input_on_context("temporary")
    assert retrieve_input_from_context() == "temporary"
    detach_input_from_context(token)
    assert retrieve_input_from_context() == before


def test_context_stash_overwrites_within_same_context():
    """Later stash in the same context replaces earlier (LIFO behavior).

    OTel's ``attach`` stacks contexts; ``detach(token)`` pops back to
    the prior state. Verifies the stack discipline matches what we
    expect for nested LLM calls."""
    token1 = stash_input_on_context("first")
    try:
        assert retrieve_input_from_context() == "first"
        token2 = stash_input_on_context("second")
        try:
            assert retrieve_input_from_context() == "second"
        finally:
            detach_input_from_context(token2)
        # After detaching the inner token, we see "first" again.
        assert retrieve_input_from_context() == "first"
    finally:
        detach_input_from_context(token1)


def test_context_stash_isolated_across_threads():
    """Each thread sees its OWN stashed value, never another thread's.

    This is the headline concurrency property. The Day 8 thread-local
    had the same property in theory (it WAS thread-local) but with
    last-writer-wins semantics within a thread; nested calls or
    sequential calls without intervening on_end would mask each other.
    Context-keyed storage doesn't have that risk because each
    attach/detach is its own scope.

    Test design: 4 threads each stash their own value, then synchronize
    on a barrier (so all are stashed before any retrieves), then each
    retrieves and reports. With per-thread isolation we expect each
    thread to see ITS OWN value."""
    results: dict[int, object] = {}
    errors: dict[int, Exception] = {}
    barrier = threading.Barrier(4)

    def worker(thread_id: int) -> None:
        try:
            input_value = f"input-from-thread-{thread_id}"
            token = stash_input_on_context(input_value)
            try:
                # Wait for all threads to stash before any retrieves.
                # Maximizes the chance of cross-thread contamination
                # if the implementation is broken.
                barrier.wait(timeout=5.0)
                results[thread_id] = retrieve_input_from_context()
            finally:
                detach_input_from_context(token)
        except Exception as exc:  # noqa: BLE001
            errors[thread_id] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert errors == {}, f"Threads failed: {errors}"
    for thread_id in range(4):
        assert results[thread_id] == f"input-from-thread-{thread_id}", (
            f"Thread {thread_id} saw {results[thread_id]!r} — "
            f"cross-thread contamination!"
        )


def test_context_stash_isolated_across_asyncio_tasks():
    """Each asyncio task sees its OWN stashed value.

    This is the modern equivalent of the threading test. Many real
    RAG production loops are async — e.g., FastAPI request handlers,
    LangChain async chains. Without per-task isolation, two concurrent
    requests could see each other's inputs at on_end time, producing
    cross-request audit-trail contamination.

    Each task stashes its own value, yields via asyncio.sleep so the
    scheduler interleaves them, then retrieves. With proper contextvars
    propagation across asyncio.Task creation, each task sees only its
    own value."""

    async def task(task_id: int):
        input_value = f"input-from-task-{task_id}"
        token = stash_input_on_context(input_value)
        try:
            # Yield to let other tasks stash before we retrieve.
            await asyncio.sleep(0.01)
            return task_id, retrieve_input_from_context()
        finally:
            detach_input_from_context(token)

    async def run_all():
        return await asyncio.gather(*(task(i) for i in range(4)))

    results = asyncio.run(run_all())
    for task_id, retrieved in results:
        assert retrieved == f"input-from-task-{task_id}", (
            f"Task {task_id} saw {retrieved!r} — cross-task contamination!"
        )
