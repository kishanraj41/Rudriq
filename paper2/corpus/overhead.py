"""
Phase 4 . S8 overhead harness - the FIRST non-deterministic measurement.

Every prior measurement was deterministic (byte-identical corpus, same linker
outcome). Latency is not: it varies with scheduling, cache state, GC, load,
warmup. So the byte-identical discipline does NOT apply - applying it would
produce an authoritative-looking number that is actually noise. The rigor
here is DISTRIBUTIONAL:

  * Warmup discarded (disclosed), steady-state only.
  * Median + p95/p99 reported - never the mean alone. For an always-on audit
    tool the TAIL is what a buyer-under-load cares about.
  * The measurement's own NOISE FLOOR is measured (an empty op). If a
    strategy's latency is at/below the floor, that is reported honestly -
    "below our ability to measure" is a strong result, not a gap.
  * BOTH baselines: absolute per-strategy correlate() latency (the mechanism
    cost) AND end-to-end marginal cost the linker adds per call (the
    buyer-facing cost). They answer different questions.
  * Swept BY STRATEGY (surfacing content_hash's storage I/O) and BY REGISTRY
    SIZE across orders of magnitude (surfacing the scaling curve: flat O(1)
    or degrading O(n) - a degradation curve is a FINDING, not a failure).
  * Environment stated - latency numbers are meaningless without the machine
    that produced them.

The accuracy results said the linker is *correct*; the overhead results say
it is *affordable* - the empirical backing for "always-on" audit.
"""

from __future__ import annotations

import gc
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rudriq.core.schema import NodeKind, TraceNode, compute_content_hash
from rudriq.linker import (
    clear_object_registry,
    correlate,
    link_by_content_hash,
    link_by_object_identity,
    link_by_substring,
    register_object_identity,
)
from paper2.corpus.primitives import FROZEN_TS, new_isolated_storage


def _percentile(sorted_ns: list[int], p: float) -> float:
    if not sorted_ns:
        return 0.0
    k = (len(sorted_ns) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(sorted_ns[int(k)])
    return sorted_ns[f] * (c - k) + sorted_ns[c] * (k - f)


@dataclass
class Stats:
    n: int
    median_ns: float
    p95_ns: float
    p99_ns: float
    mean_ns: float
    stdev_ns: float
    min_ns: int

    @classmethod
    def from_samples(cls, samples: list[int]) -> "Stats":
        s = sorted(samples)
        return cls(
            n=len(s),
            median_ns=_percentile(s, 0.50),
            p95_ns=_percentile(s, 0.95),
            p99_ns=_percentile(s, 0.99),
            mean_ns=statistics.fmean(s),
            stdev_ns=statistics.pstdev(s) if len(s) > 1 else 0.0,
            min_ns=s[0],
        )

    def render(self) -> str:
        return (f"median {_ns(self.median_ns)} | p95 {_ns(self.p95_ns)} | "
                f"p99 {_ns(self.p99_ns)} | min {_ns(self.min_ns)} | "
                f"mean {_ns(self.mean_ns)} (+/-{_ns(self.stdev_ns)}, n={self.n})")


def _ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f}ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f}us"
    return f"{ns / 1_000_000:.3f}ms"


def measure(fn: Callable[[], object], iters: int, warmup: int) -> Stats:
    """Time ``fn`` per-call, ``iters`` times after ``warmup`` discarded calls.
    GC disabled during the loop so a GC pause is not mis-attributed to the
    operation."""
    for _ in range(warmup):
        fn()
    samples: list[int] = []
    gc_was = gc.isenabled()
    gc.disable()
    try:
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            fn()
            samples.append(time.perf_counter_ns() - t0)
    finally:
        if gc_was:
            gc.enable()
    return Stats.from_samples(samples)


# ---------------------------------------------------------------------------
# Registry population for the sweeps
# ---------------------------------------------------------------------------


def _populate_identity(size: int) -> list[str]:
    clear_object_registry()
    objs = [f"identity object number {i} with sufficient descriptive length" for i in range(size)]
    for i, o in enumerate(objs):
        register_object_identity(o, f"id-{i}", extract_content=False)
    return objs


def _populate_content(size: int) -> list[str]:
    clear_object_registry()
    docs = [f"content document {i}: a passage of meaningful length for substring scanning {i}"
            for i in range(size)]
    for i, d in enumerate(docs):
        register_object_identity(d, f"cn-{i}", extract_content=True)
    return docs


def _populate_storage(size: int, tmp: Path):
    storage = new_isolated_storage(tmp / f"oh-{size}.duckdb")
    payloads = [f"hash payload number {i} reconciled snapshot final {i}" for i in range(size)]
    for i, p in enumerate(payloads):
        node = TraceNode(node_id=f"hn-{i}", kind=NodeKind.DATA_READ, library="oh",
                         operation="seed", started_at=FROZEN_TS, ended_at=FROZEN_TS,
                         content_hash=compute_content_hash(p))
        storage.save_node(node, "oh-run")
    return payloads, storage


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

WARMUP = 200
SIZES = (10, 100, 1000, 10000)


def main() -> int:
    print("=" * 80)
    print("Phase 4 . S8 overhead harness (distributional; first non-deterministic measure)")
    print("=" * 80)

    # Environment - numbers are meaningless without it.
    print("\n--- Measurement environment (numbers are environment-specific; "
          "the SHAPE is the portable finding) ---")
    print(f"  platform   : {platform.platform()}")
    print(f"  python     : {platform.python_version()} ({platform.python_implementation()})")
    print(f"  processor  : {platform.processor() or 'n/a'}")
    print(f"  cpu_count  : {__import__('os').cpu_count()}")
    print(f"  perf_counter resolution: {time.get_clock_info('perf_counter').resolution * 1e9:.1f}ns")
    print("  load note  : single dev machine; absolute values are illustrative, "
          "strategy RATIOS and scaling SHAPE are the findings.")

    # Noise floor - the measurement instrument's own cost.
    floor = measure(lambda: None, iters=20000, warmup=WARMUP)
    print("\n--- Noise floor (timing an empty callable) ---")
    print(f"  {floor.render()}")
    print(f"  -> any strategy at/below ~{_ns(floor.median_ns)} is BELOW the measurement floor.")

    tmp = Path(__import__("tempfile").mkdtemp(prefix="rudriq_oh_"))

    # By strategy, swept by registry size.
    print("\n--- BY STRATEGY x BY REGISTRY SIZE (median; the scaling curve) ---")
    print(f"  {'size':>7} | {'object_identity':>18} | {'content_hash':>18} | {'substring':>18}")
    curve: dict[str, dict[int, Stats]] = {"object_identity": {}, "content_hash": {}, "substring": {}}
    for size in SIZES:
        # object_identity: hit the middle entry (O(1) dict lookup expected).
        objs = _populate_identity(size)
        target = objs[size // 2]
        curve["object_identity"][size] = measure(
            lambda: link_by_object_identity(target, {}), iters=5000, warmup=WARMUP)

        # content_hash: storage-backed lookup (I/O; indexed by hash).
        payloads, _storage = _populate_storage(size, tmp)
        ch_input = payloads[size // 2]
        curve["content_hash"][size] = measure(
            lambda: link_by_content_hash(ch_input, {}), iters=1000, warmup=WARMUP // 2)

        # substring: scans the whole content registry (O(n) expected).
        docs = _populate_content(size)
        sub_input = f"context: {docs[size // 2]} end"
        sub_iters = 1000 if size <= 1000 else 300
        curve["substring"][size] = measure(
            lambda: link_by_substring(sub_input, {}), iters=sub_iters, warmup=50)

        print(f"  {size:>7} | {_ns(curve['object_identity'][size].median_ns):>18} | "
              f"{_ns(curve['content_hash'][size].median_ns):>18} | "
              f"{_ns(curve['substring'][size].median_ns):>18}")

    # Full distributions at a representative mid size.
    mid = 1000
    print(f"\n--- Full distributions at registry size {mid} (tail matters for audit-under-load) ---")
    for strat in ("object_identity", "content_hash", "substring"):
        st = curve[strat][mid]
        floored = " [AT/BELOW NOISE FLOOR]" if st.median_ns <= floor.median_ns * 2 else ""
        print(f"  {strat:<16}: {st.render()}{floored}")

    # End-to-end marginal cost the linker adds per call (buyer-facing).
    print("\n--- End-to-end: marginal cost the linker ADDS per call (buyer-facing) ---")
    clear_object_registry()
    counter = {"i": 0}

    def _baseline():
        i = counter["i"]; counter["i"] += 1
        return f"payload {i} of representative length for an llm call input"

    def _instrumented():
        i = counter["i"]; counter["i"] += 1
        data = f"payload {i} of representative length for an llm call input"
        register_object_identity(data, f"e2e-{i}", extract_content=True)
        return correlate(data, {})

    base = measure(_baseline, iters=3000, warmup=WARMUP)
    counter["i"] = 0
    clear_object_registry()
    inst = measure(_instrumented, iters=3000, warmup=WARMUP)
    delta_median = inst.median_ns - base.median_ns
    print(f"  baseline (build input only)        : {base.render()}")
    print(f"  instrumented (register + correlate): {inst.render()}")
    print(f"  -> marginal linker cost per call (median delta): {_ns(delta_median)}")
    for llm_label, llm_ms in (("a real embeddings call ~10ms", 10), ("a real chat call ~500ms", 500)):
        frac = (delta_median / 1e6) / llm_ms * 100
        print(f"     vs {llm_label}: {frac:.4f}% overhead")

    # The content_hash caveat flag.
    print("\n--- content_hash storage-I/O caveat check ---")
    oi_mid = curve["object_identity"][mid].median_ns
    ch_mid = curve["content_hash"][mid].median_ns
    ratio = ch_mid / max(1.0, oi_mid)
    print(f"  content_hash / object_identity median ratio at size {mid}: {ratio:.1f}x")
    if ratio >= 10:
        print("  CAVEAT WARRANTED: content_hash carries a storage-I/O cost (DuckDB round-trip). "
              "Honest framing: in-process strategies dominate latency-sensitive paths; "
              "content_hash can be batched/deferred.")
    else:
        print("  No special caveat: content_hash is within an order of magnitude of in-process "
              "strategies.")

    # Scaling-shape findings.
    print("\n--- Scaling SHAPE (the portable finding) ---")
    for strat in ("object_identity", "content_hash", "substring"):
        small = curve[strat][SIZES[0]].median_ns
        large = curve[strat][SIZES[-1]].median_ns
        growth = large / max(1.0, small)
        shape = "FLAT (O(1))" if growth < 3 else (f"GROWS {growth:.0f}x over {SIZES[0]}->{SIZES[-1]}")
        print(f"  {strat:<16}: {_ns(small)} @ {SIZES[0]:>5}  ->  {_ns(large)} @ {SIZES[-1]:<5}   {shape}")

    print("\n" + "=" * 80)
    print("RESULT: distributions reported (not single numbers); noise floor measured; "
          "both baselines; swept by strategy and registry size; environment stated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
