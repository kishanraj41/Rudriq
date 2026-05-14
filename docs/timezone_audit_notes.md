# Timezone audit — May 13, 2026 (Day 12, Phase B)

## Motivation

Two timezone bugs already shipped and were caught:

- **Day 2 — DuckDB roundtrip.** Storage layer silently stripped UTC tzinfo on
  read. Fixed by `TIMESTAMPTZ` schema + `_from_db()` normalizer.
- **Day 7 — AutoLineage naive interpretation.** A naive local-time string from
  AutoLineage's `TransformationRecord.timestamp` was labeled as UTC via
  `dt.replace(tzinfo=timezone.utc)`, producing 5-hour duration errors on
  non-UTC systems. Fixed in [rudriq/auto.py:82](rudriq/auto.py#L82) by parsing
  the ISO string and calling `parsed.astimezone(timezone.utc)`, which
  correctly interprets naive datetimes as local time.

This audit makes every ingestion point explicit and enforces correct handling
at the boundary so a third bug of this class cannot land.

## Ingestion inventory

Source categories:
- **Internal**: `datetime.now(timezone.utc)` — RudriQ controls construction, low risk.
- **OTel nanos**: integer nanoseconds since UNIX epoch — well-defined, low risk if converted correctly.
- **AutoLineage naive**: naive local time emitted by `datetime.now().isoformat()` — **already burned us once**.
- **DuckDB roundtrip**: `TIMESTAMPTZ` schema returns system-local-tz datetimes — already normalized by `_from_db`.
- **External / user metadata**: any datetime in a user-supplied `metadata` dict — not currently typed; validate at boundary.

| # | Location | Source category | What it accepts | Current handling | Risk |
|---|---|---|---|---|---|
| 1 | [rudriq/adapters/otel_ingest.py:67-69](rudriq/adapters/otel_ingest.py#L67) `_ns_to_datetime` | OTel nanos | `int` nanoseconds | `datetime.fromtimestamp(ns/1e9, tz=timezone.utc)` | Low — already aware UTC. Replace with `otel_nanos_to_utc`. |
| 2 | [rudriq/adapters/otel_ingest.py:202-203](rudriq/adapters/otel_ingest.py#L202) `otel_span_to_node` | OTel nanos | `start_time_ns`, `end_time_ns` from `ReadableSpan` | Delegates to `_ns_to_datetime` | Low. |
| 3 | [rudriq/auto.py:82-104](rudriq/auto.py#L82) `_autolineage_record_to_started_at` | AutoLineage naive | ISO-8601 string, naive local | `datetime.fromisoformat(ts).astimezone(timezone.utc)` (Day 7 fix) | **Critical surface** — already burned. Replace with `autolineage_timestamp_to_utc` for explicit intent at the boundary. |
| 4 | [rudriq/auto.py:104](rudriq/auto.py#L104) `_autolineage_record_to_started_at` fallback | Internal | none | `datetime.now(timezone.utc)` | Low. |
| 5 | [rudriq/auto.py:275-276](rudriq/auto.py#L275) `_on_assign_id` stub | Internal | none | `datetime.now(timezone.utc)` for both started_at/ended_at | Low. |
| 6 | [rudriq/storage/duckdb_backend.py:84-95](rudriq/storage/duckdb_backend.py#L84) `_ensure_utc` (write path) | Mixed | aware or naive datetime | **Naive → `dt.replace(tzinfo=timezone.utc)`** (the Day 7 anti-pattern) with warning. Aware non-UTC → returned as-is (NOT converted). | **Medium** — silent fallback masks naive arrivals. Replace with strict `ensure_utc` that raises on naive. Also covers non-UTC awareness. |
| 7 | [rudriq/storage/duckdb_backend.py:98-110](rudriq/storage/duckdb_backend.py#L98) `_from_db` (read path) | DuckDB roundtrip | datetime returned from `TIMESTAMPTZ` column | Naive → `.replace(tzinfo=timezone.utc)`; aware → `.astimezone(timezone.utc)` | Low (schema is `TIMESTAMPTZ` so naive should never appear). Keep; this is the read side. |
| 8 | [rudriq/storage/duckdb_backend.py:157](rudriq/storage/duckdb_backend.py#L157) `ensure_run` | Internal | none | `_ensure_utc(datetime.now(timezone.utc))` | Low (redundant wrapper around aware now). |
| 9 | [rudriq/storage/duckdb_backend.py:180](rudriq/storage/duckdb_backend.py#L180) `replace_run` | TraceGraph.created_at | `datetime` from caller | `_ensure_utc(graph.created_at)` | Inherits #6 risk; tightened by strict helper. |
| 10 | [rudriq/storage/duckdb_backend.py:202-203](rudriq/storage/duckdb_backend.py#L202) `replace_run` (nodes) | TraceNode.{started,ended}_at | `datetime` | `_ensure_utc(...)` | Inherits #6 risk. |
| 11 | [rudriq/storage/duckdb_backend.py:251](rudriq/storage/duckdb_backend.py#L251) `save_node` (run insert) | Internal | none | `datetime.now(timezone.utc)` — unwrapped | Low. |
| 12 | [rudriq/storage/duckdb_backend.py:261-262](rudriq/storage/duckdb_backend.py#L261) `save_node` | TraceNode.{started,ended}_at | `datetime` | `_ensure_utc(...)` | Inherits #6 risk. |
| 13 | [rudriq/processors/linking.py:216-222](rudriq/processors/linking.py#L216) `RudriQSpanProcessor.on_end` | OTel | `span.start_time`, `span.end_time` (nanos) | Passes through to `otel_span_to_node` (entry #2) | Low (delegated). |
| 14 | [rudriq/export/audit.py:299](rudriq/export/audit.py#L299) `_generated_at` | Internal | none | `datetime.now(timezone.utc).isoformat()` | Low. |
| 15 | [rudriq/export/audit.py:486](rudriq/export/audit.py#L486) `_empty_graph` | Internal | none | `datetime.now(timezone.utc)` | Low. |

## AutoLineage emission sites (the source of #3)

Confirmed AutoLineage emits naive local times in:

- [autolineage/core/__init__.py:35](C:/Users/kisha/OneDrive/Documents/AI/autolineage/autolineage/core/__init__.py#L35) — `TransformationRecord.timestamp: str = field(default_factory=lambda: datetime.now().isoformat())`
- [autolineage/core/tracker.py:135](C:/Users/kisha/OneDrive/Documents/AI/autolineage/autolineage/core/tracker.py#L135) — `'created_at': datetime.now().isoformat()`
- [autolineage/core/analyzer.py:53](C:/Users/kisha/OneDrive/Documents/AI/autolineage/autolineage/core/analyzer.py#L53) — same pattern

These are upstream and not under this audit's scope; RudriQ owns the boundary
conversion (entry #3 above).

## Decisions

1. **Replace `duckdb_backend._ensure_utc` with strict `schema.ensure_utc`** that
   raises `TimezoneViolationError` on naive input rather than silently coercing.
   The silent path was Day 7's actual bug pattern hiding in the storage layer
   as a "safety net" that actually corrupts timestamps. Removing the silent
   path turns a latent corruption into a loud test failure.
2. **Keep `_from_db` as-is in duckdb_backend** — it's the read side, and naive
   datetimes there indicate a schema regression (TIMESTAMP instead of
   TIMESTAMPTZ), which we want to defensively handle, not crash on.
3. **Add `autolineage_timestamp_to_utc` helper** even though `auto.py` already
   uses the correct pattern — having the canonical helper named after the
   source makes future contributors choose the right thing without
   having to read the docstring rationale.

## Out of scope

- User-supplied `metadata` dicts: TraceNode/Edge metadata is `dict[str, Any]`
  and may contain user-provided datetimes. We do not currently introspect
  metadata. Adding type-safety here is a v0.1 task.
- OpenAI/Anthropic response timestamps: handled inside vendor SDKs; we ingest
  the OTel span timestamps (entries #1–2), not the response objects directly.
