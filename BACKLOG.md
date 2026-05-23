# RudriQ Backlog

Things deferred from the sprint, tracked here so they don't get lost.

## Open

### Critical for June 1 design partner readiness

_(All Critical-priority items resolved as of Day 11. Remaining work is Important/Operational.)_

### Important but deferable

#### Evaluation engine — content-preview enrichment for evaluators
**Status:** Open (surfaced Day 13)
**Origin:** Day 13 validation against `examples/realistic_rag_pipeline.py`

The Day 13 eval framework + 2 evaluators (`retrieval_relevance`, `groundedness`) are in. Validation against the realistic pipeline produced SKIPPED for every LLM node: the metadata RudriQ persists for spans and AL records carries identity/usage/timing/links but **no text content**.

Exact gap inventory from the validation run:

* LLM nodes (`llm_chat`, `llm_embedding`) carry `gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.*`, and our `rudriq.lineage_parent / link_method / link_confidence / domain` — but NOT `gen_ai.prompt` or `gen_ai.completion`. openllmetry-openai gates prompt/completion capture behind opt-in env (e.g., `TRACELOOP_TRACE_CONTENT=true` / OTel's `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`). Worth wiring this into our `[llm]` extra docs and the realistic pipeline.
* Data nodes (`data_read`, `data_transform`) carry `autolineage.shape / columns / content_hash / source / duration_ms` — but NOT a content preview. AutoLineage's record model doesn't include sample rows. Likely fix: have rudriq's mirror callback capture a small preview (e.g., first N strings of the relevant column) at record time and persist as `autolineage.preview` in node metadata.

Both fixes are mechanical and unlock the existing evaluators end-to-end. Evaluators already produce SKIPPED (not ERROR) when these are missing — they're the right correctness behavior; the gap is upstream metadata enrichment.

#### Evaluation engine — three more evaluators
**Status:** Open (deferred from Day 13)

Day 13 shipped framework + 2 of the planned 5 evaluators. Remaining for Day 14: drift (response distribution change over time), consistency (same prompt → same answer across runs), coherence (response internal logical structure). Framework is designed so adding these is mechanical: implement a class with `metric` + `evaluate(graph) → list[EvalResult]`, register it in `_EVALUATOR_REGISTRY` in `rudriq/cli.py`.

#### Concurrency-safe input stash
**Status:** Resolved (Day 12 Phase C, v0.0.9.dev1)
**Origin:** Day 8 thread-local fallback

Day 8's `set_recent_input_fallback` was a thread-local "most recent input" — last-writer-wins per thread. Concurrent / deeply-nested LLM calls could mask each other.

**Resolution:** replaced with `stash_input_on_context` / `retrieve_input_from_context` in `rudriq.processors.linking`, built on `opentelemetry.context` (Python `contextvars`). Each thread / asyncio task / OTel context sees its own value. The wrapper attaches without detaching (on_end fires after Traceloop's outer span-context detach, so a wrapper-side detach clears the value too early); LRU bounds the by-span-id channel.

Discovered while fixing: openllmetry-openai's *embeddings* wrapper detaches the OTel context before `span.end` fires, so the context fallback returns None for embedding spans. The by-span-id channel was switched from pop to peek+bounded LRU to compensate, which also fixed a previously-undiagnosed bug where two registered `RudriQSpanProcessor` instances raced on the single-use channel and only one of them got linked edges.

6 concurrency tests in `tests/test_concurrency_input_stash.py` (threads + asyncio tasks).

#### Timezone discipline audit
**Status:** Resolved (Day 12 Phase B, v0.0.9.dev1)
**Origin:** Day 7 timezone bug

Two timezone bugs caught (Day 2 DuckDB roundtrip, Day 7 AutoLineage timestamp). Phase B added `ensure_utc(dt, source=...)`, `autolineage_timestamp_to_utc`, and `otel_nanos_to_utc` in `rudriq.core.schema` and routed every ingestion path through them (DuckDB save/load, AutoLineage records, OTel nanos). 12 tests in `tests/test_timezone_discipline.py`.

#### Lineage edge through Traceloop openai instrumentor (object_identity test)
**Status:** Open / informational
**Origin:** Day 7

`tests/test_traceloop_integration.py::test_traceloop_path_with_registered_object_creates_lineage_edge` self-skips when Traceloop's openai instrumentor's serialization path bypasses our auto_capture wrapper. Day 8's thread-local fallback closes most of this gap in practice (the realistic pipeline links 3/43 calls now, not 0/43), but the by-span_id channel remains brittle. Re-confirm against future openai/Traceloop SDK versions; pin a known-good combination in `[llm]` extras when stable.

### Operational

#### PDF export of audit reports
**Status:** Stub raises NotImplementedError
**Origin:** Day 5

The CLI accepts `--format pdf` and prints a friendly error pointing at pandoc. Real PDF rendering (via reportlab, weasyprint, or pandoc bridging) is deferred to v0.3 / when a design partner asks for it.

#### User-supplied audit templates
**Status:** Stub raises NotImplementedError
**Origin:** Day 5

`generate_audit_report(template="custom-internal")` raises NotImplementedError. Real templates would let a customer supply their own Jinja2 template that maps the trace graph onto their internal compliance format. Deferred to v0.3.

## Resolved

### Day 12 Phase A — PyPI publish of autolineage 0.6.1 ✅
**Resolved:** May 15, 2026 (Day 12 Phase A, v0.0.9)

Built from the `v0.6.1` tag via a detached worktree (paper-related uncommitted changes on `v0.4.1-cleanup` were left intact). `twine check` PASSED for both wheel and sdist; TestPyPI dress rehearsal skipped since the token issued was real-PyPI scoped and `twine check` had already validated README rendering. Live at https://pypi.org/project/autolineage/0.6.1/.

RudriQ's `[lineage]` and `[all]` extras updated from `autolineage @ git+...@v0.6.1` to `autolineage>=0.6.1,<0.7`. Customers now `pip install rudriq[lineage]` without needing git access. Smoke-checked: fresh install resolves 0.6.1 from PyPI, 137 tests pass, realistic pipeline regression still 23/43 linkage.

### Day 12 Phase C — Concurrency-safe input stash + peek-LRU by-span-id channel ✅
**Resolved:** May 9, 2026 (Day 12 Phase C, v0.0.9.dev1)

Replaced Day 8's thread-local fallback with `opentelemetry.context`-based stash (`stash_input_on_context` / `retrieve_input_from_context` / `detach_input_from_context`). Each thread / asyncio task / OTel context sees its own value via Python's `contextvars`.

**Wrapper does not detach.** OTel's `start_as_current_span` detaches its context layer BEFORE `span.end()` fires `on_end`. The auto_capture wrapper's `attach` is layered on top of Traceloop's span context, so a wrapper-side detach would unwind both. Without detach, subsequent calls stack new layers; LRU on the by-span-id channel bounds memory.

**By-span-id channel switched from pop to peek + bounded LRU (max 1000).** A second issue surfaced: openllmetry-openai's *embeddings* wrapper detaches the OTel context before `span.end` fires, so the context fallback is empty at on_end time for embedding spans (still works for chat spans). With pop semantics, two registered `RudriQSpanProcessor` instances raced — only the first to fire got the input. Peek lets all processors read the same value; LRU naturally evicts stale entries.

6 concurrency tests (`tests/test_concurrency_input_stash.py` — basic stash/retrieve, LIFO discipline, 4-thread isolation via barrier, 4-task asyncio isolation via gather). Realistic pipeline still produces 23/43 linkage (3 batch embeddings via object_identity, 20 chats via substring). 137 total tests passing.

### Day 12 Phase B — Timezone discipline audit ✅
**Resolved:** May 9, 2026 (Day 12 Phase B, v0.0.9.dev1)

Centralized timezone conversion in `rudriq.core.schema`: `ensure_utc(dt, source=...)`, `autolineage_timestamp_to_utc`, `otel_nanos_to_utc`, plus a `TimezoneViolationError`. Every ingestion path (DuckDB save/load, AutoLineage records, OTel nanos) now routes through these helpers with explicit source labels for debugging. 12 tests in `tests/test_timezone_discipline.py`.

### LRU cap on linker registries ✅
**Resolved:** May 9, 2026 (Day 11)
**Commit:** [hash from this push]

Both `_object_registry` and `_content_registry` are now LRU-bounded. Default 50,000 entries each, configurable via `RUDRIQ_LINKER_CACHE_SIZE` env var (clamped to a minimum of 100 to prevent pathological eviction; invalid values silently fall back to default).

**Implementation.** Switched both registries from `dict` to `collections.OrderedDict`. Added `_registry_lock` and `_content_registry_lock` (separate locks so substring scanning's content-side hold doesn't block rapid object-identity registrations). Eviction is FIFO via `popitem(last=False)`; access (lookup or re-register) refreshes position via `move_to_end`.

**Correctness invariants preserved.**
- First-write-wins on `_content_registry` (Day 10a) — re-registration of the same `node_id` refreshes LRU position but does NOT overwrite content. The Day 10a regression test still passes.
- All-element scan in `link_by_object_identity` (Day 8) — preserved. Non-zero-aligned slice lookups still work.
- `(None, "object_identity", 1.0)` on miss (Day 2 contract) — preserved.

**Cross-registry consistency.** Eviction is independent per registry. If `_content_registry` evicts node X but `_object_registry` still has `id(obj) → X`, the linker can still match X via object identity (the substring path just won't trigger for X). Acceptable degradation; cross-registry coordination would need a different concurrency model.

**Tests added (11 in `tests/test_linker_lru_cap.py`).** Default size, env var override / invalid / clamp. Object-registry eviction order. Refresh on re-register. Refresh on lookup. Content-registry eviction order. Refresh on substring lookup. Concurrent registration smoke test (4 threads, 200 registrations under cap=100, ends at exactly 100). Runtime cap shrinking via `_set_cache_size_for_tests`.

**Bug caught during test development.** First version of two tests used `register_object_identity(object(), ...)` without holding references to the created objects. CPython reuses ids for unreferenced objects, so 50 sequential `object()` calls in a loop reused the same id, making the registry record 50 LRU refreshes instead of 50 inserts (final size: 1, not 50). Fixed by holding references in a list. Worth flagging because it's a real surprise: `object()` in a loop with no aliasing is NOT a sequence of distinct objects from the registry's perspective.

**Realistic pipeline regression check.** 23/43 LLM call linkage preserved (pipeline uses ~191 registry entries, well below default 50K cap). Tests: 118 passed nemo / 120 passed .venv-full.

### Retrieval-aware linker (substring matching) ✅
**Resolved:** May 7-8, 2026 (Day 10a + 10b)
**Commits:** addf915 (algorithm + 12 unit tests + LinkMethod.SUBSTRING enum), Day 10b (notebook narrative + version bump)

Object-identity linking covered batch embeddings (input list passed directly from pandas) but missed chat completions (messages dicts contain prompt-formatted strings) and query embeddings (fresh strings).

**Solution:** substring-based matching with provenance metadata.

- New `link_by_substring` strategy in `rudriq.linker`, confidence 0.7 (strong) or 0.5 (marginal).
- Parallel `_content_registry` mapping `node_id` → list of strings extracted from tracked objects (handles `str`, `list[str]`, pandas `Series` of strings). First-write-wins to preserve bulk content against Day 8's per-element tolist propagation.
- Bidirectional containment check (upstream contained in input OR input contained in upstream) with `min_match_length=20`, `min_match_fraction=0.3`. Asymmetric guard: EITHER fraction must clear threshold (lets a short doc fully contained in a long prompt match cleanly).
- `LinkMethod` enum extended with `SUBSTRING = "substring"` (caught during integration validation: a missing enum value was raising `ValueError` in `SpanProcessor.on_end`'s `LinkMethod(method)` cast, swallowed silently by Day 8's defensive try/except — unit tests passed but no edges landed).
- Strategy ordering: `object_identity` (1.0) → `content_hash` (0.8) → `substring` (0.7/0.5) → `name_match` (0.5).

**Realistic pipeline impact:** 3/43 → 23/43 linked LLM calls (7x improvement).

| LLM call type | Count | Linkage rate | Strategy |
|---|---|---|---|
| Batch embeddings | 3 | 3/3 (100%) | Object identity |
| Chat completions | 20 | 20/20 (100%) | Substring (new) |
| Query embeddings | 20 | 0/20 (0%) | Unlinked by design — fresh strings, no upstream source |

12 unit tests in `tests/test_substring_linker.py`. Notebook `examples/realistic_rag_pipeline.ipynb` cell 15 updated with the principled framing. Confidence-by-design: query embeddings are unlinked because matching fresh strings would produce false positives that destroy audit-report trust value.

### Notebook variant of realistic pipeline ✅
**Resolved:** May 9, 2026 (Day 9)
`examples/realistic_rag_pipeline.ipynb` is the Jupyter version of the Day 8 script with narrative markdown structure for design partner outreach. Executes end-to-end in ~17s, produces 54 ops / 32 edges / 3 lineage links / 11-step longest chain. Honest framing about which LLM calls link today vs the v0.0.8+ retrieval-aware linker.

### Six bugs from realistic pipeline ✅
**Resolved:** May 7, 2026 (Day 8)
Linker first-element-only check, `Series.tolist` element registration, `assign_id` callback persistence, `post_record` identity registration, autolineage `get_or_assign` callback firing on reuse (autolineage v0.6.1), OpenLLMetry inner-wrapper span context (thread-local fallback). 7 regression tests in `tests/test_scale_fixes.py`.

### v0.0.6: Mirror AutoLineage records into RudriQ DuckDB ✅
**Resolved:** May 5, 2026 (Day 7)
**Mechanism:** AutoLineage v0.6.0 added `register_post_record_callback`. rudriq.auto wires a callback that mirrors each `TransformationRecord` as a `TraceNode` in DuckDB and creates `EdgeKind.DIRECT` edges from each `parent_id`. `RudriQSpanProcessor.__init__` wires its `run_id` into the mirroring callback so subsequent records land in the run that hosts the LLM spans.
**Result:** Audit reports for in-process pipelines now show full data→LLM ancestry without `external` placeholders.
**Caveat:** The `external/unmirrored` placeholder is still emitted when (a) autolineage<0.6 is installed, or (b) AutoLineage records fire before any `RudriQSpanProcessor` is constructed.

### Production Traceloop-managed flow verification ✅
**Resolved:** May 5, 2026 (Day 7)
**Mechanism:** `tests/test_traceloop_integration.py` exercises the production path: `Traceloop.init()` with a fake `TRACELOOP_API_KEY` so its SDK provider is created, `RudriQSpanProcessor` added to that provider, an openai call through OpenLLMetry's instrumentor with `httpx.MockTransport`.
**Bonus production hardening:** `RudriQSpanProcessor.on_end` wraps its body in a defensive `try/except` so a stale processor cannot poison sibling processors on the same TracerProvider.
