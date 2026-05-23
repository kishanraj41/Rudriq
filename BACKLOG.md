# RudriQ Backlog

Things deferred from the sprint, tracked here so they don't get lost.

## Open

### Critical for June 1 design partner readiness

_(All Critical-priority items resolved as of Day 11. Remaining work is Important/Operational.)_

### Important but deferable

#### Evaluation engine — consistency grouping when retrieval lives inside the user message
**Status:** Partial fix landed Day 17 Thread B; residual is a property of prompt-assembly choice

Day 17 Thread B added role-separated capture: `rudriq.user_message_preview` is the user-role text only, distinct from any system message in `rudriq.prompt_preview`. ConsistencyEvaluator prefers the user message for grouping. Where the system message carries retrieval context, the fix fully resolves the artifact.

Residual: pipelines that put retrieval context *inside* the user-role message (e.g., `Context:...Question: <q>` as one user turn — the realistic demo's pattern) still see shared context dominate the embedding. Confirmed on `examples/realistic_rag_pipeline.py`: consistency still groups all 20 distinct queries together because their user-role content is structurally similar.

This is a prompt-assembly choice, not a metric bug. The fix that's defensible without becoming brittle: detect a `Question:` / `Query:` / `User:` delimiter inside the user message and embed only what follows. Fragile across pipelines; left for a future session that has a representative cross-pipeline corpus to validate against. The user-role separation alone is a real correctness win for RAG patterns that follow the OpenAI/Anthropic recommended "retrieval-in-system" convention.

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

### Day 16 — Eval results in the audit report (`--include-evals`) ✅
**Resolved:** May 23, 2026 (Day 16, v0.1.0.dev3)

Eval scores now appear in the artifact a compliance officer reads, not just in CLI output. `rudriq audit --run-id X --include-evals` embeds evaluation results into both JSON and Markdown audits.

**Schema bumped `rudriq.audit/1.0` → `1.1`** (additive). Two new top-level keys: `evaluations` (deterministically-ordered list of `EvalResult.to_dict()`) and `evaluation_summary` (per-metric traffic-light dict). Both are `null` when `--include-evals` is not passed, so the version string honestly describes the writer's capability. 1.0 consumers ignore the new keys without breakage.

**Determinism preserved** (Day 6 discipline). Eval list sorted by `(metric, node_id-or-empty-string)`; outer JSON uses `sort_keys=True`. Verified byte-identical across two consecutive eval-augmented exports of the realistic-pipeline run with real fastembed (119 782 bytes, identical apart from the documented `generated_at`).

**Traffic-light bands** (configurable via module constants):

* `green`  — mean score ≥ 0.7
* `yellow` — mean score ≥ 0.4
* `red`    — mean score < 0.4
* `gray`   — no scoreable OK results

Markdown renders the summary as a table (emoji circles — a deliberate style exception for a human-read artifact) plus a `### Notable findings` list of non-OK or sub-green results, capped at 20 entries with a "see JSON for the full list" footer.

**CLI:** `--include-evals`, `--eval-metrics`, `--baseline-run-id` added to the audit subcommand. The default metric set is the four single-trace content-aware evaluators (drift omitted unless explicitly requested + baseline provided).

8 new tests in `tests/test_audit_with_evals.py`, including the load-bearing byte-determinism guard (with `_now_utc_iso` monkeypatched), the traffic-light banding unit test, list-ordering stability, and a notable-findings Markdown surface test. 182 total tests passing.

Live audit on the realistic pipeline run: 🟢 retrieval_relevance 0.94 (22/23), 🟢 groundedness 1.00 (20/23), 🟢 coherence 0.76 (20/23), 🟢 consistency 1.00 (1/1, reflects the known grouping artifact). The SKIPs land in `### Notable findings` with their exact `node_id`s, which is what an auditor needs to drill in.

### Day 15 — Three more evaluators: coherence, consistency, drift ✅
**Resolved:** May 23, 2026 (Day 15, v0.1.0.dev2 unpushed)

Completes the 5-evaluator set planned for v0.1.0.

**Coherence (`rudriq.evaluate.coherence`)** — within-node holistic response-vs-context cosine similarity. Distinct from groundedness: groundedness asks "are individual claims supported?", coherence asks "does the response USE the context?". The two disagree productively — high groundedness + low coherence is the retrieval-decorative anti-pattern (model answered from parametric knowledge; claims happen to pattern-match retrieval). Single-call metric, no extra infrastructure. Shipped Phase 1 commit `39ae83b` (pushed).

**Consistency (`rudriq.evaluate.consistency`)** — within-run. Greedy-clusters LLM calls by prompt similarity (default threshold 0.85), scores mean pairwise response similarity within each group of size ≥ 2. Singletons SKIP. Honest about its applicability: runs of all-distinct prompts SKIP wholesale; fires on agentic loops, retries, batch-similar workloads.

**Drift (`rudriq.evaluate.drift`)** — cross-run, requires `--baseline-run-id`. Two signals:
* `drift_structural` — Counter of `library.operation` strings, normalized L1 distance between baseline and current distributions.
* `drift_response` — aligns LLM calls across runs by prompt-cosine ≥ 0.85, measures mean response drift on the aligned pairs. Unaligned current-run calls are surfaced as "new behavior."

Drift is the only evaluator that takes a constructor argument (the baseline graph), so it's special-cased in `_cmd_evaluate` rather than living in `_EVALUATOR_REGISTRY`. The CLI loads the baseline run and injects it; absence falls through to a SKIPPED result with a "drift is cross-run by nature" message.

**Validation against `examples/realistic_rag_pipeline.py` (two runs, capture on):**

| metric              | results | OK | SKIP | mean score |
|---------------------|--------:|---:|-----:|-----------:|
| retrieval_relevance |   23    | 18 | 5    | 0.94       |
| groundedness        |   23    | 18 | 5    | 1.00       |
| coherence           |   23    | 18 | 5    | 0.76       |
| consistency         |    1    |  1 | 0    | 1.00       |
| drift_structural    |    1    |  1 | 0    | 1.00       |
| drift_response      |    1    |  1 | 0    | 1.00       |

Consistency grouped all 20 chat calls because their prompt previews are dominated by shared retrieved-doc content (the prompts embed roughly the same context); responses are byte-identical from the mock so similarity = 1.0. Drift correctly reports ~0 for two identical pipeline runs — sanity check pass. The 5 SKIPs across the content-aware evaluators are consistent across all three (same nodes), suggesting a structural property of the pipeline worth a brief Day 16 look (probably the embedding spans without paired completion_preview).

16 new tests total across the three evaluators; 174 passing. v0.1.0 final once `--include-evals` lands in audit reports.

### Day 14 — Opt-in content-preview capture for evaluation ✅
**Resolved:** May 23, 2026 (Day 14, v0.1.0.dev1)

Day 13's eval framework + 2 evaluators were SKIPPING on every node because RudriQ persisted operation metadata but no text content. Day 14 unblocks them: opt-in, privacy-conscious, default-off content capture on both LLM and data sides.

**Config (`rudriq/core/config.py`):** `RUDRIQ_CAPTURE_CONTENT` (default off — operator must explicitly enable), `RUDRIQ_PREVIEW_CHARS` (default 500, clamped 50–10000). Read on every call so capture can be toggled without process restart and tests can `monkeypatch.setenv` without resetting cached state.

**LLM-side (`rudriq/auto.py` + `rudriq/adapters/otel_ingest.py`):**
- `_activate_traceloop` propagates `TRACELOOP_TRACE_CONTENT=true` and `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true` to openllmetry — but only when RudriQ's flag is on.
- otel ingest extracts content from openllmetry 0.60's new semconv (`gen_ai.input.messages`, `gen_ai.output.messages` — JSON arrays of `{role, parts:[{type:text, content}]}`) with a fallback to the pre-0.60 indexed shape (`gen_ai.prompt.N.content`). Stored as truncated `rudriq.prompt_preview` / `rudriq.completion_preview` in node metadata.

**Data-side (mirror + assign_id callbacks):**
- Reuses the substring linker's `_extract_strings` (Day 10) for str / list / pandas Series of strings.
- DataFrame fallback samples the first 5 rows of a string-typed column. Column choice via `obj.dtypes` (NOT `obj[col]`) — crucial because our patched `DataFrame.__getitem__` would otherwise propagate the iterated column's lid into `_content_registry` first-write-wins, locking the registry to whichever column scans first (typically `doc_id`) and starving the substring linker of actual text. Preferred column names: text, content, document, body, message, response, prompt; fall back to first string column.
- Mirror callback reordered to look up `obj` early (single strong ref carried through preview extraction and register_object_identity) instead of looking it up twice.

**Evaluators** now read the new preview keys in addition to legacy ones.

**Privacy docs:** README "Data handling and content capture" section. Default-off posture, local DuckDB only, no outbound network calls — pitched at the regulated-buyer audience.

**Validation:** realistic pipeline with `RUDRIQ_CAPTURE_CONTENT=true` produces:
- 43/43 LLM nodes with `rudriq.prompt_preview`
- 20/43 LLM nodes with `rudriq.completion_preview` (chats; embeddings have no output text per spec)
- 10/11 data nodes with `rudriq.content_preview`; 3/3 linked-parent data nodes covered
- Real evaluator scores: `retrieval_relevance` 23/23 OK (mean 0.93), `groundedness` 20/20 OK + 3 SKIPPED (embedding spans have no response)
- 23/43 LLM-call lineage linkage preserved end-to-end (no regression)

8 new tests in `tests/test_content_capture_config.py`. 159 total passing.

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
