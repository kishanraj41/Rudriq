# RudriQ Backlog

Things deferred from the sprint, tracked here so they don't get lost.

## Open

### Critical for June 1 design partner readiness

#### Retrieval-aware linker (v0.0.8 / Day 11-12)
**Status:** Open
**Priority:** Critical — without this, demo shows ~7% LLM call linkage
**Origin:** Day 8 realistic pipeline test

Object-identity linking covers batch embeddings but not chat completions (messages dicts contain prompt-formatted strings) or query embeddings (fresh strings). The realistic RAG pipeline links 3/43 LLM calls today.

Two approaches:

1. **Substring-matching with provenance.** When an LLM input contains substrings of upstream-tracked strings (e.g., the chat message's `content` field includes a registered DataFrame's text rows), emit a lineage edge with confidence ~0.7. Must avoid false positives on common substrings (apply a min-length threshold or hash-window approach).
2. **Framework-level integration with RAG libraries** (LangChain, LlamaIndex) where the framework owns the prompt assembly and can register lineage explicitly.

For v1.0 this is "linker strategies pluggable + retrieval-aware default."

#### LRU cap on linker registry
**Status:** Open
**Priority:** High — production safety
**Origin:** Day 8 unbounded growth observation

`rudriq.linker._object_registry` is a `dict[int, str]` that grows on every captured output. After Day 8's per-element tolist registration, growth is roughly 1 entry per text element passed through embeddings. After 250 ops it has ~191 entries; after a long-running production pipeline (10K+ ops with batch sizes of 1000), it grows toward memory exhaustion.

**Fix candidates:** LRU cap (e.g., 50K entries, evict oldest); use `weakref` where Python type permits; expose `register_object_identity_with_ttl(...)` for callers that want explicit lifecycle. Cache must be concurrent-safe.

**Day 8 partial mitigation:** the `Series.tolist` per-element registration is gated to lists with fewer than 100k elements, so a single 1M-token tolist cannot cripple the registry. But long-running pipelines still accumulate.

### Important but deferable

#### Concurrency-safe input stash
**Status:** Open
**Origin:** Day 8 thread-local fallback

`rudriq.processors.linking.set_recent_input_fallback` closes the OpenLLMetry span-context gap. It's a thread-local "most recent input" — last-writer-wins per thread. Sequential calls correlate correctly. Concurrent or deeply-nested LLM calls degrade (one thread's input can mask another's; nested calls overwrite the outer's stash before its span ends).

**Fix candidates:** OTel context-keyed input (uses context propagation correctly across async boundaries); per-thread LIFO stack (handles nested but still racy under concurrency).

#### Timezone discipline audit
**Status:** Open
**Origin:** Day 7 timezone bug

Two timezone bugs caught so far (Day 2 DuckDB roundtrip, Day 7 AutoLineage timestamp interpretation). A comprehensive review of every place we accept a timestamp from an external source: AutoLineage `TransformationRecord.timestamp` (naive local), OTel `start_time`/`end_time` (nanoseconds since epoch UTC), OpenAI response timestamps (Unix seconds). Every ingestion path should explicitly state and enforce its timezone convention.

#### Lineage edge through Traceloop openai instrumentor (object_identity test)
**Status:** Open / informational
**Origin:** Day 7

`tests/test_traceloop_integration.py::test_traceloop_path_with_registered_object_creates_lineage_edge` self-skips when Traceloop's openai instrumentor's serialization path bypasses our auto_capture wrapper. Day 8's thread-local fallback closes most of this gap in practice (the realistic pipeline links 3/43 calls now, not 0/43), but the by-span_id channel remains brittle. Re-confirm against future openai/Traceloop SDK versions; pin a known-good combination in `[llm]` extras when stable.

### Operational

#### PyPI publish of autolineage 0.6.x
**Status:** Open
**Target:** ~May 13 or later, after RudriQ has soaked the dependency for ~1 week

autolineage v0.5.0, v0.6.0, v0.6.1 are tagged on github. RudriQ's `[lineage]` extra resolves them via `git+...@v0.6.1`. Asymmetric risk: a buggy public release on PyPI is permanent (PyPI doesn't allow version reuse). A week of RudriQ integration runs will surface any bugs before the public release.

**To publish:**

```powershell
cd C:\Users\kisha\OneDrive\Documents\AI\autolineage
git checkout v0.6.1
python -m build
twine upload dist/autolineage-0.6.1*
```

Then update RudriQ's `pyproject.toml` `[lineage]` extra to `"autolineage>=0.6.1,<0.7"` and remove the git URL.

#### PDF export of audit reports
**Status:** Stub raises NotImplementedError
**Origin:** Day 5

The CLI accepts `--format pdf` and prints a friendly error pointing at pandoc. Real PDF rendering (via reportlab, weasyprint, or pandoc bridging) is deferred to v0.3 / when a design partner asks for it.

#### User-supplied audit templates
**Status:** Stub raises NotImplementedError
**Origin:** Day 5

`generate_audit_report(template="custom-internal")` raises NotImplementedError. Real templates would let a customer supply their own Jinja2 template that maps the trace graph onto their internal compliance format. Deferred to v0.3.

## Resolved

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
