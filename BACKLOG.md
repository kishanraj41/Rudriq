# RudriQ Backlog

Things deferred from the sprint, tracked here so they don't get lost.

## Open

### Linker registry growth is unbounded
**Status:** Open
**Priority:** Before any long-running production deployment
**Origin:** Day 8 / v0.0.7

`rudriq.linker._object_registry` is a `dict[int, str]` that grows on every captured output. After Day 8's per-element tolist registration, the growth rate is roughly 1 entry per text element passed through embeddings. For a long-running RAG service processing millions of documents, this becomes a memory leak.

**Fix candidates:** LRU cap (e.g., 100k entries, evict oldest); use weakref where Python type permits; expose `register_object_identity_with_ttl(...)` for callers that want explicit lifecycle.

**Day 8 partial mitigation:** the `Series.tolist` per-element registration is gated to lists with fewer than 100k elements. So a single 1M-token tolist won't cripple the registry. But long-running pipelines accumulate.

### Lineage edge for query / chat completions
**Status:** Open / informational
**Origin:** Day 8 / realistic_rag_pipeline.py

The realistic RAG pipeline links 3/43 LLM calls — only the batch embeddings whose inputs preserve element identity from a tolist'd Series. Query embeddings (fresh strings) and chat completions (fresh messages dicts containing format-string content) cannot link via object identity.

Linking those properly would require:

1. A retrieval-aware linker that understands the prompt-template pattern: the "context" inside a chat message often quotes documents that ARE registered upstream. The linker could fuzzy-match content fragments against tracked content_hashes.
2. Or framework-level integration with RAG libraries (LangChain, LlamaIndex) where the framework owns the prompt assembly and can register lineage explicitly.

For v1.0 this is "linker strategies pluggable + retrieval-aware default."

### Thread-local input fallback breaks under concurrency
**Status:** Open
**Origin:** Day 8 / v0.0.7

`rudriq.processors.linking.set_recent_input_fallback` is the layer that closes the OpenLLMetry span-context gap. It's a thread-local "most recent input" slot — last-writer-wins per thread. Sequential calls correlate correctly. Concurrent or deeply-nested LLM calls degrade (the second call's input overwrites the first before the first's span ends).

**Fix candidates:** OTel context-keyed input (uses context propagation correctly across async boundaries); per-thread LIFO stack (handles nested calls but still racy under concurrency).

## Resolved

### v0.0.6: Mirror AutoLineage records into RudriQ DuckDB ✅
**Resolved:** May 5, 2026 (Day 7)
**Mechanism:** AutoLineage v0.6.0 added `register_post_record_callback`. rudriq.auto wires a callback that mirrors each `TransformationRecord` as a `TraceNode` in DuckDB and creates `EdgeKind.DIRECT` edges from each `parent_id`. `RudriQSpanProcessor.__init__` wires its `run_id` into the mirroring callback so subsequent records land in the run that hosts the LLM spans.
**Result:** Audit reports for in-process pipelines now show full data→LLM ancestry without `external` placeholders. Demo notebook produces 2 nodes + 2 edges (data filter + LLM embed; direct read→filter + lineage filter→LLM).
**Caveat:** The `external/unmirrored` placeholder is still emitted when (a) autolineage<0.6 is installed, or (b) AutoLineage records fire before any `RudriQSpanProcessor` is constructed (the run_id setter is called from `__init__`). Audit exporter's note explains this.

### Production Traceloop-managed flow verification ✅
**Resolved:** May 5, 2026 (Day 7)
**Mechanism:** `tests/test_traceloop_integration.py` exercises the production path: `Traceloop.init()` (with a fake `TRACELOOP_API_KEY` so its SDK provider is created), `RudriQSpanProcessor` added to that provider, an openai call through OpenLLMetry's instrumentor with `httpx.MockTransport` for offline run.
**Result:** 2 tests pass, 1 self-skips (the lineage-edge test self-skips when Traceloop's openai instrumentor's serialization path bypasses our auto_capture wrapper — that's a known degradation mode for some openai-SDK versions, not a regression).
**Bonus:** Production hardening — `RudriQSpanProcessor.on_end` now wraps its body in a defensive `try/except` so a stale processor (e.g., one whose DuckDB handle was closed by a prior test or a multi-tenant scenario) cannot poison sibling processors on the same TracerProvider.

## v0.0.7+ candidates

### After ~1 week of soak: PyPI publish of autolineage 0.5.0 / 0.6.0
**Status:** Open
**Target:** May 12+

autolineage v0.5.0 + v0.6.0 are tagged on github. RudriQ's `[lineage]` extra resolves them via git URL. Asymmetric risk: a buggy public release on PyPI is permanent (can't reuse version numbers). A week of RudriQ integration use will surface any bugs before the public release.

**To publish:**

```powershell
cd C:\Users\kisha\OneDrive\Documents\AI\autolineage
git checkout v0.6.0
python -m build
twine upload dist/autolineage-0.6.0*
```

Then update RudriQ's `pyproject.toml` `[lineage]` extra to `"autolineage>=0.6,<0.7"` and remove the git URL.

### PDF export of audit reports
**Status:** Stub raises NotImplementedError
**Origin:** Day 5

The CLI accepts `--format pdf` and prints a friendly error pointing at pandoc. Real PDF rendering (probably via reportlab or weasyprint, or rendering markdown to PDF in-process) is deferred to v0.3 / when a design partner asks for it.

### User-supplied audit templates
**Status:** Stub raises NotImplementedError
**Origin:** Day 5

`generate_audit_report(template="custom-internal")` raises NotImplementedError. Real templates would let a customer supply their own Jinja2 template that maps the trace graph onto their internal compliance format. Deferred to v0.3.

### Lineage edge with object_identity through Traceloop's openai instrumentor
**Status:** Open / informational
**Origin:** Day 7

`tests/test_traceloop_integration.py::test_traceloop_path_with_registered_object_creates_lineage_edge` self-skips when Traceloop's openai instrumentor's serialization path bypasses our auto_capture wrapper. Re-confirm against future openai/Traceloop SDK versions (the wrapper-vs-instrumentor layering is the brittle bit; pin a known-good combination in `[llm]` extras when this stabilizes).
