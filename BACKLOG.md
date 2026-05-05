# RudriQ Backlog

Things deferred from the sprint, tracked here so they don't get lost.

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
