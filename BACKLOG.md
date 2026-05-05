# RudriQ Backlog

Things deferred from the sprint, tracked here so they don't get lost.

## v0.0.6 candidates

### Mirror AutoLineage records into RudriQ DuckDB

**Status:** Open
**Priority:** Before Day 7 demo polish
**Origin:** Day 4 / v0.0.4 demo notebook

The audit JSON's `lineage_chains` walks parents through edges, but when a parent_id points at an AutoLineage lineage ID (a string from AutoLineage's tracker, not a node in RudriQ's DuckDB), the chain step renders as `library="external", kind="unknown"`. The data is there in `al_tracker.nodes`; RudriQ just doesn't yet copy it.

**To fix:** extend rudriq.auto's AutoLineage callback wiring to also `save_node` a TraceNode mirror at registration time, using the AL lid as `node_id`. Probably also wants a `register_post_record_callback` in autolineage to mirror records as `EdgeKind.DIRECT` edges between data nodes.

When this lands, audit reports show the full data→LLM chain in a single `load_run` without falling back to AL's tracker.

### Production Traceloop flow verification

**Status:** Open
**Priority:** Before first design partner demo (target: late June 2026)
**Origin:** Day 4 / v0.0.4 verification gap

`v0.0.4`'s `[llm]` install verified install-time and wrap-time clean integration: 64 tests pass in a fresh venv with `[dev,llm,lineage]`, and our auto_capture wrapper survives Traceloop's wrap on top. The full production flow (Traceloop manages TracerProvider, user makes plain `client.embeddings.create(...)`, Traceloop's instrumentor emits the span, `RudriQSpanProcessor` on Traceloop's provider receives it and runs the linker) was NOT exercised — the demo notebook overrides the TracerProvider for offline reproducibility.

**To verify:**

1. Set `OPENAI_API_KEY` (or use a recorded VCR cassette).
2. Build a notebook that does NOT call `trace.set_tracer_provider`.
3. After `import rudriq.auto`, make a real OpenAI call.
4. Inspect the trace — confirm the LLM span landed in DuckDB and was linked to upstream pandas operations.

The plumbing is in place (rudriq.auto adds RudriQSpanProcessor to whatever `trace.get_tracer_provider()` returns after Traceloop init). The verification is the missing test.

### Publish autolineage 0.5.0 to PyPI

**Status:** Open — waiting for ~1 week of integration soak
**Origin:** Day 4 / Day 5

autolineage v0.5.0 is tagged at `v0.5.0` on github (commit `f18ed58` on `v0.4.1-cleanup` branch). RudriQ's `[lineage]` extra resolves it via `git+https://...@v0.5.0`. Asymmetric risk: a buggy 0.5.0 on PyPI is permanent (can't reuse version numbers). A week of RudriQ integration use will surface any bugs before the public release.

**To publish:**

```powershell
cd C:\Users\kisha\OneDrive\Documents\AI\autolineage
git checkout v0.5.0
python -m build
twine upload dist/autolineage-0.5.0*
```

Then update RudriQ's `pyproject.toml` `[lineage]` extra to `"autolineage>=0.5,<0.6"` and remove the git URL.

## v0.0.7+ candidates

### PDF export of audit reports

**Status:** Stub raises NotImplementedError
**Origin:** Day 5

The CLI accepts `--format pdf` and prints a friendly error pointing at pandoc. Real PDF rendering (probably via reportlab or weasyprint, or rendering markdown to PDF in-process) is deferred to v0.3 / when a design partner asks for it.

### User-supplied audit templates

**Status:** Stub raises NotImplementedError
**Origin:** Day 5

`generate_audit_report(template="custom-internal")` raises NotImplementedError. Real templates would let a customer supply their own Jinja2 template that maps the trace graph onto their internal compliance format. Deferred to v0.3.
