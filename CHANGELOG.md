# Changelog

## v0.1.0 — May 23, 2026

First feature-complete release. The full originally-planned v0.1.0 scope is shipped and validated: cross-domain lineage linking, a five-metric evaluation engine, audit-report integration, and deviation-weighted root-cause analysis. 213 tests passing.

### Capture
- One-import activation (`import rudriq.auto`) wiring AutoLineage hooks, OpenLLMetry instrumentors, SDK input capture, and the span processor
- AutoLineage records mirrored into RudriQ's DuckDB for self-contained audit reports
- Concurrency-safe LLM-input capture via OTel context propagation (per-thread / per-asyncio-task isolation, peek+LRU on the by-span-id channel)

### Linking
- Three strategies in confidence order: object identity (1.0), content hash (0.8), retrieval-aware substring (0.7/0.5)
- 23/43 LLM calls linked on the realistic RAG pipeline; query embeddings correctly unlinked (fresh strings)
- LRU-bounded registries (default 50K, configurable via `RUDRIQ_LINKER_CACHE_SIZE`)

### Evaluation
- Five evaluators: retrieval relevance, groundedness, coherence, consistency, drift
- Local embeddings via fastembed (optional `rudriq[evaluate]` extra), graceful degradation, per-evaluator failure isolation
- Opt-in, privacy-first content capture (`RUDRIQ_CAPTURE_CONTENT`, default off, truncated, local-only)
- User-role messages captured separately from full assembled prompts (`rudriq.user_message_preview`) so consistency-style metrics don't collapse on shared retrieval context

### Audit reports
- JSON (schema `rudriq.audit/1.1`) and Markdown
- Byte-deterministic: identical runs produce identical bytes, no excluded fields — reports can be hashed as proof of provenance
- Evaluation results embedded with a traffic-light quality summary and a Notable findings list (`rudriq audit --include-evals`)
- Not-applicable SKIPs (e.g. groundedness on embedding spans) suppressed from Notable findings so the list shows real issues only

### Analysis
- Deviation-weighted root-cause ranking (`rudriq diagnose --run-id X --target NODE`): structural deviation + lineage proximity + path link-confidence. Self-loop and parallel-edge safe.
- Explicit framing: a ranked-suspects heuristic, NOT a causal proof. Full causal inference is a v1.0+ research track.

### Dependencies
- Built on AutoLineage 0.6.1 (published to PyPI; no git URL required)
- Self-hosted, zero outbound network calls in core
