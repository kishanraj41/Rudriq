# Changelog

## Unreleased

### Changed
- AutoLineage floor raised to 0.7.1 (`autolineage>=0.7.1,<0.8` in the `lineage` and `all` extras). 0.7.1 gives every derived pandas frame its own lineage id; 0.7.0 and older recorded `parent == child` for most transforms. The previous pin (`<0.7`) also blocked every 0.7.x release.
- README rewritten: research-prototype status, install from GitHub (not on PyPI yet), a quickstart that obtains `run_id` the way the examples do, and no compliance, HIPAA BAA, SOC 2 or enterprise-tier claims.

### Fixed
- AutoLineage records are mapped to the right node kinds. AutoLineage emits `train`, `predict`, `evaluate`, `preprocess` and `split`; RudriQ only knew `model_train` and `model_predict`, so every scikit-learn node was stored as `unknown`. Both spellings are accepted now.

### Tests
- `tests/test_autolineage_071.py` drives real pandas through AutoLineage's hooks into RudriQ storage and checks there are no self-loop edges, plus the category mapping. The self-loop guard in `DeviationRCA` stays for traces from older AutoLineage versions or other adapters.

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
