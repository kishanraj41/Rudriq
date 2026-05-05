# Contributing to RudriQ

Thanks for considering a contribution. RudriQ is in active sprint development; we welcome issues, discussions, and PRs.

## Quick start

```bash
git clone https://github.com/kishanraj41/rudriq.git
cd rudriq
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev,llm,lineage]"
pytest tests/ -v
```

You should see 78+ tests passing.

## What we welcome

- **Bug reports.** Especially anything where RudriQ produces wrong output (incorrect linking, wrong audit report content, broken roundtrip). Open an issue with a minimal reproducer.
- **Documentation improvements.** Any docs change that would have helped you understand the project faster.
- **Tests.** Additional tests for edge cases, especially in the linker and audit exporter.
- **New linker strategies.** The three current strategies (object identity, content hash, name match) cover the common cases. New strategies for new domains are welcome via the `LinkerHook` Protocol.

## What we are deferring

- **New LLM provider hooks.** We rely on [OpenLLMetry](https://github.com/traceloop/openllmetry) for SDK coverage. Contribute to OpenLLMetry instead.
- **Hosted dashboards.** RudriQ is self-hosted-first. We do not plan to build a hosted UI.
- **Cryptographic signing of artifacts.** Deferred to v1.0+.

## Code style

- Type hints required on all public APIs.
- Docstrings on all public functions and classes.
- `ruff` for linting (`ruff check rudriq/`).
- `pytest` for testing. Every bug fix requires a regression test — see Day 2's bug-1 through bug-5 fixes for examples of how regressions are documented inline.

## Pull requests

PRs welcome from anyone. We review based on:

1. Does it solve a real problem?
2. Is it tested?
3. Does it preserve existing behavior?
4. Is it consistent with the architectural commitments below?

For substantial changes, open an issue first to discuss approach.

## Architectural commitments

RudriQ has five locked-in architectural rules. PRs that violate them will be asked to revise:

1. **Zero cloud dependencies in Core.** No outbound network calls in the core library. DuckDB by default; everything self-hosted.
2. **RudriQ's canonical schema is canonical, not OTel.** OpenTelemetry is one input/output format; the internal data model is richer (first-class causal edges, cross-domain lineage links, audit metadata).
3. **OSS Core is genuinely free forever.** No crippleware to upsell paid tier.
4. **Every feature serves either OSS funnel or paid customer (or both).** No features that serve neither.
5. **Boring/working/shippable beats clever.** Day 2's split of `replace_run` vs `ensure_run` is canonical: the destructive path is named for what it does; helpers are explicit; nothing is implicit.

## License

By contributing to RudriQ, you agree that your contributions will be licensed under the MIT License.
