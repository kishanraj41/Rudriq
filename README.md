# RudriQ

> **Connect LLM behavior to its upstream data lineage.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange.svg)](https://github.com/kishanraj41/rudriq)

When your AI system breaks, RudriQ walks back through your data pipeline to the operation that caused it.

## The gap RudriQ fills

LLM observability tools (OpenLLMetry, Langfuse, Phoenix, LangSmith) tell you **what** happened in production: which prompt, which response, how many tokens. They are excellent at this and we recommend using them.

Data lineage tools (AutoLineage, OpenLineage, Marquez) tell you **what happened upstream**: which CSV, which transform, which model artifact.

**No tool connects the two.**

When an LLM response changes between runs, the cause is usually upstream — a stale embedding index, a filter that dropped the wrong rows, a CSV that was refreshed without notice. RudriQ is the bridge that makes those upstream causes visible from your LLM trace.

## Architecture

RudriQ is a *bridge*, not a tracing library. It builds on top of:

- **[OpenLLMetry](https://github.com/traceloop/openllmetry)** for LLM call instrumentation, emitted as OpenTelemetry GenAI semantic conventions.
- **[AutoLineage](https://github.com/kishanraj41/autolineage)** for data pipeline instrumentation across pandas, scikit-learn, and PySpark.

The novel contribution is the **linker** ([rudriq/linker.py](rudriq/linker.py)) which correlates spans from both domains into a single unified trace. The linker uses three matching strategies (object identity → content hash → name match), and annotates LLM spans with the upstream lineage operations that produced their input data.

## Quickstart

```bash
pip install "rudriq[all]"     # OpenLLMetry + AutoLineage + linker
```

```python
import rudriq.auto    # activates lineage tracking, LLM tracing, and the linker

# ... your existing pandas / scikit-learn / OpenAI / Anthropic / RAG code runs unchanged ...

from rudriq import diagnose
print(diagnose(target_metric='answer_quality'))
```

That's it. One import. Every pandas operation, every LLM call, every cross-domain link is in your unified trace.

## What you can do with the unified trace

| Capability | Status |
| --- | --- |
| Cross-domain root-cause analysis (`rudriq diagnose`) | v0.2 (May 21) |
| Audit report generation (`rudriq audit`) | v0.2 (May 27) |
| Slack alerts on anomalies | Future |
| Renders in Datadog / SigNoz / Jaeger / Langfuse / Phoenix | v0.1 (May 13) |

## Roadmap

| Version | Target date | Highlights |
| --- | --- | --- |
| **v0.0.1** | May 6, 2026  | Foundation, linker spec, demo notebook (this release) |
| v0.1.0 | May 13, 2026 | Linker implementation: object-identity + content-hash matching |
| v0.2.0 | May 27, 2026 | Cross-domain root-cause analyzer, audit report exporter |

## Why now

In 2026, LLM observability is converging on OpenTelemetry GenAI semantic conventions. Data lineage tools have always emitted their own formats. The seam between the two is empty. RudriQ fills that seam, using the standard so you don't get locked in.

## License

MIT — use it however you want.

## Author

Built by [Kishan Raj VG](https://github.com/kishanraj41) at RudriQ Research, Austin, TX.