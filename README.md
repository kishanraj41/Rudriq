# RudriQ

> **Trace a bad LLM answer back to the data step that caused it.**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange.svg)](https://github.com/kishanraj41/rudriq)
[![Paper](https://img.shields.io/badge/paper-SSRN-blue.svg)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7222278)

When an LLM application gives a wrong or drifting answer, the cause is often upstream of the model call: a filter dropped the right documents, a merge duplicated rows, or retrieval returned stale text. RudriQ joins two traces that are usually kept apart:

- data operations (pandas and scikit-learn, recorded by [AutoLineage](https://github.com/kishanraj41/autolineage))
- LLM calls (OpenAI, through [OpenLLMetry](https://github.com/traceloop/openllmetry) and OpenTelemetry)

It links them into one graph and ranks the upstream operations most likely to explain a change. Everything runs locally: traces go to a DuckDB file on your machine, and the core makes no outbound network calls.

**Status: research prototype.** It works on the bundled examples and in its test suite. Nobody outside the author has run it in production yet. Expect the API to change.

## Install

RudriQ is not on PyPI yet. Install it from GitHub:

```bash
pip install "rudriq[lineage,llm] @ git+https://github.com/kishanraj41/rudriq"
```

| Extra | What it adds |
|---|---|
| `lineage` | AutoLineage (>= 0.7.1) |
| `llm` | OpenLLMetry and the OpenAI SDK |
| `evaluate` | Optional: the evaluators |
| `pdf` | Optional: PDF export |

## Quickstart

```python
import rudriq.auto  # wires AutoLineage, OpenLLMetry and the linker

from opentelemetry import trace
from rudriq.processors import RudriQSpanProcessor

proc = RudriQSpanProcessor()
provider = trace.get_tracer_provider()
if hasattr(provider, "add_span_processor"):
    provider.add_span_processor(proc)
run_id = proc.run_id

# Your existing code runs unchanged.
import pandas as pd
from openai import OpenAI

docs = pd.read_csv("docs.csv")
docs = docs[docs["lang"] == "en"]
OpenAI().embeddings.create(model="text-embedding-3-small", input=docs["text"].tolist())

from rudriq.export.audit import export_audit_markdown
print(export_audit_markdown(run_id))
```

From the command line:

```bash
rudriq audit --run-id <id> --format markdown --output report.md
rudriq diagnose --run-id <id> --target <node-id>   # ranked upstream suspects
```

## How the link is made

The linker tries three strategies in order and keeps the first one that matches:

| Strategy | Confidence | Mechanism |
|---|---|---|
| Object identity | 1.0 | The object passed to the LLM call is the same Python object AutoLineage recorded as an operation's output |
| Content hash | 0.8 | A SHA-256 hash of the input matches a recorded output |
| Retrieval-aware substring | 0.7 / 0.5 | Tracked text appears inside the prompt |

On the realistic example, 23 of 43 LLM calls are linked. The other 20 are query embeddings built from fresh strings with no upstream data, and they are left unlinked on purpose. The method and its measured accuracy are in the [RudriQ linker paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7222278) (SSRN preprint, not peer reviewed).

Root-cause ranking (`rudriq diagnose`) scores each upstream operation on three things:

- how far it deviates from a baseline run
- how close it sits in the lineage graph
- how confident the links on the path are

The output is a ranked list of suspects, not a causal proof.

## Data handling

**Default: metadata only.** RudriQ stores shapes, timings, model names, token counts and lineage links. It does not store prompt, response or document text.

**Opt-in content capture.** The evaluators (groundedness, retrieval relevance) need text. To enable them, set `RUDRIQ_CAPTURE_CONTENT=true`. With capture on:

- RudriQ stores truncated previews: 500 characters by default, set with `RUDRIQ_PREVIEW_CHARS` (50–10000).
- Previews go only to the local DuckDB file (`~/.rudriq/traces.duckdb`).
- The flag is read at process start. Turning it off later does not redact runs already captured.

If your data contains PII or PHI, keep capture off or use synthetic data. Review the stored previews before you share an export.

## Examples

All examples run offline. The OpenAI HTTP layer is mocked, so you need no API key.

```bash
git clone https://github.com/kishanraj41/rudriq.git && cd rudriq
pip install -e ".[dev,llm,lineage]"
pytest tests/
jupyter notebook examples/rag_with_lineage.ipynb        # 1 read, 1 filter, 1 LLM call
jupyter notebook examples/realistic_rag_pipeline.ipynb  # 5 CSVs, ~200 operations, 43 LLM calls
jupyter notebook examples/drift_demo.ipynb              # baseline vs. a perturbed run
```

## Next

These are plans, not promises:

- `rudriq demo`: one command, offline, ends in a report.
- `rudriq mcp`: a read-only MCP server, so an assistant can ask why a run changed.
- A PyPI release.

Anything beyond that waits for what early users ask for. See [BACKLOG.md](BACKLOG.md).

## License and author

MIT. Built by [Kishan Raj VG](https://github.com/kishanraj41) in Austin, TX, who also wrote the [AutoLineage preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6683825).
