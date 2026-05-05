"""
Realistic RAG pipeline demo for RudriQ stress testing.

Goal: ~250 operations across data lineage and LLM calls. Surfaces bugs
that toy-scale notebooks miss (linker performance at scale, DuckDB
size, edge case interactions, memory growth).

Mocks OpenAI HTTP layer via httpx.MockTransport so the script runs
offline, deterministically, and without API costs.

Usage:
    python examples/realistic_rag_pipeline.py
"""

from __future__ import annotations

# IMPORTANT ORDER (load-bearing):
#   1. Set TRACELOOP_API_KEY *before* import rudriq.auto. rudriq.auto
#      runs _activate_traceloop() at import time, which calls
#      Traceloop.init(). Without an API key set, Traceloop bails before
#      swapping the global ProxyTracerProvider, leaving us with no
#      SDK provider to attach RudriQSpanProcessor to.
#   2. Then import rudriq.auto — auto-capture wrappers + AutoLineage
#      callback get wired.
#   3. Then construct RudriQSpanProcessor and add it to Traceloop's
#      provider.
import os
os.environ.setdefault("TRACELOOP_API_KEY", "tl_test_key_not_real")

import json
import tempfile
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

import rudriq.auto  # noqa: E402  (must come after env-var setup)
from openai import OpenAI
from opentelemetry import trace
from rudriq.processors import RudriQSpanProcessor
from rudriq.storage import get_default_storage
from rudriq.export.audit import export_audit_json, export_audit_markdown


# ---------------------------------------------------------------------
# OTel setup
# ---------------------------------------------------------------------

provider = trace.get_tracer_provider()
rudriq_proc = RudriQSpanProcessor()

# Defensive: when Traceloop is installed AND TRACELOOP_API_KEY is set,
# rudriq.auto already added a SpanProcessor and Traceloop swapped the
# global provider to an SDK one. Add ours too. If Traceloop isn't
# installed, the global stays a ProxyTracerProvider, which doesn't
# expose add_span_processor — fall back to creating our own.
if hasattr(provider, "add_span_processor"):
    provider.add_span_processor(rudriq_proc)
else:
    from opentelemetry.sdk.trace import TracerProvider
    fallback = TracerProvider()
    fallback.add_span_processor(rudriq_proc)
    trace.set_tracer_provider(fallback)
    provider = fallback

print(f"RudriQ run_id: {rudriq_proc.run_id}")
print(f"Provider type: {type(provider).__name__}")


# ---------------------------------------------------------------------
# Mock OpenAI HTTP layer
# ---------------------------------------------------------------------

def _mock_handler(request: httpx.Request) -> httpx.Response:
    """Return canned OpenAI responses based on the endpoint."""
    url = str(request.url)
    body = json.loads(request.content) if request.content else {}

    if "embeddings" in url:
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]
        n = len(inputs) or 1
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [0.01 * (i + j) for j in range(8)],
                    }
                    for i in range(n)
                ],
                "model": body.get("model", "text-embedding-3-small"),
                "usage": {"prompt_tokens": n * 5, "total_tokens": n * 5},
            },
        )
    if "chat/completions" in url:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": body.get("model", "gpt-4"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Mock response based on retrieved context.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )
    return httpx.Response(404, json={"error": "unknown endpoint"})


http_client = httpx.Client(transport=httpx.MockTransport(_mock_handler))
client = OpenAI(api_key="sk-test-fake", http_client=http_client)


# ---------------------------------------------------------------------
# Phase 1: Read 5 CSV files
# ---------------------------------------------------------------------

print("\n=== Phase 1: Reading source data ===")
phase_start = time.time()

tmp_dir = Path(tempfile.gettempdir()) / "rudriq_demo_data"
tmp_dir.mkdir(exist_ok=True)

for i in range(5):
    csv_path = tmp_dir / f"docs_{i}.csv"
    if not csv_path.exists():
        df = pd.DataFrame({
            "doc_id": [f"d{i}_{j}" for j in range(50)],
            "text": [
                f"document {i}.{j} about topic {j % 10}" for j in range(50)
            ],
            "lang": ["en" if j % 3 != 0 else "es" for j in range(50)],
            "score": np.random.RandomState(i).rand(50),
            "category": [f"cat_{j % 5}" for j in range(50)],
        })
        df.to_csv(csv_path, index=False)

dataframes = [pd.read_csv(tmp_dir / f"docs_{i}.csv") for i in range(5)]
print(f"  Read {len(dataframes)} CSVs, "
      f"{sum(len(d) for d in dataframes)} total rows")


# ---------------------------------------------------------------------
# Phase 2: Transformations
# ---------------------------------------------------------------------

print("\n=== Phase 2: Transformations ===")

transformed = []
for i, df in enumerate(dataframes):
    df_en = df[df["lang"] == "en"]
    df_dedup = df_en.drop_duplicates(subset=["text"])
    df_sorted = df_dedup.sort_values("score", ascending=False)
    df_top = df_sorted.head(30)
    transformed.append(df_top)

combined = pd.concat(transformed, ignore_index=True)
print(f"  Combined: {len(combined)} rows from 5 inputs")

grouped = combined.groupby("category").agg({"score": "mean"}).reset_index()
print(f"  Grouped: {len(grouped)} categories")

final_docs = combined.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------
# Phase 3: Embed in batches
# ---------------------------------------------------------------------

print("\n=== Phase 3: Batched embeddings ===")

texts = final_docs["text"].tolist()
batch_size = 50
all_embeddings = []

batch_t0 = time.time()
for batch_idx in range(0, len(texts), batch_size):
    batch = texts[batch_idx:batch_idx + batch_size]
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=batch,
    )
    all_embeddings.extend(e.embedding for e in response.data)
batch_dt = time.time() - batch_t0
print(f"  {len(all_embeddings)} embeddings in "
      f"{(len(texts) + batch_size - 1) // batch_size} batches "
      f"({batch_dt * 1000:.0f}ms)")


# ---------------------------------------------------------------------
# Phase 4: Build synthetic index
# ---------------------------------------------------------------------

print("\n=== Phase 4: Index construction ===")
embeddings_array = np.array(all_embeddings)
print(f"  Index shape: {embeddings_array.shape}")


# ---------------------------------------------------------------------
# Phase 5: Run 20 queries
# ---------------------------------------------------------------------

print("\n=== Phase 5: Running queries ===")

queries = [f"Tell me about topic {i}" for i in range(20)]

queries_t0 = time.time()
for q_idx, query in enumerate(queries):
    q_embed_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
    )
    q_vec = np.array(q_embed_response.data[0].embedding)

    norms = (np.linalg.norm(embeddings_array, axis=1)
             * np.linalg.norm(q_vec) + 1e-9)
    scores = embeddings_array @ q_vec / norms
    top_5_indices = scores.argsort()[-5:][::-1]
    top_docs = [texts[i] for i in top_5_indices]

    context = "\n".join(top_docs)
    messages = [
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    chat_response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
    )

    if (q_idx + 1) % 5 == 0:
        print(f"  Queries {q_idx - 3}-{q_idx + 1} done")

queries_dt = time.time() - queries_t0
phase_total = time.time() - phase_start
print(f"\n=== Pipeline complete in {phase_total:.2f}s "
      f"({queries_dt * 1000 / 20:.1f}ms per query avg) ===")


# ---------------------------------------------------------------------
# Phase 6: Inspect the trace
# ---------------------------------------------------------------------

print("\n=== Trace inspection ===")
storage = get_default_storage()
graph = storage.load_run(rudriq_proc.run_id)

if graph is None:
    print("  ERROR: No graph found in storage")
    raise SystemExit(1)

print(f"  Total nodes: {len(graph.nodes)}")
print(f"  Total edges: {len(graph.edges)}")

kind_counts = Counter(n.kind.value for n in graph.nodes)
for kind, count in sorted(kind_counts.items()):
    print(f"    {kind}: {count}")

edge_kinds = Counter(e.kind.value for e in graph.edges)
for kind, count in sorted(edge_kinds.items()):
    print(f"  Edge {kind}: {count}")

llm_node_ids = {n.node_id for n in graph.nodes if n.kind.value.startswith("llm_")}
linked_llm = {
    e.child_id for e in graph.edges
    if e.kind.value == "lineage" and e.child_id in llm_node_ids
}
print(f"  LLM calls linked to upstream: {len(linked_llm)}/{len(llm_node_ids)}")

# Check linker registry size — proxy for memory growth
from rudriq.linker import _object_registry
print(f"  Linker registry size: {len(_object_registry)} entries")


# ---------------------------------------------------------------------
# Phase 7: Audit export performance
# ---------------------------------------------------------------------

print("\n=== Audit export performance ===")

t1 = time.time()
audit_json = export_audit_json(rudriq_proc.run_id)
t2 = time.time()
audit_md = export_audit_markdown(rudriq_proc.run_id)
t3 = time.time()

print(f"  JSON export: {(t2 - t1) * 1000:.1f}ms, {len(audit_json)} bytes")
print(f"  Markdown export: {(t3 - t2) * 1000:.1f}ms, {len(audit_md)} bytes")

out_dir = Path(tempfile.gettempdir()) / "rudriq_demo_output"
out_dir.mkdir(exist_ok=True)
(out_dir / "audit.json").write_text(audit_json, encoding="utf-8")
(out_dir / "audit.md").write_text(audit_md, encoding="utf-8")
print(f"  Outputs: {out_dir}")


# ---------------------------------------------------------------------
# Phase 8: DuckDB size
# ---------------------------------------------------------------------

print("\n=== DuckDB size ===")
db_path = Path(os.environ.get("RUDRIQ_HOME", Path.home() / ".rudriq")) / "traces.duckdb"
if db_path.exists():
    print(f"  {db_path}: {db_path.stat().st_size:,} bytes")


# ---------------------------------------------------------------------
# Phase 9: Lineage chain depth distribution
# ---------------------------------------------------------------------

print("\n=== Lineage chain depths ===")
parsed = json.loads(audit_json)
chains = parsed.get("lineage_chains", [])
if chains:
    depths = [c["chain_length"] for c in chains]
    print(f"  Chains: {len(chains)}")
    print(f"  Depth min/median/max: "
          f"{min(depths)}/{sorted(depths)[len(depths)//2]}/{max(depths)}")
    longest = max(chains, key=lambda c: c["chain_length"])
    print(f"  Longest chain: {longest['chain_length']} steps "
          f"({longest['llm_library']}.{longest['llm_operation']})")
else:
    print("  No lineage chains")

print("\n=== Done. ===")
