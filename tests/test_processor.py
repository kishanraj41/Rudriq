"""Tests for RudriQSpanProcessor — the OTel hookpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture
def isolated_storage(tmp_path: Path, monkeypatch):
    """Each test gets its own DuckDB."""
    from rudriq.storage import duckdb_backend
    from rudriq.linker import clear_object_registry
    from rudriq.processors.linking import clear_input_registry

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "proc_test.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    clear_object_registry()
    clear_input_registry()
    yield db
    db.close()


@pytest.fixture
def tracer_provider():
    """Build a fresh TracerProvider for each test, register our processor."""
    from rudriq.processors import RudriQSpanProcessor

    provider = TracerProvider()
    rudriq_proc = RudriQSpanProcessor()
    provider.add_span_processor(rudriq_proc)
    yield provider, rudriq_proc


def test_processor_persists_genai_span_as_node(isolated_storage, tracer_provider):
    """A simple OpenAI-like span should land in DuckDB as an LLM_CHAT node."""
    provider, rudriq_proc = tracer_provider
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("openai.chat") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "gpt-4")

    graph = isolated_storage.load_run(rudriq_proc.run_id)
    assert graph is not None
    assert len(graph.nodes) == 1
    node = graph.nodes[0]
    assert node.library == "openai"
    assert node.metadata["gen_ai.request.model"] == "gpt-4"


def test_processor_links_llm_span_to_registered_object(
    isolated_storage, tracer_provider,
):
    """End-to-end: register an object, emit an OpenAI span with that object
    as input, and verify a LINEAGE_LINK edge appears."""
    from rudriq.linker import register_object_identity
    from rudriq.processors.linking import record_llm_input

    provider, rudriq_proc = tracer_provider
    tracer = provider.get_tracer("test")

    # 1) Simulate AutoLineage: register a tracked DataFrame-like object.
    upstream_data = {"text": ["doc1", "doc2", "doc3"]}
    register_object_identity(upstream_data, "upstream-node-XYZ")

    # 2) Emit an LLM span representing openai.embeddings.create(input=upstream_data).
    with tracer.start_as_current_span("openai.embeddings") as span:
        span.set_attribute("gen_ai.system", "openai")
        span.set_attribute("gen_ai.request.model", "text-embedding-3-small")
        # User-side instrumentation registers the raw input.
        span_id_hex = format(span.context.span_id, "016x")
        record_llm_input(span_id_hex, upstream_data)

    # 3) Verify the linker created a cross-domain edge.
    graph = isolated_storage.load_run(rudriq_proc.run_id)
    assert graph is not None
    assert len(graph.nodes) == 1
    llm_node = graph.nodes[0]

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.parent_id == "upstream-node-XYZ"
    assert edge.child_id == llm_node.node_id
    assert edge.kind.value == "lineage"
    assert edge.confidence == 1.0
    assert edge.link_method.value == "object_identity"

    # The node also carries the RUDRIQ_* attributes for downstream
    # observability tools.
    assert llm_node.metadata.get("rudriq.lineage_parent") == "upstream-node-XYZ"
    assert llm_node.metadata.get("rudriq.domain") == "linked"


def test_processor_links_via_content_hash_when_no_object_identity(
    isolated_storage, tracer_provider,
):
    """If object identity fails but content hash matches a stored node,
    we should still produce a LINEAGE_LINK with confidence 0.8."""
    from datetime import datetime, timezone
    from rudriq.core.schema import (
        NodeKind,
        TraceGraph,
        TraceNode,
        compute_content_hash,
    )
    from rudriq.processors.linking import record_llm_input

    # Pre-populate storage with a data node whose content_hash matches.
    payload = {"text": ["specific input"]}
    payload_hash = compute_content_hash(payload)

    upstream = TraceGraph(run_id="prior-run", created_at=datetime.now(timezone.utc))
    upstream.add_node(TraceNode(
        node_id="data-source-node",
        kind=NodeKind.DATA_TRANSFORM,
        library="pandas",
        operation="filter",
        started_at=datetime.now(timezone.utc),
        content_hash=payload_hash,
    ))
    isolated_storage.replace_run(upstream)

    provider, rudriq_proc = tracer_provider
    tracer = provider.get_tracer("test")

    # Emit a span whose input has identical content but different identity.
    different_object_same_content = {"text": ["specific input"]}
    with tracer.start_as_current_span("openai.embeddings") as span:
        span.set_attribute("gen_ai.system", "openai")
        span_id_hex = format(span.context.span_id, "016x")
        record_llm_input(span_id_hex, different_object_same_content)

    graph = isolated_storage.load_run(rudriq_proc.run_id)
    assert graph is not None
    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.parent_id == "data-source-node"
    assert edge.link_method.value == "content_hash"
    assert edge.confidence == 0.8


def test_processor_does_not_link_unrelated_http_span(
    isolated_storage, tracer_provider,
):
    """Non-GenAI spans should be persisted but never linked."""
    provider, rudriq_proc = tracer_provider
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("http.client.request") as span:
        span.set_attribute("http.url", "https://example.com")

    graph = isolated_storage.load_run(rudriq_proc.run_id)
    assert graph is not None
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0


def test_processor_does_not_wipe_pre_existing_edges_in_same_run(
    isolated_storage, tracer_provider,
):
    """
    Bug 5 regression: when AutoLineage writes a direct edge under run_id="X"
    and the SpanProcessor (also configured with run_id="X") emits its first
    span, the SpanProcessor's run-initialization step must not wipe the
    pre-existing edge.

    Original bug: _ensure_run_exists called save_run with an empty graph,
    which triggered DELETE FROM edges WHERE run_id = ?, wiping the
    AutoLineage-written edge.

    Fix: ensure_run is row-only and never touches edges.
    """
    from rudriq.core.schema import (
        EdgeKind, LinkMethod, NodeKind, TraceEdge, TraceNode,
    )
    from rudriq.processors import RudriQSpanProcessor
    from datetime import datetime, timezone

    # Use a known, shared run_id between the simulated AutoLineage writes
    # and the SpanProcessor.
    shared_run_id = "shared-run-bug5"

    # 1) Simulate AutoLineage writing two data nodes and a direct edge first.
    isolated_storage.ensure_run(shared_run_id)
    now = datetime.now(timezone.utc)
    isolated_storage.save_node(
        TraceNode(
            node_id="data-A",
            kind=NodeKind.DATA_READ,
            library="pandas",
            operation="read_csv",
            started_at=now,
        ),
        run_id=shared_run_id,
    )
    isolated_storage.save_node(
        TraceNode(
            node_id="data-B",
            kind=NodeKind.DATA_TRANSFORM,
            library="pandas",
            operation="filter",
            started_at=now,
        ),
        run_id=shared_run_id,
    )
    isolated_storage.save_edge(
        TraceEdge(
            parent_id="data-A",
            child_id="data-B",
            kind=EdgeKind.DIRECT,
            confidence=1.0,
            link_method=LinkMethod.OBJECT_IDENTITY,
        ),
        run_id=shared_run_id,
    )

    # Sanity check: the edge is there before the SpanProcessor runs.
    pre = isolated_storage.load_run(shared_run_id)
    assert len(pre.edges) == 1, "test setup: pre-existing edge missing"

    # 2) Build a fresh SpanProcessor configured with the SAME run_id.
    #    Don't use the autouse tracer_provider fixture — its processor
    #    has its own auto-generated run_id. We need to configure ours.
    provider = TracerProvider()
    rudriq_proc = RudriQSpanProcessor(run_id=shared_run_id)
    provider.add_span_processor(rudriq_proc)
    tracer = provider.get_tracer("bug5-test")

    # 3) Emit any span — even a non-GenAI one — to trigger ensure_run_exists.
    with tracer.start_as_current_span("http.client.request") as span:
        span.set_attribute("http.url", "https://example.com")

    # 4) The pre-existing edge MUST still be present.
    post = isolated_storage.load_run(shared_run_id)
    assert post is not None
    edge_ids = {(e.parent_id, e.child_id) for e in post.edges}
    assert ("data-A", "data-B") in edge_ids, (
        "Bug 5 regression: pre-existing edge was wiped when SpanProcessor "
        "initialized its run."
    )
    # And the http span node should also be present (the processor still
    # persists non-GenAI spans).
    operations = {n.operation for n in post.nodes}
    assert "http.client.request" in operations


def test_processor_handles_genai_span_with_no_link_match(
    isolated_storage, tracer_provider,
):
    """An LLM span with input we've never seen should persist without an edge."""
    from rudriq.processors.linking import record_llm_input

    provider, rudriq_proc = tracer_provider
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("openai.chat") as span:
        span.set_attribute("gen_ai.system", "openai")
        span_id_hex = format(span.context.span_id, "016x")
        record_llm_input(span_id_hex, "completely orphan input")

    graph = isolated_storage.load_run(rudriq_proc.run_id)
    assert graph is not None
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0
    # Node should be tagged as 'llm' but not 'linked'.
    assert graph.nodes[0].metadata.get("rudriq.domain") == "llm"
