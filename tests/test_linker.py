"""Linker behavioral tests, including the real v0.0.2 implementations."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch) -> None:
    """Each test gets a fresh DuckDB."""
    from rudriq.storage import duckdb_backend
    from rudriq.linker import clear_object_registry

    db = duckdb_backend.DuckDBStorage(db_path=tmp_path / "linker_test.duckdb")
    monkeypatch.setattr(duckdb_backend, "_default", db)
    clear_object_registry()
    yield
    db.close()


def test_is_genai_span_recognizes_otel_genai_prefix() -> None:
    from rudriq.linker import is_genai_span
    assert is_genai_span("gen_ai.chat.completions") is True
    assert is_genai_span("gen_ai.embeddings") is True


def test_is_genai_span_recognizes_openllmetry_prefixes() -> None:
    from rudriq.linker import is_genai_span
    assert is_genai_span("openai.chat") is True
    assert is_genai_span("anthropic.messages") is True
    assert is_genai_span("traceloop.workflow") is True


def test_is_genai_span_rejects_unrelated_spans() -> None:
    from rudriq.linker import is_genai_span
    assert is_genai_span("http.client.request") is False
    assert is_genai_span("db.query") is False


def test_correlate_returns_none_when_no_strategy_matches() -> None:
    from rudriq.linker import correlate
    parent_id, _, confidence = correlate("orphan-input", {})
    assert parent_id is None
    assert confidence == 0.0


def test_link_by_object_identity_matches_registered_object() -> None:
    from rudriq.linker import link_by_object_identity, register_object_identity

    obj = {"data": [1, 2, 3]}
    register_object_identity(obj, "node-XYZ")

    parent_id, method, conf = link_by_object_identity(obj, {})
    assert parent_id == "node-XYZ"
    assert method == "object_identity"
    assert conf == 1.0


def test_link_by_object_identity_returns_none_for_unregistered() -> None:
    from rudriq.linker import link_by_object_identity

    parent_id, _, _ = link_by_object_identity(["random"], {})
    assert parent_id is None


def test_link_by_object_identity_matches_first_element_of_list() -> None:
    """RAG pattern: df['text'].tolist() loses outer identity, elements survive."""
    from rudriq.linker import link_by_object_identity, register_object_identity

    payload = {"text": "hello"}
    register_object_identity(payload, "node-first")

    # Caller passes [payload, ...] (e.g., embeddings input)
    parent_id, method, conf = link_by_object_identity([payload], {})
    assert parent_id == "node-first"
    assert method == "object_identity"
    assert conf == 0.95  # slightly reduced confidence vs direct match


def test_link_by_content_hash_matches_persisted_node() -> None:
    from datetime import datetime, timezone
    from rudriq.core.schema import (
        NodeKind, TraceGraph, TraceNode, compute_content_hash,
    )
    from rudriq.linker import link_by_content_hash
    from rudriq.storage import get_default_storage

    payload = {"text": "specific input value"}
    payload_hash = compute_content_hash(payload)

    storage = get_default_storage()
    g = TraceGraph(run_id="r", created_at=datetime.now(timezone.utc))
    g.add_node(TraceNode(
        node_id="data-source",
        kind=NodeKind.DATA_TRANSFORM,
        library="pandas",
        operation="filter",
        started_at=datetime.now(timezone.utc),
        content_hash=payload_hash,
    ))
    storage.save_run(g)

    parent_id, method, conf = link_by_content_hash(payload, {})
    assert parent_id == "data-source"
    assert method == "content_hash"
    assert conf == 0.8


def test_link_by_content_hash_returns_none_when_no_match() -> None:
    from rudriq.linker import link_by_content_hash
    parent_id, _, _ = link_by_content_hash("nothing matches", {})
    assert parent_id is None


def test_install_linker_safe_when_disabled() -> None:
    from rudriq.linker import install_linker
    install_linker(lineage_enabled=False, llm_enabled=False)
    install_linker(lineage_enabled=True, llm_enabled=False)
    install_linker(lineage_enabled=False, llm_enabled=True)
    install_linker(lineage_enabled=True, llm_enabled=True)
