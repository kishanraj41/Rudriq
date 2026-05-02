"""Tests for OTel -> canonical adapter."""

from __future__ import annotations

from datetime import timezone

from rudriq.adapters.otel_ingest import (
    classify_span_kind,
    otel_span_to_node,
)
from rudriq.core.schema import NodeKind


def test_classify_chat_span_by_name() -> None:
    assert classify_span_kind("openai.chat", {}) == NodeKind.LLM_CHAT
    assert classify_span_kind("anthropic.messages.create", {}) == NodeKind.LLM_CHAT
    assert classify_span_kind("gen_ai.chat.completions", {}) == NodeKind.LLM_CHAT


def test_classify_embedding_span_by_name() -> None:
    assert (
        classify_span_kind("openai.embeddings.create", {}) == NodeKind.LLM_EMBEDDING
    )
    assert (
        classify_span_kind("gen_ai.embeddings", {}) == NodeKind.LLM_EMBEDDING
    )


def test_classify_falls_back_to_attributes() -> None:
    assert (
        classify_span_kind("vendor.api.call",
                           {"gen_ai.operation.name": "embeddings"})
        == NodeKind.LLM_EMBEDDING
    )
    assert (
        classify_span_kind("vendor.api.call",
                           {"gen_ai.operation.name": "chat"})
        == NodeKind.LLM_CHAT
    )


def test_classify_unknown_when_nothing_matches() -> None:
    assert classify_span_kind("http.client.request", {}) == NodeKind.UNKNOWN


def test_classify_uses_gen_ai_system_as_last_resort() -> None:
    # Generic span name but gen_ai.system is set
    kind = classify_span_kind("client.request", {"gen_ai.system": "openai"})
    assert kind == NodeKind.LLM_COMPLETION


def test_otel_span_to_node_extracts_basic_fields() -> None:
    node = otel_span_to_node(
        span_name="openai.chat",
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.usage.total_tokens": 150,
            "irrelevant.attr": "ignored",
        },
        start_time_ns=1_700_000_000_000_000_000,  # 2023-11-14T22:13:20Z
        end_time_ns=1_700_000_001_000_000_000,
        span_id="abc123",
    )

    assert node.node_id == "abc123"
    assert node.kind == NodeKind.LLM_CHAT
    assert node.library == "openai"
    assert node.metadata["gen_ai.request.model"] == "gpt-4"
    assert node.metadata["gen_ai.usage.total_tokens"] == 150
    assert "irrelevant.attr" not in node.metadata


def test_otel_span_to_node_preserves_rudriq_attributes() -> None:
    node = otel_span_to_node(
        span_name="openai.embeddings",
        attributes={
            "gen_ai.system": "openai",
            "rudriq.lineage_parent": "upstream-node-id",
            "rudriq.link_confidence": 0.95,
        },
        start_time_ns=1_700_000_000_000_000_000,
        span_id="span-1",
    )

    assert node.metadata["rudriq.lineage_parent"] == "upstream-node-id"
    assert node.metadata["rudriq.link_confidence"] == 0.95


def test_otel_span_to_node_emits_utc_timestamps() -> None:
    node = otel_span_to_node(
        span_name="openai.chat",
        attributes={"gen_ai.system": "openai"},
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_001_000_000_000,
        span_id="ts-test",
    )
    assert node.started_at.tzinfo == timezone.utc
    assert node.ended_at.tzinfo == timezone.utc


def test_operation_normalization_strips_redundant_library_prefix() -> None:
    """openai.embeddings.create with gen_ai.system=openai -> 'embeddings.create'"""
    node = otel_span_to_node(
        span_name="openai.embeddings.create",
        attributes={"gen_ai.system": "openai"},
        start_time_ns=1_700_000_000_000_000_000,
        span_id="norm-1",
    )
    assert node.library == "openai"
    assert node.operation == "embeddings.create"
    # Rendered as f"{library}.{operation}" should be clean.
    assert f"{node.library}.{node.operation}" == "openai.embeddings.create"


def test_operation_normalization_strips_gen_ai_prefix() -> None:
    """gen_ai.embeddings.create with gen_ai.system=openai -> 'embeddings.create'"""
    node = otel_span_to_node(
        span_name="gen_ai.embeddings.create",
        attributes={"gen_ai.system": "openai"},
        start_time_ns=1_700_000_000_000_000_000,
        span_id="norm-2",
    )
    assert node.library == "openai"
    assert node.operation == "embeddings.create"


def test_operation_normalization_preserves_unknown_libraries() -> None:
    """When library is unknown, leave the span name alone."""
    node = otel_span_to_node(
        span_name="custom.user.span",
        attributes={},
        start_time_ns=1_700_000_000_000_000_000,
        span_id="norm-3",
    )
    assert node.library == "unknown"
    assert node.operation == "custom.user.span"


def test_operation_normalization_idempotent_on_already_normalized() -> None:
    """Span name 'embeddings.create' stays 'embeddings.create'."""
    node = otel_span_to_node(
        span_name="embeddings.create",
        attributes={"gen_ai.system": "openai"},
        start_time_ns=1_700_000_000_000_000_000,
        span_id="norm-4",
    )
    assert node.operation == "embeddings.create"
