"""
Linker behavioral contracts.

These tests describe what the linker *will* do in v0.1. v0.0.1 ships
stubs that satisfy the API; the implementation lands next sprint week.
"""

from __future__ import annotations


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

    parent_id, method, confidence = correlate(
        llm_input="some-string",
        span_attributes={},
    )
    assert parent_id is None
    assert confidence == 0.0


def test_link_strategies_return_correct_method_labels() -> None:
    """Stubs return the right method label even when no match is found."""
    from rudriq.linker import (
        link_by_content_hash,
        link_by_name_match,
        link_by_object_identity,
    )

    _, method_a, conf_a = link_by_object_identity("x", {})
    _, method_b, conf_b = link_by_content_hash("x", {})
    _, method_c, conf_c = link_by_name_match("x", {})

    assert method_a == "object_identity" and conf_a == 1.0
    assert method_b == "content_hash" and conf_b == 0.8
    assert method_c == "name_match" and conf_c == 0.5


def test_install_linker_is_idempotent_and_safe_when_subsystems_disabled() -> None:
    """Calling install_linker with both flags False is a clean no-op."""
    from rudriq.linker import install_linker

    install_linker(lineage_enabled=False, llm_enabled=False)
    install_linker(lineage_enabled=True, llm_enabled=False)
    install_linker(lineage_enabled=False, llm_enabled=True)