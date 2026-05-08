"""Tests for the retrieval-aware (substring) linker strategy."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_registries():
    """Each test starts with empty registries."""
    from rudriq.linker import clear_object_registry
    clear_object_registry()
    yield
    clear_object_registry()


def test_substring_linker_matches_upstream_doc_in_prompt():
    """A chat message containing an upstream document should link to it."""
    from rudriq.linker import register_object_identity, link_by_substring

    upstream_doc = (
        "The quick brown fox jumps over the lazy dog and continues "
        "running through the forest"
    )
    register_object_identity(upstream_doc, "upstream-doc-node")

    messages = [
        {"role": "system", "content": "Answer based on this context."},
        {
            "role": "user",
            "content": (
                f"Context:\n{upstream_doc}\n\n"
                "Question: What is the fox doing?"
            ),
        },
    ]

    parent_id, method, confidence = link_by_substring(messages, {})
    assert parent_id == "upstream-doc-node"
    assert method == "substring"
    assert confidence >= 0.5


def test_substring_linker_returns_none_when_no_match():
    """If no upstream string is contained in the LLM input, return no match."""
    from rudriq.linker import register_object_identity, link_by_substring

    register_object_identity(
        "Some completely unrelated content here that is long enough", "node-X"
    )
    messages = [{"role": "user", "content": "Tell me about cats and dogs"}]
    parent_id, method, confidence = link_by_substring(messages, {})

    assert parent_id is None
    assert method == "substring"
    assert confidence == 0.0


def test_substring_linker_skips_short_strings():
    """Strings below the minimum length should not produce matches."""
    from rudriq.linker import register_object_identity, link_by_substring

    register_object_identity("hello", "short-node")
    messages = [{"role": "user", "content": "Please say hello to the user kindly"}]

    parent_id, _, _ = link_by_substring(messages, {})
    assert parent_id is None


def test_substring_linker_handles_list_of_strings():
    """Batch embedding input (list of strings) should work."""
    from rudriq.linker import register_object_identity, link_by_substring

    documents = [
        "The first document about machine learning fundamentals",
        "The second document about deep learning architectures",
    ]
    register_object_identity(documents, "batch-node")

    # Re-create a list with same content (different object identity).
    fresh_list = [
        "The first document about machine learning fundamentals",
        "The second document about deep learning architectures",
    ]
    parent_id, _, confidence = link_by_substring(fresh_list, {})

    assert parent_id == "batch-node"
    assert confidence >= 0.5


def test_substring_linker_handles_query_embedding_string():
    """A single-string LLM input that's a substring of an upstream string should match."""
    from rudriq.linker import register_object_identity, link_by_substring

    full_query_template = (
        "User asks: tell me everything about quantum computing applications today"
    )
    register_object_identity(full_query_template, "query-source-node")

    actual_input = "tell me everything about quantum computing applications today"
    parent_id, _, _ = link_by_substring(actual_input, {})

    assert parent_id == "query-source-node"


def test_substring_linker_finds_best_match_among_multiple():
    """When multiple upstream strings overlap, pick the best (highest score)."""
    from rudriq.linker import register_object_identity, link_by_substring

    register_object_identity(
        "common substring shared between many docs in the corpus",
        "common-node",
    )
    register_object_identity(
        "highly specific text that matches the LLM input strongly",
        "specific-node",
    )

    messages = [{
        "role": "user",
        "content": (
            "Quote: 'highly specific text that matches the LLM input strongly' "
            "— please confirm"
        ),
    }]

    parent_id, _, _ = link_by_substring(messages, {})
    assert parent_id == "specific-node"


def test_substring_linker_handles_multimodal_content():
    """Chat messages with content as a list of typed parts (multimodal) should work."""
    from rudriq.linker import register_object_identity, link_by_substring

    upstream = (
        "Important reference document content for the LLM to use as context"
    )
    register_object_identity(upstream, "multimodal-source")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Please analyze: {upstream}"},
            {"type": "image_url", "image_url": "..."},
        ],
    }]

    parent_id, _, _ = link_by_substring(messages, {})
    assert parent_id == "multimodal-source"


def test_substring_linker_confidence_scales_with_match_strength():
    """A near-perfect match should get higher confidence than a marginal one."""
    from rudriq.linker import (
        clear_object_registry, link_by_substring, register_object_identity,
    )

    # Strong match: candidate IS the upstream — fractions both 1.0.
    strong = (
        "this is a substantial document with lots of meaningful content here"
    )
    register_object_identity(strong, "strong-node")
    _, _, strong_confidence = link_by_substring(strong, {})

    clear_object_registry()

    # Marginal match: a 32-char upstream embedded in a 230-char candidate.
    register_object_identity(
        "exactly thirty character string!",  # 32 chars, just past threshold
        "marginal-node",
    )
    longer_input = (
        "exactly thirty character string! "
        + "x" * 200
    )
    _, _, marginal_confidence = link_by_substring(longer_input, {})

    assert strong_confidence >= marginal_confidence


def test_correlate_dispatches_to_substring_when_object_identity_misses():
    """End-to-end through the correlate dispatcher."""
    from rudriq.linker import correlate, register_object_identity

    upstream = (
        "Long enough document content here for matching against an LLM input"
    )
    register_object_identity(upstream, "dispatch-test-node")

    # Build a fresh messages list — different object identity from upstream.
    fresh = [{"role": "user", "content": f"Reference: {upstream} — analyze."}]

    parent_id, method, confidence = correlate(fresh, {})

    assert parent_id == "dispatch-test-node"
    assert method == "substring"
    assert confidence > 0.0


def test_extract_llm_input_strings_handles_edge_cases():
    """The string extractor should handle weird inputs gracefully."""
    from rudriq.linker import _extract_llm_input_strings

    assert _extract_llm_input_strings(None) == []
    assert _extract_llm_input_strings("") == [""]
    assert _extract_llm_input_strings([]) == []
    assert _extract_llm_input_strings([1, 2, 3]) == []
    assert _extract_llm_input_strings(["hello", 42, "world"]) == ["hello", "world"]
    assert _extract_llm_input_strings([{"role": "user", "content": 42}]) == []
    assert _extract_llm_input_strings([{"role": "user"}]) == []


def test_extract_strings_handles_pandas_series():
    """The content extractor should handle pandas Series of strings."""
    pytest.importorskip("pandas")
    import pandas as pd
    from rudriq.linker import _extract_strings

    s = pd.Series(["doc 1", "doc 2", "doc 3"])
    result = _extract_strings(s)
    assert result == ["doc 1", "doc 2", "doc 3"]

    # Numeric Series should return empty
    s2 = pd.Series([1, 2, 3])
    assert _extract_strings(s2) == []


def test_first_write_wins_preserves_bulk_content():
    """A bulk content registration must NOT be overwritten by a
    smaller per-element registration on the same node_id. This is the
    invariant that makes the substring linker work after Day 8's
    per-element tolist propagation: the Series's 50 strings must
    remain in _content_registry even though each element later calls
    register_object_identity with the SAME node_id."""
    from rudriq.linker import (
        _content_registry, register_object_identity,
    )

    docs = [f"document number {i} about topic of interest here" for i in range(20)]

    # Bulk registration first (the Day 8 tolist patch's first call).
    register_object_identity(docs, "shared-lid")
    assert len(_content_registry["shared-lid"]) == 20

    # Per-element registrations on the SAME lid (the Day 8 patch's
    # subsequent calls). These must NOT overwrite the bulk content.
    for elem in docs:
        register_object_identity(elem, "shared-lid")

    assert len(_content_registry["shared-lid"]) == 20, (
        "first-write-wins violated: per-element reg overwrote bulk content"
    )
