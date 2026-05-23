"""Tests for the OTel ingest adapter — Day 17 Thread B focuses on
the role-separated parser that powers the user-message split.
"""

from __future__ import annotations

import json

from rudriq.adapters.otel_ingest import (
    _extract_messages_text,
    _parse_genai_messages,
)


def test_parse_genai_messages_separates_user_role():
    """system + user → distinct ``user`` / ``system`` / ``full`` bins."""
    messages = json.dumps([
        {"role": "system", "parts": [{"type": "text", "content": "You are helpful."}]},
        {"role": "user", "parts": [{"type": "text", "content": "What is the capital?"}]},
    ])
    parsed = _parse_genai_messages(messages)
    assert parsed["user"] == "What is the capital?"
    assert parsed["system"] == "You are helpful."
    assert "What is the capital?" in parsed["full"]
    assert "You are helpful." in parsed["full"]


def test_parse_genai_messages_handles_garbage():
    """Non-JSON, empty, and non-list inputs → all-empty result, no raise."""
    empty = {"full": "", "user": "", "system": ""}
    assert _parse_genai_messages("not json") == empty
    assert _parse_genai_messages("") == empty
    assert _parse_genai_messages('{"not": "a list"}') == empty
    assert _parse_genai_messages("null") == empty


def test_parse_genai_messages_handles_legacy_content_shape():
    """Older shape: top-level ``content`` string instead of ``parts`` list."""
    messages = json.dumps([
        {"role": "user", "content": "legacy-shape message"},
    ])
    parsed = _parse_genai_messages(messages)
    assert parsed["user"] == "legacy-shape message"
    assert parsed["full"] == "legacy-shape message"


def test_parse_genai_messages_multiple_user_messages_concatenated():
    """Two user-role messages → both land in ``user`` joined by newline."""
    messages = json.dumps([
        {"role": "user", "parts": [{"type": "text", "content": "first question"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "first answer"}]},
        {"role": "user", "parts": [{"type": "text", "content": "follow-up"}]},
    ])
    parsed = _parse_genai_messages(messages)
    assert "first question" in parsed["user"]
    assert "follow-up" in parsed["user"]
    # The assistant message is NOT user, so user must not contain it.
    assert "first answer" not in parsed["user"]
    # But full does contain everything.
    assert "first answer" in parsed["full"]


def test_extract_messages_text_backward_compat_returns_full():
    """The old API still works — returns the same ``full`` blob."""
    messages = json.dumps([
        {"role": "user", "parts": [{"type": "text", "content": "hello"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "hi"}]},
    ])
    assert _extract_messages_text(messages) == _parse_genai_messages(messages)["full"]


def test_parse_genai_messages_skips_non_text_parts():
    """Non-text parts (e.g. images) are ignored; only text content surfaces."""
    messages = json.dumps([
        {"role": "user", "parts": [
            {"type": "image", "url": "http://example/x.png"},
            {"type": "text", "content": "describe this"},
        ]},
    ])
    parsed = _parse_genai_messages(messages)
    assert parsed["user"] == "describe this"
