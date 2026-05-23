"""Tests for content-capture opt-in config.

Content capture is OFF by default. Tests verify both polarities of every
config knob, plus the truncation marker shape (downstream consumers
parse it for total-length reporting).
"""

from __future__ import annotations

from rudriq.core.config import (
    DEFAULT_PREVIEW_CHARS,
    content_capture_enabled,
    preview_char_limit,
    truncate_preview,
)


def test_capture_off_by_default(monkeypatch):
    monkeypatch.delenv("RUDRIQ_CAPTURE_CONTENT", raising=False)
    assert content_capture_enabled() is False


def test_capture_on_with_truthy_values(monkeypatch):
    for val in ("1", "true", "TRUE", "yes", "on", "On"):
        monkeypatch.setenv("RUDRIQ_CAPTURE_CONTENT", val)
        assert content_capture_enabled() is True, f"expected True for {val!r}"


def test_capture_off_with_falsy_values(monkeypatch):
    for val in ("0", "false", "no", "off", "", "garbage"):
        monkeypatch.setenv("RUDRIQ_CAPTURE_CONTENT", val)
        assert content_capture_enabled() is False, f"expected False for {val!r}"


def test_preview_limit_default(monkeypatch):
    monkeypatch.delenv("RUDRIQ_PREVIEW_CHARS", raising=False)
    assert preview_char_limit() == DEFAULT_PREVIEW_CHARS


def test_preview_limit_clamped(monkeypatch):
    monkeypatch.setenv("RUDRIQ_PREVIEW_CHARS", "10")
    assert preview_char_limit() == 50  # clamped to minimum
    monkeypatch.setenv("RUDRIQ_PREVIEW_CHARS", "999999")
    assert preview_char_limit() == 10_000  # clamped to maximum


def test_preview_limit_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RUDRIQ_PREVIEW_CHARS", "not-a-number")
    assert preview_char_limit() == DEFAULT_PREVIEW_CHARS


def test_truncate_short_text_unchanged(monkeypatch):
    monkeypatch.setenv("RUDRIQ_PREVIEW_CHARS", "500")
    assert truncate_preview("short") == "short"


def test_truncate_long_text_marked(monkeypatch):
    monkeypatch.setenv("RUDRIQ_PREVIEW_CHARS", "50")
    long = "x" * 100
    result = truncate_preview(long)
    # First 50 chars preserved, marker follows.
    assert result.startswith("x" * 50)
    assert "truncated" in result
    assert "100 chars total" in result
