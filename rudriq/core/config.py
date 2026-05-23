"""RudriQ runtime configuration, read from environment.

Content capture is OPT-IN and OFF BY DEFAULT. RudriQ does not store
prompt, response, or document text unless the operator explicitly
enables it. This is deliberate: prompt/response content frequently
contains PII or PHI, and RudriQ's regulated-buyer audience needs the
default to be privacy-preserving.

Enable:        export RUDRIQ_CAPTURE_CONTENT=true
Adjust limit:  export RUDRIQ_PREVIEW_CHARS=500   # 50..10000

Reading the env on every call (rather than caching at import time) lets
operators flip capture on/off without restarting the process, and lets
tests use ``monkeypatch.setenv`` without resetting cached state.
"""

from __future__ import annotations

import os

# Default preview length when content capture is enabled. Truncated to
# bound exposure while keeping enough text for semantic evaluation.
DEFAULT_PREVIEW_CHARS = 500

_MIN_PREVIEW_CHARS = 50
_MAX_PREVIEW_CHARS = 10_000


def content_capture_enabled() -> bool:
    """True only if the operator explicitly enabled content capture.

    Enable by setting ``RUDRIQ_CAPTURE_CONTENT`` to a truthy value
    (``1``, ``true``, ``yes``, ``on`` — case-insensitive). Anything
    else, including unset, means content capture is OFF.
    """
    raw = os.environ.get("RUDRIQ_CAPTURE_CONTENT", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def preview_char_limit() -> int:
    """Max characters stored per content preview. Configurable, defaulted.

    Clamped to ``[_MIN_PREVIEW_CHARS, _MAX_PREVIEW_CHARS]`` so a typo
    can't accidentally store unbounded text or shrink the preview below
    a usefully evaluable size.
    """
    raw = os.environ.get("RUDRIQ_PREVIEW_CHARS")
    if raw is None:
        return DEFAULT_PREVIEW_CHARS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_PREVIEW_CHARS
    return max(_MIN_PREVIEW_CHARS, min(value, _MAX_PREVIEW_CHARS))


def truncate_preview(text: str) -> str:
    """Truncate text to the configured preview length, with an ellipsis marker.

    The marker carries the original length so downstream consumers can
    reason about how much content was elided without storing the rest.
    """
    limit = preview_char_limit()
    if len(text) <= limit:
        return text
    return text[:limit] + f"… [truncated, {len(text)} chars total]"
