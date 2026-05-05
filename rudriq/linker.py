"""
The linker — RudriQ's core technical contribution (v0.0.2 — real impl).

Three matching strategies, in order of confidence:

1. Object identity (1.0) — same Python object via id().
2. Content hash (0.8) — same SHA-256 content fingerprint.
3. Name match (0.5) — heuristic correlation by variable/column name.

The linker queries DuckDB storage to find candidate parent nodes; in
v0.1 we'll also consult AutoLineage's in-memory tracker directly.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from rudriq.core.schema import LinkMethod, compute_content_hash
from rudriq.storage import get_default_storage

_LOG = logging.getLogger("rudriq.linker")

_GENAI_SPAN_PREFIXES = ("gen_ai.", "openai.", "anthropic.", "traceloop.")

# In-process registry of object-identity links. AutoLineage hooks
# register here when they create a tracked DataFrame; the linker
# consults it on every gen_ai.* span. This is process-local because
# id() is process-local.
_object_registry: dict[int, str] = {}


def register_object_identity(obj: Any, node_id: str) -> None:
    """Called by AutoLineage hooks (in v0.1) to register tracked outputs."""
    try:
        _object_registry[id(obj)] = node_id
    except Exception:
        pass


def clear_object_registry() -> None:
    """Test-only: reset the in-process registry."""
    _object_registry.clear()


class LinkerHook(Protocol):
    def __call__(
        self,
        llm_input: Any,
        span_attributes: dict[str, Any],
    ) -> tuple[str | None, str, float]:
        ...


# ---------------------------------------------------------------------------
# Real strategies
# ---------------------------------------------------------------------------


def link_by_object_identity(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """Match by Python id() against the in-process object registry.

    Strategies, in order:
    1. The input itself is registered (confidence 1.0).
    2. The input is a list/tuple and ANY of its elements is registered
       (confidence 0.95). This handles the common RAG pattern where the
       user passes ``texts[batch_idx:batch_idx + N]`` to embeddings —
       a fresh slice list whose elements share identity with the
       parent list. We scan elements rather than only checking the
       first because non-zero-aligned slices (e.g., texts[50:100])
       have a different first element than the parent list.
    """
    try:
        candidate = _object_registry.get(id(llm_input))
        if candidate is not None:
            return candidate, LinkMethod.OBJECT_IDENTITY.value, 1.0

        if isinstance(llm_input, (list, tuple)) and llm_input:
            # Cap the scan so a 1M-token input doesn't tank the linker.
            # For typical batch-embed sizes (50-1000), this is cheap.
            for elem in llm_input[:10_000]:
                candidate = _object_registry.get(id(elem))
                if candidate is not None:
                    return candidate, LinkMethod.OBJECT_IDENTITY.value, 0.95
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("object_identity match failed: %s", exc)

    return None, LinkMethod.OBJECT_IDENTITY.value, 1.0


def link_by_content_hash(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """Match by SHA-256 against persisted node content_hash values."""
    try:
        h = compute_content_hash(llm_input)
        if h is None:
            return None, LinkMethod.CONTENT_HASH.value, 0.8

        storage = get_default_storage()
        matches = storage.find_nodes_by_hash(h)
        if matches:
            # find_nodes_by_hash returns most-recent-first.
            _, node_id = matches[0]
            return node_id, LinkMethod.CONTENT_HASH.value, 0.8
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("content_hash match failed: %s", exc)

    return None, LinkMethod.CONTENT_HASH.value, 0.8


def link_by_name_match(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """Heuristic match by variable/column name. v0.1 implements; v0.0.2 stub."""
    return None, LinkMethod.NAME_MATCH.value, 0.5


_DEFAULT_STRATEGIES: tuple[LinkerHook, ...] = (
    link_by_object_identity,
    link_by_content_hash,
    link_by_name_match,
)


def correlate(
    llm_input: Any,
    span_attributes: dict[str, Any],
    strategies: tuple[LinkerHook, ...] = _DEFAULT_STRATEGIES,
) -> tuple[str | None, str, float]:
    """Apply strategies in order; return first non-None match."""
    for strategy in strategies:
        parent_id, method, confidence = strategy(llm_input, span_attributes)
        if parent_id is not None:
            return parent_id, method, confidence
    return None, "", 0.0


def install_linker(
    *,
    lineage_enabled: bool,
    llm_enabled: bool,
) -> None:
    """
    Install the linker as an OTel SpanProcessor.

    v0.0.2: registers callbacks but the SpanProcessor lands in v0.0.3.
    Object-identity registry is active immediately so AutoLineage hooks
    can begin populating it.
    """
    if not lineage_enabled and not llm_enabled:
        _LOG.debug("Linker not installed: no subsystems active.")
        return

    _LOG.info(
        "Linker installed (lineage=%s, llm=%s). SpanProcessor lands v0.0.3.",
        lineage_enabled,
        llm_enabled,
    )


def is_genai_span(span_name: str) -> bool:
    return any(span_name.startswith(p) for p in _GENAI_SPAN_PREFIXES)
