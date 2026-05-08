"""
The linker — RudriQ's core technical contribution.

Four matching strategies, in order of confidence:

1. Object identity (1.0) — same Python object via id().
2. Content hash (0.8) — same SHA-256 content fingerprint.
3. Substring (0.7 / 0.5) — retrieval-aware: an upstream string is
   contained in the LLM input (or vice versa). Closes the chat-
   completion gap where the input is a freshly-built messages list
   whose ``content`` field embeds upstream documents.
4. Name match (0.5) — heuristic correlation by variable/column name
   (stub; v0.1).

The linker maintains two parallel registries:

* ``_object_registry`` (id(obj) -> node_id) — for identity matching.
* ``_content_registry`` (node_id -> list of strings) — for substring
  matching. First-write-wins: a bulk-content registration (e.g., a
  Series of 50 strings) is preserved against subsequent per-element
  registrations by Day 8's tolist propagation patch.
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

# Parallel content registry: node_id -> list of strings extracted from
# the tracked object. Used by link_by_substring for retrieval-aware
# matching. First-write-wins: once a node_id has content, subsequent
# registrations of the same node_id (e.g., per-element registrations
# from Day 8's tolist propagation) do NOT overwrite. This preserves
# the bulk content (typically registered first) against finer-grained
# per-element registrations that follow.
_content_registry: dict[str, list[str]] = {}


def _extract_strings(obj: Any, max_strings: int = 10_000) -> list[str]:
    """
    Extract searchable strings from a tracked object.

    Handles:
    - str: returns [obj]
    - list/tuple of str: returns string elements
    - pandas Series of str dtype: returns the values as a list

    Returns at most ``max_strings`` to bound the cost of registration.
    Returns ``[]`` for objects with no extractable strings (numbers,
    DataFrames, custom classes, None, etc.).
    """
    if obj is None:
        return []

    if isinstance(obj, str):
        return [obj]

    if isinstance(obj, (list, tuple)):
        return [s for s in obj[:max_strings] if isinstance(s, str)]

    # pandas Series of strings — common case for ``df['text_col']``.
    try:
        import pandas as pd
        if isinstance(obj, pd.Series):
            if obj.dtype == object or pd.api.types.is_string_dtype(obj):
                return [
                    s for s in obj.head(max_strings).tolist()
                    if isinstance(s, str)
                ]
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("_extract_strings pandas branch failed: %s", exc)

    return []


def register_object_identity(
    obj: Any,
    node_id: str,
    extract_content: bool = True,
) -> None:
    """Register a tracked object so subsequent LLM calls can correlate.

    Maps ``id(obj) -> node_id`` for object-identity matching. When
    ``extract_content`` is True (the default), also extracts string
    content from ``obj`` into ``_content_registry[node_id]`` for the
    substring linker. First-write-wins on ``_content_registry``: a
    later call with the same node_id but smaller/empty content does
    NOT overwrite an earlier bulk registration.
    """
    try:
        _object_registry[id(obj)] = node_id
    except Exception:
        return

    if extract_content:
        try:
            strings = _extract_strings(obj)
            if strings and node_id not in _content_registry:
                _content_registry[node_id] = strings
        except Exception as exc:  # noqa: BLE001
            _LOG.debug("content extraction for %s failed: %s", node_id, exc)


def clear_object_registry() -> None:
    """Test-only: reset both the identity and content registries."""
    _object_registry.clear()
    _content_registry.clear()


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


def _extract_llm_input_strings(
    llm_input: Any, max_extracted: int = 100,
) -> list[str]:
    """Extract searchable strings from an LLM input.

    Handles the common LLM input shapes:

    * ``str``: returns ``[llm_input]``
    * ``list``/``tuple`` of str: returns the strings
    * list of message dicts (``{role, content}``): returns each
      ``content`` string. ``content`` may itself be a list of typed
      parts (multimodal), in which case we extract each ``text`` part.

    Caps at ``max_extracted`` strings to bound substring scanning.
    """
    if llm_input is None:
        return []

    if isinstance(llm_input, str):
        return [llm_input]

    if isinstance(llm_input, (list, tuple)):
        results: list[str] = []
        for item in llm_input[:max_extracted]:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    results.append(content)
                elif isinstance(content, list):
                    # Multimodal: list of {type, text} parts.
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text = part["text"]
                            if isinstance(text, str):
                                results.append(text)
        return results

    return []


def link_by_substring(
    llm_input: Any,
    span_attributes: dict[str, Any],
    *,
    min_match_length: int = 20,
    min_match_fraction: float = 0.3,
) -> tuple[str | None, str, float]:
    """Substring-based linker — retrieval-aware matching.

    Algorithm
    ---------
    1. Extract candidate strings from the LLM input (``str``, list of
       str for batch embeddings, list of message dicts for chat
       completions).
    2. For each candidate, scan ``_content_registry`` for upstream
       strings that contain it OR are contained within it.
    3. Score by combined fraction; pick the highest-scoring match.

    Thresholds
    ----------
    A match is considered only if:

    * Both the candidate and the upstream string are at least
      ``min_match_length`` characters (default 20). Below this,
      false-positive risk from common substrings dominates. 20 was
      chosen empirically: long enough that "Answer based on the"
      (19 chars) and similar boilerplate doesn't cross-link, short
      enough to handle real RAG retrieval chunks of 30-100 chars.
    * EITHER fraction (candidate-of-upstream OR upstream-of-candidate)
      meets ``min_match_fraction`` (default 0.3). The asymmetric guard
      lets a short upstream document fully contained in a long chat
      prompt (fraction_of_candidate ~5%) match — because the
      fraction_of_upstream is 1.0 (full containment).

    Returns
    -------
    ``(parent_node_id, "substring", confidence)``: confidence 0.7 for
    strong matches (combined score >= 0.6) or 0.5 for marginal ones.
    Returns ``(None, "substring", 0.0)`` on no match.
    """
    candidates = _extract_llm_input_strings(llm_input)
    if not candidates:
        return None, LinkMethod.SUBSTRING.value, 0.0

    # Snapshot to avoid holding any lock during scanning. Python's GIL
    # makes the dict copy atomic for our purposes.
    registry_snapshot = list(_content_registry.items())

    best_match_id: str | None = None
    best_score: float = 0.0

    for candidate in candidates:
        if len(candidate) < min_match_length:
            continue

        for node_id, upstream_strings in registry_snapshot:
            for upstream in upstream_strings:
                if len(upstream) < min_match_length:
                    continue

                # Either-direction containment.
                if upstream in candidate:
                    fraction_of_candidate = len(upstream) / len(candidate)
                    fraction_of_upstream = 1.0
                elif candidate in upstream:
                    fraction_of_candidate = 1.0
                    fraction_of_upstream = len(candidate) / len(upstream)
                else:
                    continue

                # Reject if BOTH directions are below threshold.
                if (
                    fraction_of_candidate < min_match_fraction
                    and fraction_of_upstream < min_match_fraction
                ):
                    continue

                score = (fraction_of_candidate + fraction_of_upstream) / 2.0
                if score > best_score:
                    best_score = score
                    best_match_id = node_id

    if best_match_id is None:
        return None, LinkMethod.SUBSTRING.value, 0.0

    confidence = 0.7 if best_score >= 0.6 else 0.5
    return best_match_id, LinkMethod.SUBSTRING.value, confidence


def link_by_name_match(
    llm_input: Any,
    span_attributes: dict[str, Any],
) -> tuple[str | None, str, float]:
    """Heuristic match by variable/column name. v0.1 implements; v0.0.2 stub."""
    return None, LinkMethod.NAME_MATCH.value, 0.5


_DEFAULT_STRATEGIES: tuple[LinkerHook, ...] = (
    link_by_object_identity,    # 1.0 / 0.95
    link_by_content_hash,        # 0.8
    link_by_substring,           # 0.7 / 0.5
    link_by_name_match,          # 0.5 (stub)
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
