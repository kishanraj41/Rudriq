"""Local sentence-embedding helper for evaluators that need semantic similarity.

Uses fastembed (optional ``evaluate`` extra). Degrades gracefully: if
fastembed is not installed, :func:`get_embedder` returns ``None`` and
callers produce a DEGRADED ``EvalResult`` rather than crashing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

_LOG = logging.getLogger("rudriq.evaluate.embeddings")

# Module-level cache: the embedding model is expensive to construct
# (loads the ONNX model into memory). Build once, reuse.
_embedder: Any = None
_embedder_attempted: bool = False

# fastembed default; ~130MB, good quality across general English text.
# Alternative: "sentence-transformers/all-MiniLM-L6-v2" is also offered
# by fastembed if a smaller / faster model is needed.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def get_embedder() -> Any | None:
    """Return a cached embedding model, or ``None`` if fastembed is unavailable.

    The model is constructed lazily on first call and cached. Subsequent
    calls return the cached instance. If fastembed is not installed (or
    the model fails to load), returns ``None`` — callers should produce
    a DEGRADED ``EvalResult``.

    We remember the failed-attempt state so that repeated calls don't
    keep paying for the ImportError each time.
    """
    global _embedder, _embedder_attempted
    if _embedder is not None:
        return _embedder
    if _embedder_attempted:
        return None

    _embedder_attempted = True
    try:
        from fastembed import TextEmbedding

        _embedder = TextEmbedding(model_name=DEFAULT_MODEL)
        _LOG.info("RudriQ eval: embedding model loaded (%s)", DEFAULT_MODEL)
        return _embedder
    except ImportError:
        _LOG.warning(
            "RudriQ eval: fastembed not installed. Semantic evaluators will "
            "degrade. Install with: pip install 'rudriq[evaluate]'"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("RudriQ eval: failed to load embedding model: %s", exc)
        return None


def embed_texts(texts: list[str]) -> Optional[list[Any]]:
    """Embed a list of texts. Returns list of vectors, or ``None`` if unavailable."""
    embedder = get_embedder()
    if embedder is None:
        return None
    try:
        # fastembed returns a generator of numpy arrays.
        return list(embedder.embed(texts))
    except Exception as exc:  # noqa: BLE001
        _LOG.warning("RudriQ eval: embedding failed: %s", exc)
        return None


def cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    """Cosine similarity between two vectors. Returns a float in [-1, 1].

    Uses numpy if available; falls back to pure-Python so the helper
    works in environments where ``[evaluate]`` extras are not installed
    (the cosine path is still reachable from tests with stub vectors).
    """
    try:
        import numpy as np

        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float(np.dot(a, b) / denom)
    except ImportError:
        import math

        dot = sum(float(x) * float(y) for x, y in zip(vec_a, vec_b))
        na = math.sqrt(sum(float(x) * float(x) for x in vec_a))
        nb = math.sqrt(sum(float(y) * float(y) for y in vec_b))
        return dot / (na * nb + 1e-9)


def reset_embedder_for_tests() -> None:
    """Reset the cached embedder. For tests only."""
    global _embedder, _embedder_attempted
    _embedder = None
    _embedder_attempted = False
