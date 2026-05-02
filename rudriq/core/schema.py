"""
RudriQ canonical trace schema.

This is the canonical data model for RudriQ. It is intentionally NOT
OpenTelemetry. OTel is used for ingestion (we accept OTel spans from
OpenLLMetry) and export (we can emit OTel spans for the customer's
existing observability stack), but our internal model is richer:

* It has first-class causal edges between nodes.
* It supports cross-domain links (data lineage <-> LLM call) as a
  primary concept rather than as ad-hoc span attributes.
* It carries evaluation results and audit metadata that don't fit in
  OTel's span model.

The trade is: more engineering complexity now, real defensibility later.
Pure OTel-based competitors are constrained by what OTel can express;
we are not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NodeKind(str, Enum):
    """The kind of operation this node represents."""

    DATA_READ = "data_read"
    DATA_TRANSFORM = "data_transform"
    DATA_WRITE = "data_write"
    MODEL_TRAIN = "model_train"
    MODEL_PREDICT = "model_predict"
    METRIC_EVAL = "metric_eval"
    LLM_CHAT = "llm_chat"
    LLM_EMBEDDING = "llm_embedding"
    LLM_COMPLETION = "llm_completion"
    RETRIEVAL = "retrieval"
    PROMPT_ASSEMBLY = "prompt_assembly"
    UNKNOWN = "unknown"


class EdgeKind(str, Enum):
    """How two nodes are related."""

    DIRECT = "direct"          # output of A is input of B, same domain
    LINEAGE_LINK = "lineage"   # cross-domain (A's data flowed into B's input)
    CAUSAL = "causal"          # A is a learned causal parent of B
    SIBLING = "sibling"        # A and B are independent ops sharing a parent


class LinkMethod(str, Enum):
    """How the linker established a cross-domain link."""

    OBJECT_IDENTITY = "object_identity"
    CONTENT_HASH = "content_hash"
    NAME_MATCH = "name_match"
    UNKNOWN = "unknown"


@dataclass
class TraceNode:
    """
    A single operation in the RudriQ trace graph.

    A node corresponds to one captured operation, regardless of which
    subsystem (AutoLineage, OpenLLMetry, custom hook) emitted it.
    """

    node_id: str
    kind: NodeKind
    library: str
    operation: str
    started_at: datetime
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str | None = None  # SHA-256 of output, when available

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "library": self.library,
            "operation": self.operation,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }


@dataclass
class TraceEdge:
    """
    A directed edge between two trace nodes.
    """

    parent_id: str
    child_id: str
    kind: EdgeKind
    confidence: float = 1.0  # in [0.0, 1.0]
    link_method: LinkMethod = LinkMethod.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "kind": self.kind.value,
            "confidence": self.confidence,
            "link_method": self.link_method.value,
            "metadata": dict(self.metadata),
        }


@dataclass
class TraceGraph:
    """
    The full unified trace.

    A TraceGraph is the canonical representation of one production
    run (or one debug session). It can be serialized to JSON, queried
    by the analyzer, exported to OTel, or persisted to DuckDB.
    """

    run_id: str
    created_at: datetime
    nodes: list[TraceNode] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: TraceNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: TraceEdge) -> None:
        self.edges.append(edge)

    def get_node(self, node_id: str) -> TraceNode | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_parents(self, node_id: str) -> list[TraceNode]:
        parent_ids = {e.parent_id for e in self.edges if e.child_id == node_id}
        return [n for n in self.nodes if n.node_id in parent_ids]

    def get_lineage_parents(self, node_id: str) -> list[TraceNode]:
        """Cross-domain parents only."""
        parent_ids = {
            e.parent_id
            for e in self.edges
            if e.child_id == node_id and e.kind == EdgeKind.LINEAGE_LINK
        }
        return [n for n in self.nodes if n.node_id in parent_ids]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rudriq/1.0",
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Content hashing utilities
# ---------------------------------------------------------------------------


def compute_content_hash(value: Any) -> str | None:
    """
    Compute a SHA-256 hash of ``value`` if it is feasibly hashable.

    Returns None for objects whose serialization is unreasonably
    expensive (very large DataFrames, etc.) or who cannot be reduced
    to a stable byte representation. Callers should treat None as
    "no hash available" — link_by_content_hash will simply skip such
    objects.
    """
    try:
        if value is None:
            return hashlib.sha256(b"<None>").hexdigest()
        if isinstance(value, (str, int, float, bool)):
            return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        if isinstance(value, bytes):
            return hashlib.sha256(value).hexdigest()
        if isinstance(value, (list, tuple)):
            # Hash the concatenation of element hashes.
            h = hashlib.sha256()
            for item in value:
                item_hash = compute_content_hash(item)
                if item_hash is None:
                    return None
                h.update(item_hash.encode("utf-8"))
            return h.hexdigest()
        if isinstance(value, dict):
            h = hashlib.sha256()
            for k in sorted(value.keys()):
                k_hash = compute_content_hash(k)
                v_hash = compute_content_hash(value[k])
                if k_hash is None or v_hash is None:
                    return None
                h.update(k_hash.encode("utf-8"))
                h.update(v_hash.encode("utf-8"))
            return h.hexdigest()
        # For pandas DataFrames, numpy arrays, etc., delegate to a
        # repr-based hash. This is deliberately approximate: it's
        # stable for typical analytical workloads but not bytewise.
        repr_bytes = repr(value).encode("utf-8")
        if len(repr_bytes) > 5_000_000:  # 5MB cap
            return None
        return hashlib.sha256(repr_bytes).hexdigest()
    except Exception:
        return None
