"""Core data structures and tracker for RudriQ."""

from rudriq.core.schema import (
    TraceNode,
    TraceEdge,
    TraceGraph,
    NodeKind,
    EdgeKind,
    LinkMethod,
    compute_content_hash,
)
from rudriq.core.tracker import get_tracker, UnifiedTracker
from rudriq.core.span_attributes import (
    RUDRIQ_LINEAGE_PARENT,
    RUDRIQ_DOMAIN,
    RUDRIQ_LINK_METHOD,
    RUDRIQ_LINK_CONFIDENCE,
)

__all__ = [
    "get_tracker",
    "UnifiedTracker",
    "TraceNode",
    "TraceEdge",
    "TraceGraph",
    "NodeKind",
    "EdgeKind",
    "LinkMethod",
    "compute_content_hash",
    "RUDRIQ_LINEAGE_PARENT",
    "RUDRIQ_DOMAIN",
    "RUDRIQ_LINK_METHOD",
    "RUDRIQ_LINK_CONFIDENCE",
]
