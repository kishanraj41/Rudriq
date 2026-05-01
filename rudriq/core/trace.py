"""
Data structures representing the linked, cross-domain trace.

A LinkedTrace is a unified view that combines:

* OpenLLMetry/OTel spans (gen_ai.*, traceloop.*)
* AutoLineage records (pandas, scikit-learn, PySpark operations)
* RudriQ link annotations (the cross-domain edges)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TraceDomain(str, Enum):
    """Which subsystem produced a given trace node."""

    DATA_LINEAGE = "data_lineage"  # AutoLineage record
    LLM = "llm"                    # OpenLLMetry/OTel gen_ai span
    LINKED = "linked"              # Has cross-domain attribution


@dataclass
class LinkedTrace:
    """
    A unified view of one operation in the cross-domain trace.

    This is the structure returned by tracker.get_unified_trace(). Every
    node corresponds to one captured operation — could be from
    AutoLineage (e.g., a pandas filter) or from an OTel span (e.g., an
    OpenAI chat completion). Cross-domain edges are stored in
    ``lineage_parents``.

    Attributes
    ----------
    node_id:
        Unique identifier for this node within the trace.
    domain:
        Which subsystem produced this node.
    operation:
        Human-readable operation name (e.g., "DataFrame.filter",
        "openai.chat.completions.create").
    library:
        The library/framework the operation came from (e.g., "pandas",
        "openai", "anthropic").
    started_at, ended_at:
        UTC timestamps.
    metadata:
        Operation-specific metadata (model, tokens, prompt, shape, etc.).
    direct_parents:
        Node ids of operations whose output this operation consumed
        within the same domain.
    lineage_parents:
        Node ids of operations across domains that this operation's
        input is causally derived from. Populated by the linker.
    link_confidence:
        How confident the linker is that ``lineage_parents`` is correct,
        in [0.0, 1.0].
    """

    node_id: str
    domain: TraceDomain
    operation: str
    library: str
    started_at: datetime
    ended_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    direct_parents: list[str] = field(default_factory=list)
    lineage_parents: list[str] = field(default_factory=list)
    link_confidence: float = 0.0

    @property
    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "domain": self.domain.value,
            "operation": self.operation,
            "library": self.library,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
            "direct_parents": list(self.direct_parents),
            "lineage_parents": list(self.lineage_parents),
            "link_confidence": self.link_confidence,
        }