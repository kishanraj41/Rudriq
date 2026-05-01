"""
The unified cross-domain tracker singleton.

UnifiedTracker is *not* an instrumentation library. It does not hook
anything itself. Its job is to:

1. Pull AutoLineage records from the AutoLineage tracker (when present).
2. Pull OTel/OpenLLMetry spans from a configured exporter (when present).
3. Apply the linker to produce a unified LinkedTrace graph.

In v0.0.1 this is a stub: full implementation lands in v0.1.
"""

from __future__ import annotations

import threading
from typing import Any

from rudriq.core.trace import LinkedTrace


class UnifiedTracker:
    """
    Read-only view over the cross-domain unified trace.

    Implementation note
    -------------------
    Unlike AutoLineage's UnifiedTracker (which actively records
    operations), this tracker is *passive*. It assembles its view by
    pulling from upstream trackers and exporters at query time.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def get_unified_trace(self) -> list[LinkedTrace]:
        """
        Return the current unified cross-domain trace as a list of
        LinkedTrace nodes.

        v0.0.1: returns empty list. v0.1 pulls from AutoLineage +
        OTel collector.
        """
        with self._lock:
            return []

    def get_full_graph(self) -> dict[str, Any]:
        """JSON-serializable representation of the unified trace."""
        nodes = self.get_unified_trace()
        return {
            "version": "0.0.1",
            "node_count": len(nodes),
            "nodes": [n.to_dict() for n in nodes],
        }


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_tracker: UnifiedTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> UnifiedTracker:
    """Return the process-local UnifiedTracker singleton."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = UnifiedTracker()
    return _tracker