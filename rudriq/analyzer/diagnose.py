"""
Cross-domain root-cause localization.

When an LLM response changes between runs, this analyzer walks the
unified trace to identify which upstream operation most likely caused
the change. Adapts AutoLineage's deviation-weighted scoring algorithm
to operate across the data-lineage / LLM seam.

In v0.0.1, ``diagnose`` returns a structural summary. The full scoring
algorithm lands in v0.2 (Sprint Week 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rudriq.core.tracker import get_tracker


@dataclass
class RootCause:
    """A single ranked candidate for the cause of an LLM behavior change."""

    node_id: str
    operation: str
    domain: str
    impact_score: float  # in [0.0, 1.0]
    evidence: dict[str, Any] = field(default_factory=dict)


def diagnose(
    target_metric: str | None = None,
    baseline_path: str | None = None,
) -> dict[str, Any]:
    """
    Analyze the current unified trace and rank root-cause candidates.

    Parameters
    ----------
    target_metric:
        For LLM applications: the response or quality metric whose
        change is being investigated (e.g., "response_similarity",
        "answer_quality"). When None, returns a structural summary
        only.
    baseline_path:
        Path to a saved baseline fingerprint to compare against. When
        None, the analyzer falls back to single-run pathology
        detection.

    Returns
    -------
    A dictionary with keys:
        - summary: trace structural summary
        - root_causes: list[RootCause] ranked by impact_score
        - notes: list[str] of explanatory messages

    v0.0.1: returns structural summary only. v0.2 implements full
    deviation-weighted scoring.
    """
    tracker = get_tracker()
    graph = tracker.get_full_graph()

    return {
        "summary": {
            "node_count": graph["node_count"],
            "version": graph["version"],
            "target_metric": target_metric,
            "has_baseline": baseline_path is not None,
        },
        "root_causes": [],
        "notes": [
            "RudriQ v0.0.1: diagnose() returns a structural summary only. "
            "Cross-domain root-cause scoring lands in v0.2.",
        ],
    }