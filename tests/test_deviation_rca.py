"""Tests for the Day 17 Thread C deviation-weighted RCA.

The algorithm ranks suspects, not proofs — tests assert the *ordering*
and *evidence* properties that make the ranking explainable, rather
than insisting on exact magic-number scores.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rudriq.analyzer.deviation_rca import DeviationRCA, RootCauseCandidate
from rudriq.core.schema import (
    EdgeKind,
    LinkMethod,
    NodeKind,
    TraceEdge,
    TraceGraph,
    TraceNode,
)


def _node(nid: str, kind: NodeKind, lib: str, op: str, md: dict | None = None):
    return TraceNode(
        node_id=nid, kind=kind, library=lib, operation=op,
        started_at=datetime.now(timezone.utc), ended_at=None,
        metadata=md or {},
    )


def _edge(parent: str, child: str, kind=EdgeKind.LINEAGE_LINK, conf: float = 1.0):
    return TraceEdge(
        parent_id=parent, child_id=child, kind=kind,
        confidence=conf, link_method=LinkMethod.OBJECT_IDENTITY, metadata={},
    )


def _chain_graph(extra_meta: dict | None = None) -> TraceGraph:
    """read_csv -> filter -> embed -> chat(target)."""
    extra_meta = extra_meta or {}
    nodes = [
        _node(
            "read", NodeKind.DATA_READ, "pandas", "read_csv",
            {"autolineage.shape": [1000, 5], **extra_meta.get("read", {})},
        ),
        _node(
            "filter", NodeKind.DATA_TRANSFORM, "pandas", "filter",
            {"autolineage.shape": [250, 5], **extra_meta.get("filter", {})},
        ),
        _node(
            "embed", NodeKind.LLM_EMBEDDING, "openai", "embeddings",
            extra_meta.get("embed", {}),
        ),
        _node(
            "chat", NodeKind.LLM_CHAT, "openai", "chat",
            extra_meta.get("chat", {}),
        ),
    ]
    edges = [
        _edge("read", "filter", kind=EdgeKind.DIRECT),
        _edge("filter", "embed"),
        _edge("embed", "chat"),
    ]
    return TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc),
        metadata={}, nodes=nodes, edges=edges,
    )


# ---------------------------------------------------------------------------
# Basic shape: empty / unknown / candidate dataclass
# ---------------------------------------------------------------------------


def test_returns_empty_for_unknown_target():
    g = _chain_graph()
    assert DeviationRCA().analyze(g, "nonexistent") == []


def test_candidate_to_dict_round_trips_evidence():
    c = RootCauseCandidate(
        node_id="x", library="pandas", operation="filter",
        score=0.42, chain_distance=2, evidence={"proximity": 0.33},
    )
    d = c.to_dict()
    assert d["node_id"] == "x"
    assert d["score"] == 0.42
    assert d["evidence"]["proximity"] == 0.33


# ---------------------------------------------------------------------------
# Upstream walk shape
# ---------------------------------------------------------------------------


def test_ranks_all_upstream_operations():
    """Walk reaches every ancestor of the target, regardless of edge kind."""
    g = _chain_graph()
    candidates = DeviationRCA().analyze(g, "chat")
    node_ids = {c.node_id for c in candidates}
    assert node_ids == {"embed", "filter", "read"}


def test_excludes_the_target_itself():
    g = _chain_graph()
    candidates = DeviationRCA().analyze(g, "chat")
    assert all(c.node_id != "chat" for c in candidates)


def test_chain_distance_is_monotonic_with_hops():
    g = _chain_graph()
    candidates = DeviationRCA().analyze(g, "chat")
    by_id = {c.node_id: c for c in candidates}
    assert by_id["embed"].chain_distance == 1
    assert by_id["filter"].chain_distance == 2
    assert by_id["read"].chain_distance == 3


# ---------------------------------------------------------------------------
# Ordering: without baseline, proximity dominates
# ---------------------------------------------------------------------------


def test_closer_operations_rank_higher_without_baseline():
    g = _chain_graph()
    candidates = DeviationRCA().analyze(g, "chat")
    # embed (1 hop) > filter (2 hop) > read (3 hop)
    by_id = {c.node_id: c for c in candidates}
    assert by_id["embed"].score > by_id["filter"].score > by_id["read"].score


def test_results_sorted_descending_by_score():
    g = _chain_graph()
    candidates = DeviationRCA().analyze(g, "chat")
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Structural deviation against a baseline boosts the score of the
# operation whose shape changed.
# ---------------------------------------------------------------------------


def test_structural_deviation_boosts_score_with_baseline():
    baseline = _chain_graph()  # filter outputs 250 rows
    current = _chain_graph()
    # Mutate current's filter to 10 rows — sharp deviation vs baseline 250.
    for n in current.nodes:
        if n.node_id == "filter":
            n.metadata["autolineage.shape"] = [10, 5]

    candidates = DeviationRCA(baseline_graph=baseline).analyze(current, "chat")
    by_id = {c.node_id: c for c in candidates}

    # rows 250→10 averaged with unchanged cols gives ~0.48 — substantial.
    assert by_id["filter"].evidence["structural_deviation"] > 0.4
    # And the load-bearing claim: filter is now the top suspect even
    # though it's 2 hops from the failing chat node. Without baseline
    # this would never happen (embed at 1 hop would dominate).
    assert candidates[0].node_id == "filter"


def test_no_baseline_means_no_structural_evidence_entry():
    g = _chain_graph()
    candidates = DeviationRCA().analyze(g, "chat")
    for c in candidates:
        assert "structural_deviation" not in c.evidence


# ---------------------------------------------------------------------------
# Path confidence: weak links reduce score downstream
# ---------------------------------------------------------------------------


def test_weak_link_confidence_lowers_score():
    """A 0.5 link should produce path_confidence=0.5 in the closer parent."""
    nodes = [
        _node("weak_parent", NodeKind.DATA_TRANSFORM, "pandas", "merge"),
        _node("chat", NodeKind.LLM_CHAT, "openai", "chat"),
    ]
    edges = [_edge("weak_parent", "chat", conf=0.5)]
    g = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc),
        metadata={}, nodes=nodes, edges=edges,
    )
    candidates = DeviationRCA().analyze(g, "chat")
    assert len(candidates) == 1
    assert candidates[0].evidence["path_confidence"] == 0.5


def test_path_confidence_is_cumulative_product():
    """A chain of two 0.5 edges → path_confidence 0.25 at the deeper node."""
    nodes = [
        _node("gp", NodeKind.DATA_READ, "pandas", "read_csv"),
        _node("p", NodeKind.DATA_TRANSFORM, "pandas", "filter"),
        _node("chat", NodeKind.LLM_CHAT, "openai", "chat"),
    ]
    edges = [
        _edge("gp", "p", conf=0.5),
        _edge("p", "chat", conf=0.5),
    ]
    g = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc),
        metadata={}, nodes=nodes, edges=edges,
    )
    by_id = {
        c.node_id: c for c in DeviationRCA().analyze(g, "chat")
    }
    assert by_id["p"].evidence["path_confidence"] == 0.5
    assert by_id["gp"].evidence["path_confidence"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Determinism: same input → same ranking, bit-for-bit
# ---------------------------------------------------------------------------


def test_ranking_is_deterministic():
    """Same graph + baseline → same list across calls, including evidence."""
    baseline = _chain_graph()
    current = _chain_graph()
    a = DeviationRCA(baseline_graph=baseline).analyze(current, "chat")
    b = DeviationRCA(baseline_graph=baseline).analyze(current, "chat")
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_self_loop_edges_are_ignored():
    """Autolineage can emit edges with parent_id == child_id; the walk
    must skip them rather than emit duplicate candidates.

    Discovered during Day 17 Thread C live validation against the
    realistic pipeline: an ``autolineage.select`` record came back as
    4 stacked candidates with identical (node_id, score) because the
    same node had 3 self-loop edges into itself.
    """
    nodes = [
        _node("loopy", NodeKind.DATA_TRANSFORM, "pandas", "select"),
        _node("chat", NodeKind.LLM_CHAT, "openai", "chat"),
    ]
    edges = [
        _edge("loopy", "chat"),
        _edge("loopy", "loopy"),  # self-loop
        _edge("loopy", "loopy"),  # another self-loop
    ]
    g = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc),
        metadata={}, nodes=nodes, edges=edges,
    )
    candidates = DeviationRCA().analyze(g, "chat")
    # Exactly ONE candidate for ``loopy`` — self-loops don't multiply.
    assert len(candidates) == 1
    assert candidates[0].node_id == "loopy"


def test_parallel_edges_collapse_to_one_candidate_with_best_score():
    """Two edges from the same parent to the same child collapse to one
    candidate with the highest score across the parallel paths."""
    nodes = [
        _node("p", NodeKind.DATA_TRANSFORM, "pandas", "merge"),
        _node("chat", NodeKind.LLM_CHAT, "openai", "chat"),
    ]
    edges = [
        _edge("p", "chat", conf=0.5),  # weak parallel link
        _edge("p", "chat", conf=1.0),  # strong parallel link
    ]
    g = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc),
        metadata={}, nodes=nodes, edges=edges,
    )
    candidates = DeviationRCA().analyze(g, "chat")
    assert len(candidates) == 1
    # The stronger link wins: path_confidence reflects the best, not
    # the weakest, traversal.
    assert candidates[0].evidence["path_confidence"] == 1.0


def test_ties_broken_by_chain_distance_then_node_id():
    """Two siblings at the same distance and confidence break the tie
    deterministically by node_id."""
    nodes = [
        _node("aaa", NodeKind.DATA_TRANSFORM, "pandas", "op"),
        _node("bbb", NodeKind.DATA_TRANSFORM, "pandas", "op"),
        _node("chat", NodeKind.LLM_CHAT, "openai", "chat"),
    ]
    edges = [_edge("aaa", "chat"), _edge("bbb", "chat")]
    g = TraceGraph(
        run_id="r", created_at=datetime.now(timezone.utc),
        metadata={}, nodes=nodes, edges=edges,
    )
    candidates = DeviationRCA().analyze(g, "chat")
    assert [c.node_id for c in candidates] == ["aaa", "bbb"]
