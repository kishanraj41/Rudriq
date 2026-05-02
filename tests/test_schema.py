"""Tests for the canonical schema and content hashing."""

from __future__ import annotations

from datetime import datetime, timezone

from rudriq.core.schema import (
    EdgeKind,
    NodeKind,
    TraceEdge,
    TraceGraph,
    TraceNode,
    compute_content_hash,
)


def test_content_hash_is_deterministic() -> None:
    a = compute_content_hash({"text": "hello", "n": 5})
    b = compute_content_hash({"n": 5, "text": "hello"})  # different key order
    assert a == b


def test_content_hash_distinguishes_different_values() -> None:
    a = compute_content_hash({"text": "hello"})
    b = compute_content_hash({"text": "world"})
    assert a != b


def test_content_hash_handles_none() -> None:
    assert compute_content_hash(None) is not None


def test_content_hash_handles_lists() -> None:
    a = compute_content_hash([1, 2, 3])
    b = compute_content_hash([1, 2, 3])
    c = compute_content_hash([3, 2, 1])
    assert a == b
    assert a != c


def test_content_hash_type_tags_prevent_string_int_collision() -> None:
    """Bug 3 regression: '1' and 1 must hash differently."""
    assert compute_content_hash("1") != compute_content_hash(1)
    assert compute_content_hash("1.0") != compute_content_hash(1.0)
    assert compute_content_hash("True") != compute_content_hash(True)
    # And bool/int distinction (bool is a subclass of int in Python)
    assert compute_content_hash(True) != compute_content_hash(1)
    assert compute_content_hash(False) != compute_content_hash(0)


def test_content_hash_distinguishes_list_and_tuple() -> None:
    """A list and tuple with same elements should hash differently."""
    assert compute_content_hash([1, 2, 3]) != compute_content_hash((1, 2, 3))


def test_content_hash_distinguishes_empty_collections() -> None:
    """Empty list, tuple, dict, str must all differ."""
    h_list = compute_content_hash([])
    h_tuple = compute_content_hash(())
    h_dict = compute_content_hash({})
    h_str = compute_content_hash("")
    assert len({h_list, h_tuple, h_dict, h_str}) == 4


def test_trace_graph_roundtrips_to_dict() -> None:
    now = datetime.now(timezone.utc)
    g = TraceGraph(run_id="r", created_at=now)
    g.add_node(TraceNode(
        node_id="n1", kind=NodeKind.DATA_READ,
        library="pandas", operation="read_csv", started_at=now,
    ))
    d = g.to_dict()
    assert d["schema_version"] == "rudriq/1.0"
    assert d["node_count"] == 1


def test_get_lineage_parents_filters_edge_kind() -> None:
    now = datetime.now(timezone.utc)
    g = TraceGraph(run_id="r", created_at=now)
    g.add_node(TraceNode(node_id="a", kind=NodeKind.DATA_READ,
                        library="x", operation="o", started_at=now))
    g.add_node(TraceNode(node_id="b", kind=NodeKind.DATA_TRANSFORM,
                        library="x", operation="o", started_at=now))
    g.add_node(TraceNode(node_id="c", kind=NodeKind.LLM_CHAT,
                        library="openai", operation="chat", started_at=now))
    g.add_edge(TraceEdge(parent_id="a", child_id="b", kind=EdgeKind.DIRECT))
    g.add_edge(TraceEdge(parent_id="b", child_id="c", kind=EdgeKind.LINEAGE_LINK))

    lineage = g.get_lineage_parents("c")
    assert len(lineage) == 1
    assert lineage[0].node_id == "b"

    direct_only = g.get_parents("b")
    assert len(direct_only) == 1
    assert direct_only[0].node_id == "a"
