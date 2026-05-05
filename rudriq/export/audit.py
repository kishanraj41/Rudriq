"""
Audit report generation from the unified trace.

The exporter produces two formats from the same TraceGraph:

* JSON: machine-readable, schema-versioned, deterministic. Designed
  to be diffed across runs and consumed by downstream compliance
  tooling without parsing tricks. Nodes and edges are sorted by
  natural key so output is stable across DuckDB instances and
  process invocations, not just within a single load_run call.
* Markdown: human-readable, designed to be read by a compliance
  officer or auditor. Sections are ordered for the question they
  ask first ("what happened, in summary?") to last ("what's the
  full trace?").

Both formats are derived from the same TraceGraph in DuckDB. Calling
either with the same input produces byte-identical output (apart from
the generated_at timestamp), so audit artifacts are reproducible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from rudriq.core.schema import (
    EdgeKind,
    NodeKind,
    TraceGraph,
    TraceNode,
)
from rudriq.storage import get_default_storage

AUDIT_SCHEMA_VERSION = "rudriq.audit/1.0"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


_DATA_KINDS = {
    NodeKind.DATA_READ,
    NodeKind.DATA_TRANSFORM,
    NodeKind.DATA_WRITE,
    NodeKind.MODEL_TRAIN,
    NodeKind.MODEL_PREDICT,
    NodeKind.METRIC_EVAL,
}

_LLM_KINDS = {
    NodeKind.LLM_CHAT,
    NodeKind.LLM_EMBEDDING,
    NodeKind.LLM_COMPLETION,
}


# ---------------------------------------------------------------------------
# Determinism helpers
# ---------------------------------------------------------------------------


def _sorted_nodes(graph: TraceGraph) -> list[TraceNode]:
    """Sort nodes by (started_at, node_id) for stable cross-instance output."""
    return sorted(graph.nodes, key=lambda n: (n.started_at, n.node_id))


def _sorted_edges(graph: TraceGraph) -> list[Any]:
    """Sort edges by (parent_id, child_id, kind) for stable cross-instance output."""
    return sorted(
        graph.edges,
        key=lambda e: (e.parent_id, e.child_id, e.kind.value),
    )


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _compute_summary(graph: TraceGraph) -> dict[str, Any]:
    """Compute the summary block from a TraceGraph."""
    if not graph.nodes:
        return {
            "run_id": graph.run_id,
            "started_at": graph.created_at.isoformat(),
            "ended_at": graph.created_at.isoformat(),
            "duration_ms": 0.0,
            "total_nodes": 0,
            "data_nodes": 0,
            "llm_nodes": 0,
            "other_nodes": 0,
            "total_edges": 0,
            "direct_edges": 0,
            "lineage_links": 0,
            "causal_edges": 0,
            "linked_llm_calls": 0,
            "unlinked_llm_calls": 0,
            "libraries_seen": [],
        }

    started_at = min(n.started_at for n in graph.nodes)
    ended_at = max(
        (n.ended_at for n in graph.nodes if n.ended_at is not None),
        default=started_at,
    )
    duration_ms = (ended_at - started_at).total_seconds() * 1000.0

    data_count = sum(1 for n in graph.nodes if n.kind in _DATA_KINDS)
    llm_count = sum(1 for n in graph.nodes if n.kind in _LLM_KINDS)
    other_count = len(graph.nodes) - data_count - llm_count

    direct_count = sum(1 for e in graph.edges if e.kind == EdgeKind.DIRECT)
    lineage_count = sum(1 for e in graph.edges if e.kind == EdgeKind.LINEAGE_LINK)
    causal_count = sum(1 for e in graph.edges if e.kind == EdgeKind.CAUSAL)

    llm_node_ids = {n.node_id for n in graph.nodes if n.kind in _LLM_KINDS}
    linked_llm_ids = {
        e.child_id
        for e in graph.edges
        if e.kind == EdgeKind.LINEAGE_LINK and e.child_id in llm_node_ids
    }
    unlinked_count = len(llm_node_ids) - len(linked_llm_ids)

    libraries = sorted({n.library for n in graph.nodes})

    return {
        "run_id": graph.run_id,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_ms": duration_ms,
        "total_nodes": len(graph.nodes),
        "data_nodes": data_count,
        "llm_nodes": llm_count,
        "other_nodes": other_count,
        "total_edges": len(graph.edges),
        "direct_edges": direct_count,
        "lineage_links": lineage_count,
        "causal_edges": causal_count,
        "linked_llm_calls": len(linked_llm_ids),
        "unlinked_llm_calls": unlinked_count,
        "libraries_seen": libraries,
    }


# ---------------------------------------------------------------------------
# Lineage chain computation
# ---------------------------------------------------------------------------


def _compute_lineage_chains(graph: TraceGraph) -> list[dict[str, Any]]:
    """
    For each LLM call in the graph, compute its full upstream lineage chain.

    The chain is the transitive closure of parents: starting at the LLM
    call, walk every parent edge (DIRECT and LINEAGE_LINK alike) to find
    every operation that contributed to this LLM call's input.

    Returns one entry per LLM call, ordered by (started_at, node_id) so
    output is stable across DuckDB instances.
    """
    parents_of: dict[str, list[dict[str, Any]]] = {}
    for edge in _sorted_edges(graph):
        parents_of.setdefault(edge.child_id, []).append({
            "parent_id": edge.parent_id,
            "edge_kind": edge.kind.value,
            "link_method": edge.link_method.value,
            "confidence": edge.confidence,
        })

    nodes_by_id = {n.node_id: n for n in graph.nodes}

    def _walk_chain(start_id: str) -> list[dict[str, Any]]:
        """Breadth-first walk from start_id back through parents."""
        visited: set[str] = set()
        chain: list[dict[str, Any]] = []
        frontier: list[tuple[str, int]] = [(start_id, 0)]

        while frontier:
            node_id, depth = frontier.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)

            node = nodes_by_id.get(node_id)
            if node is None:
                # In v0.0.6+ AutoLineage records are mirrored into
                # RudriQ DuckDB, so we shouldn't normally hit this
                # path. If we do, the record exists in AutoLineage's
                # tracker but wasn't mirrored — likely because the
                # SpanProcessor wired its run_id AFTER the relevant
                # AutoLineage record fired (callback ordering edge
                # case), or autolineage<0.6 is installed.
                chain.append({
                    "node_id": node_id,
                    "depth": depth,
                    "kind": "unknown",
                    "library": "external",
                    "operation": "unmirrored",
                    "started_at": None,
                    "metadata_keys": [],
                    "_note": (
                        "Record exists in AutoLineage but was not "
                        "mirrored into RudriQ. Check that "
                        "RudriQSpanProcessor was constructed before "
                        "AutoLineage-tracked operations fired, and "
                        "that autolineage>=0.6 is installed."
                    ),
                })
                continue

            chain.append({
                "node_id": node_id,
                "depth": depth,
                "kind": node.kind.value,
                "library": node.library,
                "operation": node.operation,
                "started_at": node.started_at.isoformat(),
                "metadata_keys": sorted(node.metadata.keys()),
            })

            for parent_info in parents_of.get(node_id, []):
                if parent_info["parent_id"] not in visited:
                    frontier.append((parent_info["parent_id"], depth + 1))

        return chain

    llm_nodes = sorted(
        (n for n in graph.nodes if n.kind in _LLM_KINDS),
        key=lambda n: (n.started_at, n.node_id),
    )

    chains = []
    for llm in llm_nodes:
        full_chain = _walk_chain(llm.node_id)
        upstream = [step for step in full_chain if step["depth"] > 0]
        chains.append({
            "llm_node_id": llm.node_id,
            "llm_operation": llm.operation,
            "llm_library": llm.library,
            "llm_started_at": llm.started_at.isoformat(),
            "chain_length": len(upstream),
            "chain": upstream,
        })

    return chains


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_audit_json(run_id: str) -> str:
    """
    Export the run's audit report as a JSON string.

    Output is deterministic for a given input graph: keys sorted, nodes
    sorted by (started_at, node_id), edges sorted by (parent_id,
    child_id, kind). The same TraceGraph produces byte-identical output
    across calls, processes, and DuckDB instances (apart from the
    ``generated_at`` timestamp at the top level).

    Raises ValueError if run_id is not in storage.
    """
    storage = get_default_storage()
    graph = storage.load_run(run_id)
    if graph is None:
        raise ValueError(f"Run not found: {run_id}")

    summary = _compute_summary(graph)
    chains = _compute_lineage_chains(graph)

    nodes_sorted = _sorted_nodes(graph)
    edges_sorted = _sorted_edges(graph)

    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": _now_utc_iso(),
        "summary": summary,
        "run": {
            "run_id": graph.run_id,
            "created_at": graph.created_at.isoformat(),
            "metadata": graph.metadata,
            "node_count": len(nodes_sorted),
            "edge_count": len(edges_sorted),
            "nodes": [n.to_dict() for n in nodes_sorted],
            "edges": [e.to_dict() for e in edges_sorted],
        },
        "lineage_chains": chains,
    }

    return json.dumps(report, indent=2, sort_keys=True, default=str)


def _now_utc_iso() -> str:
    """Wrapped for monkey-patching in determinism tests."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def export_audit_markdown(run_id: str) -> str:
    """
    Export the run's audit report as Markdown.

    The output is structured for a human reader (compliance officer,
    auditor, ML platform engineer reviewing a failure). Sections are
    ordered from highest-level (summary) to most-detailed (full graph
    appendix).
    """
    storage = get_default_storage()
    graph = storage.load_run(run_id)
    if graph is None:
        raise ValueError(f"Run not found: {run_id}")

    summary = _compute_summary(graph)
    chains = _compute_lineage_chains(graph)

    nodes_sorted = _sorted_nodes(graph)
    edges_sorted = _sorted_edges(graph)

    lines: list[str] = []

    # Header
    lines.append("# RudriQ Audit Report")
    lines.append("")
    lines.append(f"**Run ID:** `{graph.run_id}`  ")
    lines.append(f"**Generated at:** {_now_utc_iso()}  ")
    lines.append(f"**Schema version:** `{AUDIT_SCHEMA_VERSION}`")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Started:** {summary['started_at']}")
    lines.append(f"- **Ended:** {summary['ended_at']}")
    lines.append(f"- **Duration:** {summary['duration_ms']:.1f} ms")
    lines.append(f"- **Total operations:** {summary['total_nodes']}")
    lines.append(f"  - Data operations: {summary['data_nodes']}")
    lines.append(f"  - LLM operations: {summary['llm_nodes']}")
    lines.append(f"  - Other: {summary['other_nodes']}")
    lines.append(f"- **Total edges:** {summary['total_edges']}")
    lines.append(f"  - Direct (intra-domain): {summary['direct_edges']}")
    lines.append(f"  - Lineage (cross-domain): {summary['lineage_links']}")
    lines.append(f"  - Causal: {summary['causal_edges']}")
    lines.append("- **LLM call attribution:**")
    lines.append(f"  - Linked to upstream data: {summary['linked_llm_calls']}")
    lines.append(f"  - Unlinked: {summary['unlinked_llm_calls']}")
    libs = ", ".join(summary["libraries_seen"]) or "none"
    lines.append(f"- **Libraries observed:** {libs}")
    lines.append("")

    # LLM Lineage Chains
    lines.append("## LLM Lineage Chains")
    lines.append("")
    if not chains:
        lines.append("_No LLM operations recorded in this run._")
        lines.append("")
    else:
        lines.append(
            "Each LLM call is shown with the full upstream chain of "
            "operations that contributed to its input."
        )
        lines.append("")
        for chain in chains:
            lines.append(f"### {chain['llm_library']}.{chain['llm_operation']}")
            lines.append("")
            lines.append(f"- **Node ID:** `{chain['llm_node_id']}`")
            lines.append(f"- **Started at:** {chain['llm_started_at']}")
            lines.append(f"- **Upstream chain length:** {chain['chain_length']}")
            lines.append("")
            if chain["chain"]:
                lines.append("**Upstream operations (most recent first):**")
                lines.append("")
                for step in chain["chain"]:
                    indent = "  " * (step["depth"] - 1)
                    started = step["started_at"] or "n/a"
                    lines.append(
                        f"{indent}- depth {step['depth']}: "
                        f"`{step['library']}.{step['operation']}` "
                        f"({step['kind']}, started {started})"
                    )
            else:
                lines.append("_No upstream operations linked to this LLM call._")
            lines.append("")

    # Full Operations Appendix
    lines.append("## Full Operations Appendix")
    lines.append("")

    if nodes_sorted:
        lines.append("### All operations")
        lines.append("")
        lines.append("| Node ID | Kind | Library | Operation | Started At |")
        lines.append("|---|---|---|---|---|")
        for n in nodes_sorted:
            lines.append(
                f"| `{n.node_id}` | {n.kind.value} | {n.library} | "
                f"{n.operation} | {n.started_at.isoformat()} |"
            )
        lines.append("")

    if edges_sorted:
        lines.append("### All edges")
        lines.append("")
        lines.append("| Parent | Child | Kind | Method | Confidence |")
        lines.append("|---|---|---|---|---|")
        for e in edges_sorted:
            lines.append(
                f"| `{e.parent_id}` | `{e.child_id}` | {e.kind.value} | "
                f"{e.link_method.value} | {e.confidence:.2f} |"
            )
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        f"_Generated by RudriQ. Schema version: `{AUDIT_SCHEMA_VERSION}`_"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Legacy API (back-compat for v0.0.x callers)
# ---------------------------------------------------------------------------


def generate_audit_report(
    run_id: str | None = None,
    template: str = "default",
    output_format: str = "json",
) -> dict[str, Any] | str:
    """
    Legacy API. Prefer ``export_audit_json`` and ``export_audit_markdown``.

    For v0.0.5 this returns the new audit report as a dict (when
    ``output_format='json'``) or string (when ``output_format='markdown'``).
    Callers from v0.0.x that depended on the earlier shape with
    ``report_version``/``data_operations``/``llm_operations`` keys must
    migrate — see CHANGELOG.md.

    If ``run_id`` is omitted, the most recent run from local DuckDB is
    selected. Raises ValueError if storage is empty AND no run_id given.
    """
    if template != "default":
        raise NotImplementedError(
            f"Template '{template}' not supported. User-supplied "
            "templates land in v0.3."
        )

    if run_id is None:
        storage = get_default_storage()
        runs = storage.list_runs(limit=1)
        if not runs:
            # Stable empty-result shape so callers can detect "no runs".
            return {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "summary": _compute_summary(_empty_graph()),
                "run": None,
                "lineage_chains": [],
                "notes": ["No runs found in storage."],
            }
        run_id = runs[0]

    if output_format == "json":
        return json.loads(export_audit_json(run_id))
    elif output_format == "markdown":
        return export_audit_markdown(run_id)
    elif output_format == "pdf":
        raise NotImplementedError(
            "PDF rendering lands in v0.3. Use 'markdown' and convert "
            "with pandoc or similar."
        )
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


def _empty_graph() -> TraceGraph:
    return TraceGraph(run_id="<empty>", created_at=datetime.now(timezone.utc))
