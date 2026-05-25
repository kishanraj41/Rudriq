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

AUDIT_SCHEMA_VERSION = "rudriq.audit/1.2"
# 1.0 → 1.1 (Day 16, additive): two new top-level keys, ``evaluations``
# (list[dict] of EvalResult.to_dict()) and ``evaluation_summary``
# (per-metric traffic-light dict). Both are ``null`` when the writer
# was not asked to include them. A 1.0 consumer can ignore the new
# keys without breakage; the version always honestly reflects what
# the writer is capable of emitting.
# 1.1 → 1.2 (Day 1/9 post-v0.1.0, additive): one new top-level key
# ``root_cause_analysis`` carrying the deviation-weighted RCA output
# for an auto- or user-selected failure target. ``null`` when
# ``--include-rca`` is not passed. Schema rules same as 1.1: a 1.1
# consumer ignores the new key without breakage.

# Default metrics for --include-evals. Drift is omitted by default
# because it needs a baseline graph; the caller can request it
# explicitly via eval_metrics + baseline_graph.
_DEFAULT_AUDIT_EVAL_METRICS = (
    "retrieval_relevance", "groundedness", "coherence", "consistency",
)

# Traffic-light bands for the evaluation summary. Green = healthy,
# yellow = borderline, red = concerning, gray = no scoreable result
# (all SKIPPED or DEGRADED).
_TRAFFIC_GREEN_MIN = 0.7
_TRAFFIC_YELLOW_MIN = 0.4


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
# Evaluation embedding (Day 16)
# ---------------------------------------------------------------------------


def _run_audit_evaluations(
    graph: TraceGraph,
    metrics: list[str] | tuple[str, ...] | None = None,
    baseline_graph: TraceGraph | None = None,
) -> list[dict[str, Any]]:
    """Run evaluators for inclusion in an audit report.

    Returns a deterministically-ordered list of ``EvalResult.to_dict()``
    dicts (sorted by ``(metric, node_id-or-empty-string)``). The default
    metric set is the four content-aware single-trace evaluators; drift
    is only added when explicitly requested AND a baseline graph is
    provided (its constructor needs the graph; the framework's
    ``evaluate(graph)`` signature can't carry the baseline).

    Unknown metric names are silently skipped — callers can pass a
    user-supplied list without sanitization.
    """
    if metrics is None:
        metrics = _DEFAULT_AUDIT_EVAL_METRICS

    from rudriq.evaluate.base import run_evaluators
    from rudriq.evaluate.coherence import CoherenceEvaluator
    from rudriq.evaluate.consistency import ConsistencyEvaluator
    from rudriq.evaluate.groundedness import GroundednessEvaluator
    from rudriq.evaluate.retrieval_relevance import RetrievalRelevanceEvaluator

    registry = {
        "retrieval_relevance": RetrievalRelevanceEvaluator,
        "groundedness": GroundednessEvaluator,
        "coherence": CoherenceEvaluator,
        "consistency": ConsistencyEvaluator,
    }
    evaluators: list[Any] = [registry[m]() for m in metrics if m in registry]

    if "drift" in metrics and baseline_graph is not None:
        from rudriq.evaluate.drift import DriftEvaluator
        evaluators.append(DriftEvaluator(baseline_graph=baseline_graph))

    results = run_evaluators(graph, evaluators)
    result_dicts = [r.to_dict() for r in results]
    # Deterministic ordering: metric first, then node_id (empty-string
    # for trace-level results). The audit JSON's outer ``sort_keys``
    # handles dict-key ordering; this handles list ordering.
    result_dicts.sort(key=lambda d: (d["metric"], d.get("node_id") or ""))
    return result_dicts


def _summarize_evaluations(
    eval_dicts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a per-metric traffic-light summary across eval results.

    For each metric, aggregates how many OK / SKIPPED / DEGRADED / ERROR
    results landed, and the mean score among OK results. Assigns a
    traffic-light status per metric from the mean score:

    * ``green``  — mean ≥ 0.7
    * ``yellow`` — mean ≥ 0.4
    * ``red``    — mean < 0.4
    * ``gray``   — no OK results with a score

    Returns a dict keyed by metric, deterministically ordered (Python
    dicts preserve insertion; we insert in sorted metric order).
    """
    from collections import defaultdict

    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in eval_dicts:
        by_metric[d["metric"]].append(d)

    summary: dict[str, dict[str, Any]] = {}
    for metric in sorted(by_metric):
        items = by_metric[metric]
        ok_scored = [
            i for i in items
            if i["status"] == "ok" and i.get("score") is not None
        ]
        # Day 17 Thread A: distinguish spec-correct "not applicable"
        # SKIPs (the evaluator wasn't meant for this node kind) from
        # the total count so the "Evaluated" ratio is honest.
        not_applicable_count = sum(
            1 for i in items
            if (i.get("details") or {}).get("not_applicable", False)
        )

        status_counts: dict[str, int] = {}
        for i in items:
            status_counts[i["status"]] = status_counts.get(i["status"], 0) + 1

        mean_score: float | None = (
            sum(i["score"] for i in ok_scored) / len(ok_scored)
            if ok_scored else None
        )

        if mean_score is None:
            light = "gray"
        elif mean_score >= _TRAFFIC_GREEN_MIN:
            light = "green"
        elif mean_score >= _TRAFFIC_YELLOW_MIN:
            light = "yellow"
        else:
            light = "red"

        summary[metric] = {
            "mean_score": mean_score,
            "traffic_light": light,
            "status_counts": status_counts,
            "evaluated": len(ok_scored),
            "applicable_total": len(items) - not_applicable_count,
            "not_applicable": not_applicable_count,
            "total": len(items),
        }
    return summary


# ---------------------------------------------------------------------------
# Root-cause-analysis embedding (Day 1/9 post-v0.1.0 — schema 1.2)
# ---------------------------------------------------------------------------


def _auto_select_failure_target(graph: TraceGraph) -> str | None:
    """Pick the most likely failure node to diagnose for an audit-embedded RCA.

    Strategy, in priority order:

    1. Run groundedness against the graph and pick the LLM node with
       the *lowest* score (the most plausibly failing call). Best
       signal because the score is a real quality metric, not a guess.
       If groundedness is degraded (fastembed not installed) or every
       LLM node SKIPs, this branch silently falls through.
    2. Fallback: the lexicographically last LLM node_id — deterministic
       and never raises. Not a quality signal, but ensures
       ``--include-rca`` always produces *something* to look at.

    Returns ``None`` only when the graph has no LLM nodes at all.
    """
    try:
        from rudriq.evaluate.base import EvalStatus
        from rudriq.evaluate.groundedness import GroundednessEvaluator

        results = GroundednessEvaluator().evaluate(graph)
        scored = [
            r for r in results
            if r.status == EvalStatus.OK
            and r.score is not None
            and r.node_id
        ]
        if scored:
            # Sort by (score asc, node_id asc) — lowest groundedness
            # first, ties broken deterministically by node_id.
            scored.sort(key=lambda r: (r.score, r.node_id))
            return scored[0].node_id
    except Exception:  # noqa: BLE001
        # Defensive: if the eval framework fails for any reason, fall
        # through to the lexicographic fallback rather than blowing up
        # the audit-report path.
        pass

    llm_node_ids = sorted(
        n.node_id for n in graph.nodes if n.kind.value.startswith("llm_")
    )
    return llm_node_ids[-1] if llm_node_ids else None


def _run_audit_rca(
    graph: TraceGraph,
    target_node_id: str | None = None,
    baseline_graph: TraceGraph | None = None,
    top: int = 5,
) -> dict[str, Any]:
    """Run deviation-weighted RCA for inclusion in an audit report.

    When ``target_node_id`` is ``None``, auto-selects via
    ``_auto_select_failure_target`` so a user running
    ``rudriq audit --include-rca`` gets a useful diagnosis without
    having to know node_ids.

    Returned shape (stable for schema 1.2):

    * ``target_node_id`` — the diagnosed node (or ``None`` if none
      could be selected, e.g. a graph with no LLM nodes)
    * ``candidates`` — list of ``RootCauseCandidate.to_dict()``, in
      the analyzer's deterministic order (already sorted by
      ``-score``, ``chain_distance``, ``node_id``); capped at ``top``
    * ``note`` — single-sentence reminder that this is a heuristic
      suspect ranking, not a causal proof
    """
    from rudriq.analyzer.deviation_rca import DeviationRCA

    note = (
        "Ranked suspects by deviation heuristic — proximity, path "
        "confidence, and (with baseline) shape deviation. NOT a "
        "causal proof."
    )

    target = target_node_id or _auto_select_failure_target(graph)
    if target is None:
        return {
            "target_node_id": None,
            "candidates": [],
            "note": "No LLM node available to diagnose.",
        }

    rca = DeviationRCA(baseline_graph=baseline_graph)
    candidates = rca.analyze(graph, target)[:top]
    return {
        "target_node_id": target,
        "candidates": [c.to_dict() for c in candidates],
        "note": note,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_audit_report_dict(
    run_id: str,
    *,
    include_evals: bool = False,
    eval_metrics: list[str] | None = None,
    baseline_graph: TraceGraph | None = None,
    include_rca: bool = False,
    rca_target: str | None = None,
) -> dict[str, Any]:
    """Assemble the canonical audit-report dict (schema 1.2).

    This is the single source of truth that both ``export_audit_json``
    and ``export_audit_pdf`` render from — neither re-parses the
    other's output. Adding a new format means writing a new renderer
    over this dict, not re-implementing the assembly. The dict's
    structure mirrors the JSON shape one-to-one so a JSON consumer and
    a PDF consumer see the same fields by the same names.

    Determinism: nodes/edges are sorted; ``evaluations`` ordering and
    RCA candidate ordering inherit from their producers (which sort
    deterministically by construction).

    Raises ValueError if ``run_id`` is not in storage.
    """
    storage = get_default_storage()
    graph = storage.load_run(run_id)
    if graph is None:
        raise ValueError(f"Run not found: {run_id}")

    summary = _compute_summary(graph)
    chains = _compute_lineage_chains(graph)
    nodes_sorted = _sorted_nodes(graph)
    edges_sorted = _sorted_edges(graph)

    evaluations: list[dict[str, Any]] | None = None
    evaluation_summary: dict[str, dict[str, Any]] | None = None
    if include_evals:
        evaluations = _run_audit_evaluations(graph, eval_metrics, baseline_graph)
        evaluation_summary = _summarize_evaluations(evaluations)

    root_cause_analysis: dict[str, Any] | None = None
    if include_rca:
        root_cause_analysis = _run_audit_rca(
            graph, rca_target, baseline_graph,
        )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
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
        # 1.1 additions — always present, ``null`` when not requested.
        "evaluations": evaluations,
        "evaluation_summary": evaluation_summary,
        # 1.2 addition.
        "root_cause_analysis": root_cause_analysis,
    }


def export_audit_json(
    run_id: str,
    *,
    include_evals: bool = False,
    eval_metrics: list[str] | None = None,
    baseline_graph: TraceGraph | None = None,
    include_rca: bool = False,
    rca_target: str | None = None,
) -> str:
    """
    Export the run's audit report as a JSON string.

    Output is deterministic for a given input graph: keys sorted, nodes
    sorted by (started_at, node_id), edges sorted by (parent_id,
    child_id, kind). The same TraceGraph produces byte-identical output
    across calls, processes, and DuckDB instances (apart from the
    ``generated_at`` timestamp at the top level).

    Parameters
    ----------
    include_evals:
        If True, runs the configured evaluators against ``graph`` and
        embeds their output as ``evaluations`` + ``evaluation_summary``.
        Default ``False`` — capture-content + fastembed are opt-in
        dependencies; the audit report stays cheap by default.
    eval_metrics:
        Optional list of metric names. Defaults to the four content-
        aware single-trace evaluators. Pass ``["drift", ...]`` together
        with ``baseline_graph`` to include drift.
    baseline_graph:
        Required iff ``"drift"`` is in ``eval_metrics``. Loaded by the
        CLI from ``--baseline-run-id``. Also reused by RCA when
        ``include_rca`` is True (enables structural-deviation scoring).
    include_rca:
        If True, runs deviation-weighted RCA against the graph and
        embeds it under ``root_cause_analysis``. Default ``False``.
    rca_target:
        Specific node_id to diagnose. When ``None`` (the default), RCA
        auto-selects the LLM node with the lowest groundedness score,
        falling back to the lexicographically last LLM node_id if no
        scoreable result is available.

    Raises ValueError if run_id is not in storage.
    """
    report = build_audit_report_dict(
        run_id,
        include_evals=include_evals,
        eval_metrics=eval_metrics,
        baseline_graph=baseline_graph,
        include_rca=include_rca,
        rca_target=rca_target,
    )
    return json.dumps(report, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_rca_markdown(rca: dict[str, Any]) -> list[str]:
    """Render the ``## Root Cause Analysis`` Markdown section.

    Layout: a one-line "diagnosing node X" header, the honesty note
    (heuristic suspect ranking), and a ranked table with the per-signal
    evidence summarized in a single "why flagged" column. The same
    no-causal-proof framing the analyzer and CLI carry — repeated here
    because the compliance reader of the report is the one most likely
    to over-interpret the ranking.
    """
    lines: list[str] = ["", "## Root Cause Analysis", ""]

    if not rca or not rca.get("target_node_id"):
        lines.append(
            "_No root-cause candidates — no LLM node was available to "
            "diagnose._"
        )
        return lines

    lines.append(f"**Diagnosing:** node `{rca['target_node_id']}`")
    lines.append("")
    lines.append(f"_{rca.get('note', '')}_")
    lines.append("")

    candidates = rca.get("candidates") or []
    if not candidates:
        lines.append(
            "_No upstream operations linked to the target node — "
            "nothing to rank._"
        )
        return lines

    lines.append("| Rank | Operation | Score | Hops upstream | Why flagged |")
    lines.append("|---|---|---|---|---|")
    for i, c in enumerate(candidates, 1):
        ev = c.get("evidence") or {}
        why_parts: list[str] = []
        if "structural_deviation" in ev:
            why_parts.append(f"shape Δ {ev['structural_deviation']:.2f}")
        if "proximity" in ev:
            why_parts.append(f"proximity {ev['proximity']:.2f}")
        if "path_confidence" in ev:
            why_parts.append(f"conf {ev['path_confidence']:.2f}")
        why = ", ".join(why_parts) or "—"
        lines.append(
            f"| {i} | {c['library']}.{c['operation']} | "
            f"{c['score']:.3f} | {c['chain_distance']} | {why} |"
        )

    return lines


def _render_eval_markdown(
    eval_dicts: list[dict[str, Any]],
    eval_summary: dict[str, dict[str, Any]],
) -> list[str]:
    """Render the ``## Quality Evaluation`` Markdown section.

    Layout: a per-metric traffic-light summary table, then a
    "Notable findings" list of non-OK or sub-green results (capped to
    keep the report readable). Emoji traffic lights are a deliberate
    style exception — a colored circle is genuinely clearer than
    ``[GREEN]`` for a human reading a compliance artifact.
    """
    lights = {"green": "🟢", "yellow": "🟡", "red": "🔴", "gray": "⚪"}
    lines: list[str] = ["", "## Quality Evaluation", ""]

    if not eval_summary:
        lines.append(
            "_No evaluation results — evaluators all SKIPPED or fastembed "
            "is not installed._"
        )
        return lines

    lines.append("| Metric | Status | Mean Score | Evaluated | Notes |")
    lines.append("|---|---|---|---|---|")
    for metric in sorted(eval_summary):
        s = eval_summary[metric]
        light = lights.get(s["traffic_light"], "⚪")
        score = (
            f"{s['mean_score']:.2f}"
            if s["mean_score"] is not None else "N/A"
        )
        # "Evaluated" reports against ``applicable_total`` so spec-
        # correct not-applicable SKIPs don't make a fully-working
        # metric look like it failed partially. The not_applicable
        # count is surfaced separately in the Notes column.
        applicable = s.get("applicable_total", s["total"])
        not_app = s.get("not_applicable", 0)
        notes = f"{not_app} not applicable" if not_app else ""
        lines.append(
            f"| {metric} | {light} {s['traffic_light']} | "
            f"{score} | {s['evaluated']}/{applicable} | {notes} |"
        )

    # Filter Notable findings to actual issues: non-OK results and
    # sub-green scores, but suppress entries flagged ``not_applicable``
    # (Day 17 Thread A — e.g. groundedness/coherence on embedding spans:
    # those SKIPs are correct, not a gap, and listing them as
    # "findings" misleads the auditor).
    notable = [
        d for d in eval_dicts
        if (
            d["status"] != "ok"
            or (d.get("score") is not None and d["score"] < _TRAFFIC_GREEN_MIN)
        )
        and not (d.get("details") or {}).get("not_applicable", False)
    ]
    if notable:
        lines.extend(["", "### Notable findings", ""])
        for d in notable[:20]:
            node = f" (node `{d['node_id']}`)" if d.get("node_id") else ""
            score = (
                f"{d['score']:.2f}"
                if d.get("score") is not None else "—"
            )
            lines.append(
                f"- **{d['metric']}** [{d['status']}] {score}{node}: "
                f"{d['explanation']}"
            )
        if len(notable) > 20:
            lines.append(
                f"- _(+{len(notable) - 20} more notable findings omitted "
                f"for brevity; see the JSON export for the full list.)_"
            )

    return lines


def export_audit_markdown(
    run_id: str,
    *,
    include_evals: bool = False,
    eval_metrics: list[str] | None = None,
    baseline_graph: TraceGraph | None = None,
    include_rca: bool = False,
    rca_target: str | None = None,
) -> str:
    """
    Export the run's audit report as Markdown.

    The output is structured for a human reader (compliance officer,
    auditor, ML platform engineer reviewing a failure). Sections are
    ordered from highest-level (summary) to most-detailed (full graph
    appendix).

    Parameters
    ----------
    include_evals:
        If True, runs the configured evaluators and inserts a
        ``## Quality Evaluation`` section between the LLM lineage
        chains and the full-operations appendix. See
        :func:`export_audit_json` for the parameter semantics.
    """
    storage = get_default_storage()
    graph = storage.load_run(run_id)
    if graph is None:
        raise ValueError(f"Run not found: {run_id}")

    summary = _compute_summary(graph)
    chains = _compute_lineage_chains(graph)

    nodes_sorted = _sorted_nodes(graph)
    edges_sorted = _sorted_edges(graph)

    evaluations: list[dict[str, Any]] | None = None
    evaluation_summary: dict[str, dict[str, Any]] | None = None
    if include_evals:
        evaluations = _run_audit_evaluations(graph, eval_metrics, baseline_graph)
        evaluation_summary = _summarize_evaluations(evaluations)

    root_cause_analysis: dict[str, Any] | None = None
    if include_rca:
        root_cause_analysis = _run_audit_rca(
            graph, rca_target, baseline_graph,
        )

    lines: list[str] = []

    # Header — Day 17 Thread A: no export-time line. The report
    # describes a run; the run's own creation timestamp lives in the
    # Summary section below. Including a wall-clock export timestamp
    # would break byte-determinism (two exports of the same run would
    # disagree only on that line).
    lines.append("# RudriQ Audit Report")
    lines.append("")
    lines.append(f"**Run ID:** `{graph.run_id}`  ")
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

    # Quality Evaluation (1.1, optional). Placed between the LLM
    # lineage chains and the full-operations appendix so a reader who
    # already cares about the linked LLM calls sees their quality
    # signal next, before the dense appendix.
    if include_evals and evaluations is not None and evaluation_summary is not None:
        lines.extend(_render_eval_markdown(evaluations, evaluation_summary))
        lines.append("")

    # Root Cause Analysis (1.2, optional). Sits immediately after the
    # Quality Evaluation so the reader's question "which one was bad?"
    # flows naturally into "and what likely caused it?" — both
    # signals next to each other, both before the dense appendix.
    if include_rca and root_cause_analysis is not None:
        lines.extend(_render_rca_markdown(root_cause_analysis))
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
