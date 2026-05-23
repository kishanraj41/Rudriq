"""Deviation-weighted root-cause analysis (fast path).

Given a target node with a quality problem, ranks its upstream
operations by how anomalous they look. **This is a heuristic ranking
of suspects, not a causal proof.** Full causal inference
(do-calculus, gradient-based DAG learning, counterfactual analysis)
is explicitly a v1.0+ research track; nothing here resolves causation,
only correlation and structural deviation.

Why honest framing matters here: for an audit-grade tool, claiming
"this caused the failure" when you mean "this is the most anomalous
upstream operation" is exactly the kind of overclaim that loses
trust. The analyzer's output, CLI rendering, and docstrings all say
"suspect ranking" rather than "root cause," and the explanation
surfaces the evidence so a human can judge.

Scoring per upstream operation blends:

* **Structural deviation vs baseline** (output shape change — rows /
  columns). Reuses AutoLineage's signal. Only available when a
  baseline graph is provided. Range [0, 1].
* **Proximity to the failing node** along the lineage chain.
  Operations closer to the failure are more likely proximate causes.
  Weighted as ``1 / (1 + distance)``.
* **Path link confidence** along the chain from the target to this
  operation. A weak link (substring match at 0.5) carries less
  causal weight than a strong link (object-identity at 1.0). The
  cumulative product of edge confidences along the path.

Blend:

* With baseline:    ``0.5 * structural + 0.3 * proximity + 0.2 * path_conf``
* Without baseline: ``0.6 * proximity + 0.4 * path_conf``

The list is sorted by score descending (stable secondary keys:
chain_distance ascending, then node_id ascending).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rudriq.core.schema import EdgeKind, TraceGraph, TraceNode


@dataclass
class RootCauseCandidate:
    """One ranked suspect in the upstream chain of a failing operation.

    The ``evidence`` dict records the per-signal numbers that fed into
    ``score`` so a reviewer can read the explanation, not just trust
    the number.
    """

    node_id: str
    library: str
    operation: str
    score: float
    chain_distance: int
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "library": self.library,
            "operation": self.operation,
            "score": self.score,
            "chain_distance": self.chain_distance,
            "evidence": self.evidence,
        }


class DeviationRCA:
    """Walk a trace upstream from a target node and score each ancestor.

    Construct with an optional baseline graph to enable structural-
    deviation scoring against a known-good reference run; without one,
    ranking is driven purely by proximity-weighted path confidence
    (still useful for "what's right above the failure?").
    """

    def __init__(self, baseline_graph: TraceGraph | None = None) -> None:
        self.baseline_graph = baseline_graph

    def analyze(
        self, graph: TraceGraph, target_node_id: str,
    ) -> list[RootCauseCandidate]:
        """Rank upstream operations of ``target_node_id`` by deviation score.

        Returns ``[]`` when the target node is not in ``graph``. Returns
        every upstream operation (no top-N cap here — callers can slice).
        """
        nodes_by_id: dict[str, TraceNode] = {n.node_id: n for n in graph.nodes}
        if target_node_id not in nodes_by_id:
            return []

        # Build child -> [edge] adjacency for the upstream walk. We
        # follow both LINEAGE_LINK (cross-domain) and DIRECT (intra-
        # domain) edges so the walk doesn't stop at the data/LLM seam.
        upstream_edges: dict[str, list] = {}
        for e in graph.edges:
            if e.kind in (EdgeKind.LINEAGE_LINK, EdgeKind.DIRECT):
                upstream_edges.setdefault(e.child_id, []).append(e)

        baseline_shapes = self._baseline_shapes() if self.baseline_graph else {}

        # BFS upstream tracking (distance, cumulative path confidence).
        # ``visited`` guards against revisiting; ``best_by_id`` keeps at
        # most one candidate per upstream node, holding the highest
        # score across all paths that reach it. Live traces can carry
        # self-loop edges (autolineage emits ``parent == child`` for
        # some pandas operations) and parallel edges between the same
        # pair; both would otherwise produce duplicate entries.
        best_by_id: dict[str, RootCauseCandidate] = {}
        visited: set[str] = set()
        queue: deque[tuple[str, int, float]] = deque([(target_node_id, 0, 1.0)])

        while queue:
            node_id, dist, path_conf = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            for edge in upstream_edges.get(node_id, []):
                parent_id = edge.parent_id
                # Skip self-loops — they're meaningless for upstream
                # analysis and we've seen them in the wild on
                # autolineage.select records.
                if parent_id == node_id:
                    continue
                parent = nodes_by_id.get(parent_id)
                if parent is None:
                    continue

                new_dist = dist + 1
                # Path confidence is the product of edge confidences
                # walked. A 0.5 link followed by a 1.0 link gives 0.5,
                # not "average 0.75" — chain confidence shouldn't
                # recover after a weak step.
                new_path_conf = path_conf * (edge.confidence or 1.0)

                score, evidence = self._score_operation(
                    parent, new_dist, new_path_conf, baseline_shapes,
                )

                existing = best_by_id.get(parent_id)
                if existing is None or score > existing.score:
                    best_by_id[parent_id] = RootCauseCandidate(
                        node_id=parent_id,
                        library=parent.library,
                        operation=parent.operation,
                        score=score,
                        chain_distance=new_dist,
                        evidence=evidence,
                    )

                queue.append((parent_id, new_dist, new_path_conf))

        # Highest suspicion first; ties broken by closer-to-target,
        # then by node_id for determinism.
        candidates = list(best_by_id.values())
        candidates.sort(key=lambda c: (-c.score, c.chain_distance, c.node_id))
        return candidates

    # -----------------------------------------------------------------
    # Per-operation scoring
    # -----------------------------------------------------------------

    def _score_operation(
        self,
        node: TraceNode,
        distance: int,
        path_conf: float,
        baseline_shapes: dict[str, dict[str, Any]],
    ) -> tuple[float, dict[str, Any]]:
        evidence: dict[str, Any] = {}

        proximity = 1.0 / (1.0 + distance)
        evidence["proximity"] = round(proximity, 4)
        evidence["path_confidence"] = round(path_conf, 4)

        structural_dev = 0.0
        if baseline_shapes:
            sig = f"{node.library}.{node.operation}"
            cur_shape = self._node_shape(node)
            base_shape = baseline_shapes.get(sig)
            if base_shape and cur_shape:
                structural_dev = self._shape_deviation(base_shape, cur_shape)
                evidence["structural_deviation"] = round(structural_dev, 4)
                evidence["baseline_shape"] = base_shape
                evidence["current_shape"] = cur_shape

        if baseline_shapes:
            score = (structural_dev * 0.5) + (proximity * 0.3) + (path_conf * 0.2)
        else:
            score = (proximity * 0.6) + (path_conf * 0.4)

        return min(1.0, score), evidence

    def _node_shape(self, node: TraceNode) -> dict[str, Any] | None:
        """Extract (rows, columns) from whatever shape representation the
        adapter happened to use. AutoLineage stores ``autolineage.shape``
        either as a [rows, cols] list or as just a row count; we accept
        both."""
        md = node.metadata or {}
        shape: dict[str, Any] = {}

        # rows
        rs = md.get("autolineage.shape")
        if isinstance(rs, (list, tuple)) and len(rs) >= 1:
            try:
                shape["rows"] = float(rs[0])
            except (TypeError, ValueError):
                pass
        elif isinstance(rs, (int, float)):
            shape["rows"] = float(rs)
        else:
            for key in ("rows", "row_count"):
                if key in md:
                    try:
                        shape["rows"] = float(md[key])
                        break
                    except (TypeError, ValueError):
                        pass

        # columns
        if isinstance(rs, (list, tuple)) and len(rs) >= 2:
            try:
                shape["columns"] = float(rs[1])
            except (TypeError, ValueError):
                pass
        else:
            cols = md.get("autolineage.columns")
            if isinstance(cols, (list, tuple)):
                shape["columns"] = float(len(cols))
            elif isinstance(cols, (int, float)):
                shape["columns"] = float(cols)
            elif "column_count" in md:
                try:
                    shape["columns"] = float(md["column_count"])
                except (TypeError, ValueError):
                    pass

        return shape or None

    def _baseline_shapes(self) -> dict[str, dict[str, Any]]:
        shapes: dict[str, dict[str, Any]] = {}
        assert self.baseline_graph is not None  # narrowed by caller
        for n in self.baseline_graph.nodes:
            sig = f"{n.library}.{n.operation}"
            s = self._node_shape(n)
            if s:
                shapes[sig] = s
        return shapes

    def _shape_deviation(
        self, base: dict[str, Any], cur: dict[str, Any],
    ) -> float:
        """Normalized [0, 1] deviation between two shape dicts.

        Per dimension, ``|b - c| / max(|b|, |c|)`` (which is 0 when
        identical and approaches 1 when the magnitudes diverge).
        Averaged across the dimensions present in both shapes.
        """
        dev = 0.0
        count = 0
        for key in ("rows", "columns"):
            if key in base and key in cur:
                try:
                    b, c = float(base[key]), float(cur[key])
                except (TypeError, ValueError):
                    continue
                m = max(abs(b), abs(c))
                if m > 0:
                    dev += abs(b - c) / m
                    count += 1
        return dev / count if count else 0.0
