"""
RudriQ command-line interface.

    rudriq diagnose [--target METRIC] [--baseline PATH]
    rudriq audit    --run-id ID [--format FORMAT] [--output FILE]
    rudriq evaluate --run-id ID [--metrics LIST] [--format FORMAT]
    rudriq version
"""

from __future__ import annotations

import argparse
import json
import sys

from rudriq import __version__
from rudriq.analyzer.diagnose import diagnose
from rudriq.export.audit import export_audit_json, export_audit_markdown


def _cmd_diagnose(args: argparse.Namespace) -> int:
    # When --run-id is given, take the Day 17 Thread C deviation-weighted
    # RCA path. When it isn't, fall back to the legacy v0.0.1 structural
    # diagnose() stub so the original CLI contract still works for
    # callers who haven't migrated.
    if getattr(args, "run_id", None):
        return _cmd_diagnose_rca(args)
    result = diagnose(
        target_metric=args.target,
        baseline_path=args.baseline,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_diagnose_rca(args: argparse.Namespace) -> int:
    """Deviation-weighted root-cause analysis on a stored run.

    Honest framing: ranks SUSPECTS by deviation, not a causal proof.
    The CLI output and JSON shape both surface that distinction so the
    operator doesn't accidentally over-trust the ranking.
    """
    from rudriq.analyzer.deviation_rca import DeviationRCA
    from rudriq.storage import get_default_storage

    storage = get_default_storage()
    graph = storage.load_run(args.run_id)
    if graph is None:
        print(f"error: no run found with id {args.run_id}", file=sys.stderr)
        return 1

    if not args.target:
        print(
            "error: --target NODE_ID is required when --run-id is given.",
            file=sys.stderr,
        )
        return 2

    baseline = None
    if getattr(args, "baseline_run_id", None):
        baseline = storage.load_run(args.baseline_run_id)
        if baseline is None:
            print(
                f"warning: baseline run {args.baseline_run_id} not found; "
                f"structural deviation will not be scored.",
                file=sys.stderr,
            )

    top = getattr(args, "top", 5) or 5
    candidates = DeviationRCA(baseline_graph=baseline).analyze(graph, args.target)[:top]

    if args.format == "json":
        print(json.dumps(
            [c.to_dict() for c in candidates],
            indent=2, sort_keys=True, default=str,
        ))
        return 0

    if not candidates:
        print(f"No upstream operations found for {args.target}.")
        return 0
    print(f"Top {len(candidates)} root-cause suspects for {args.target}:")
    print("(Heuristic ranking by deviation — NOT a causal proof.)\n")
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c.library}.{c.operation}  (node {c.node_id})")
        print(f"   score {c.score:.3f}  | {c.chain_distance} hop(s) upstream")
        for k, v in sorted(c.evidence.items()):
            print(f"     - {k}: {v}")
        print()
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    # Resolve --include-evals + --include-rca knobs before calling into
    # the exporter so the same parsed values feed both JSON and Markdown
    # paths.
    eval_metrics: list[str] | None = None
    baseline_graph = None
    include_evals = getattr(args, "include_evals", False)
    include_rca = getattr(args, "include_rca", False)
    rca_target = getattr(args, "rca_target", None)

    if include_evals:
        raw_metrics = getattr(args, "eval_metrics", None)
        if raw_metrics:
            eval_metrics = [
                m.strip() for m in raw_metrics.split(",") if m.strip()
            ]

    # The baseline run-id is shared by --include-evals (drift) and
    # --include-rca (structural deviation). Load once if either flag
    # is set, reuse for both.
    if include_evals or include_rca:
        baseline_run_id = getattr(args, "baseline_run_id", None)
        if baseline_run_id:
            from rudriq.storage import get_default_storage
            baseline_graph = get_default_storage().load_run(baseline_run_id)
            if baseline_graph is None:
                print(
                    f"warning: baseline run {baseline_run_id} not found; "
                    f"drift / RCA structural deviation will fall back.",
                    file=sys.stderr,
                )

    try:
        if args.format == "json":
            content = export_audit_json(
                args.run_id,
                include_evals=include_evals,
                eval_metrics=eval_metrics,
                baseline_graph=baseline_graph,
                include_rca=include_rca,
                rca_target=rca_target,
            )
        elif args.format == "markdown":
            content = export_audit_markdown(
                args.run_id,
                include_evals=include_evals,
                eval_metrics=eval_metrics,
                baseline_graph=baseline_graph,
                include_rca=include_rca,
                rca_target=rca_target,
            )
        elif args.format == "pdf":
            # PDF goes straight to disk — fpdf2's output() is binary, not
            # serializable to a stdout string, and a compliance officer
            # always wants a file anyway.
            if not args.output:
                print(
                    "error: --format pdf requires --output FILE.",
                    file=sys.stderr,
                )
                return 2
            try:
                from rudriq.export.pdf import export_audit_pdf
            except ImportError as exc:
                print(f"error: PDF export not available: {exc}", file=sys.stderr)
                return 2
            try:
                export_audit_pdf(
                    args.run_id, args.output,
                    include_evals=include_evals,
                    eval_metrics=eval_metrics,
                    baseline_graph=baseline_graph,
                    include_rca=include_rca,
                    rca_target=rca_target,
                )
            except RuntimeError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            print(f"PDF audit report written to {args.output}", file=sys.stderr)
            return 0
        else:
            print(f"error: unknown format '{args.format}'", file=sys.stderr)
            return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Audit report written to {args.output}", file=sys.stderr)
    else:
        print(content)
    return 0


_EVALUATOR_REGISTRY = {
    "retrieval_relevance": "rudriq.evaluate.retrieval_relevance:RetrievalRelevanceEvaluator",
    "groundedness": "rudriq.evaluate.groundedness:GroundednessEvaluator",
    "coherence": "rudriq.evaluate.coherence:CoherenceEvaluator",
    "consistency": "rudriq.evaluate.consistency:ConsistencyEvaluator",
    # ``drift`` is special-cased in _cmd_evaluate because it takes a
    # constructor argument (baseline graph); not listed here.
}


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from rudriq.evaluate.base import run_evaluators
    from rudriq.storage import get_default_storage

    requested = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # ``drift`` isn't in _EVALUATOR_REGISTRY because it takes a baseline
    # graph in its constructor. It's accepted here as a known special.
    known = set(_EVALUATOR_REGISTRY) | {"drift"}
    unknown = [m for m in requested if m not in known]
    if unknown:
        print(
            f"error: unknown metric(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(known))}",
            file=sys.stderr,
        )
        return 2

    storage = get_default_storage()
    graph = storage.load_run(args.run_id)
    if graph is None:
        print(f"error: no run found with id {args.run_id}", file=sys.stderr)
        return 1

    evaluators = []
    for m in requested:
        if m == "drift":
            # Drift is the only evaluator that takes a constructor arg —
            # it needs a baseline graph injected. If no baseline run-id
            # was given (or the baseline doesn't load), the evaluator
            # itself emits SKIPPED.
            from rudriq.evaluate.drift import DriftEvaluator

            baseline = None
            if args.baseline_run_id:
                baseline = storage.load_run(args.baseline_run_id)
                if baseline is None:
                    print(
                        f"warning: baseline run {args.baseline_run_id} not "
                        f"found; drift will SKIP.",
                        file=sys.stderr,
                    )
            evaluators.append(DriftEvaluator(baseline_graph=baseline))
        else:
            module_path, _, cls_name = _EVALUATOR_REGISTRY[m].partition(":")
            import importlib
            module = importlib.import_module(module_path)
            evaluators.append(getattr(module, cls_name)())

    results = run_evaluators(graph, evaluators)

    if args.format == "json":
        print(json.dumps([r.to_dict() for r in results], indent=2, default=str))
    else:
        for r in results:
            score_str = f"{r.score:.2f}" if r.score is not None else "N/A"
            print(f"[{r.status.value.upper()}] {r.metric} = {score_str}")
            print(f"  {r.explanation}")
            if r.node_id:
                print(f"  (node: {r.node_id})")
            print()
    return 0


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"rudriq {__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rudriq",
        description="Connect LLM behavior to its upstream data lineage.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser(
        "diagnose",
        help=(
            "Cross-domain root-cause analysis. With --run-id + --target NODE, "
            "ranks upstream operations by deviation (Day 17 Thread C). "
            "Without --run-id, returns the legacy structural-summary stub."
        ),
    )
    # Legacy stub args (kept for backward compatibility with v0.0.1 callers).
    p_diag.add_argument(
        "--target",
        help=(
            "When --run-id is given: the target node_id to diagnose (a "
            "failing LLM call). Without --run-id: legacy 'target metric' "
            "for the v0.0.1 structural-summary stub."
        ),
    )
    p_diag.add_argument(
        "--baseline",
        help="Legacy: path to baseline fingerprint (v0.0.1 stub).",
    )
    # New deviation-weighted RCA args (Day 17 Thread C).
    p_diag.add_argument(
        "--run-id", default=None,
        help="Run ID containing the target node (enables deviation-weighted RCA).",
    )
    p_diag.add_argument(
        "--baseline-run-id", default=None,
        help=(
            "Baseline run for structural-deviation scoring. Without it, "
            "ranking falls back to proximity-weighted link confidence."
        ),
    )
    p_diag.add_argument(
        "--top", type=int, default=5,
        help="Cap on the number of suspects to return (default: 5).",
    )
    p_diag.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format for the new RCA path (default: text).",
    )
    p_diag.set_defaults(func=_cmd_diagnose)

    p_audit = sub.add_parser("audit", help="Generate audit report from unified trace")
    p_audit.add_argument(
        "--run-id", required=True,
        help="Run ID to export (use rudriq.storage.get_default_storage().list_runs() to find one)",
    )
    p_audit.add_argument(
        "--format", default="markdown", choices=["json", "markdown", "pdf"],
        help="Output format (default: markdown)",
    )
    p_audit.add_argument(
        "--output",
        help="Write to file instead of stdout",
    )
    p_audit.add_argument(
        "--include-evals", action="store_true",
        help=(
            "Run quality evaluators and embed results in the audit "
            "report. Default metrics: retrieval_relevance, groundedness, "
            "coherence, consistency. Requires fastembed for real scores "
            "(install rudriq[evaluate]); evaluators DEGRADE gracefully "
            "if missing."
        ),
    )
    p_audit.add_argument(
        "--eval-metrics", default=None,
        help=(
            "Comma-separated metrics for --include-evals. Pass 'drift' "
            "together with --baseline-run-id to include cross-run drift."
        ),
    )
    p_audit.add_argument(
        "--baseline-run-id", default=None,
        help=(
            "Baseline run for drift (if 'drift' is in --eval-metrics) AND "
            "for RCA structural-deviation scoring (if --include-rca). "
            "Ignored when neither is in play."
        ),
    )
    p_audit.add_argument(
        "--include-rca", action="store_true",
        help=(
            "Embed deviation-weighted root-cause analysis into the audit "
            "report. By default auto-targets the LLM call with the lowest "
            "groundedness score so the user need not know node_ids. "
            "Heuristic suspect ranking, NOT a causal proof."
        ),
    )
    p_audit.add_argument(
        "--rca-target", default=None,
        help=(
            "Specific node_id to diagnose for --include-rca. Default: "
            "auto-select the worst LLM call by groundedness."
        ),
    )
    p_audit.set_defaults(func=_cmd_audit)

    p_eval = sub.add_parser(
        "evaluate", help="Run quality evaluators against a trace",
    )
    p_eval.add_argument(
        "--run-id", required=True,
        help="Run ID to evaluate (see rudriq.storage.get_default_storage().list_runs())",
    )
    p_eval.add_argument(
        "--metrics", default="retrieval_relevance,groundedness",
        help=(
            "Comma-separated list of metrics to run. Available: "
            + ", ".join(sorted(set(_EVALUATOR_REGISTRY) | {"drift"}))
        ),
    )
    p_eval.add_argument(
        "--baseline-run-id", default=None,
        help=(
            "Baseline run to compare against (required for the 'drift' "
            "metric; ignored by others)."
        ),
    )
    p_eval.add_argument(
        "--format", default="text", choices=["text", "json"],
        help="Output format (default: text)",
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    p_ver = sub.add_parser("version", help="Print version")
    p_ver.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())