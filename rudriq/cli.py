"""
RudriQ command-line interface.

    rudriq diagnose [--target METRIC] [--baseline PATH]
    rudriq audit    --run-id ID [--format FORMAT] [--output FILE]
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
    result = diagnose(
        target_metric=args.target,
        baseline_path=args.baseline,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    try:
        if args.format == "json":
            content = export_audit_json(args.run_id)
        elif args.format == "markdown":
            content = export_audit_markdown(args.run_id)
        elif args.format == "pdf":
            print(
                "error: PDF rendering not yet supported. "
                "Use --format markdown and convert with pandoc.",
                file=sys.stderr,
            )
            return 2
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


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"rudriq {__version__}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rudriq",
        description="Connect LLM behavior to its upstream data lineage.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_diag = sub.add_parser("diagnose", help="Cross-domain root-cause analysis")
    p_diag.add_argument("--target", help="Target metric whose change to investigate")
    p_diag.add_argument("--baseline", help="Path to baseline fingerprint for comparison")
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
    p_audit.set_defaults(func=_cmd_audit)

    p_ver = sub.add_parser("version", help="Print version")
    p_ver.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())