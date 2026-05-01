"""
RudriQ command-line interface.

    rudriq diagnose [--target METRIC] [--baseline PATH]
    rudriq audit    [--template NAME] [--format FORMAT]
    rudriq version
"""

from __future__ import annotations

import argparse
import json
import sys

from rudriq import __version__
from rudriq.analyzer.diagnose import diagnose
from rudriq.export.audit import generate_audit_report


def _cmd_diagnose(args: argparse.Namespace) -> int:
    result = diagnose(
        target_metric=args.target,
        baseline_path=args.baseline,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    try:
        result = generate_audit_report(
            template=args.template,
            output_format=args.format,
        )
    except NotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=str))
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
    p_audit.add_argument("--template", default="default", help="Report template name")
    p_audit.add_argument(
        "--format", default="json", choices=["json", "markdown", "pdf"],
        help="Output format",
    )
    p_audit.set_defaults(func=_cmd_audit)

    p_ver = sub.add_parser("version", help="Print version")
    p_ver.set_defaults(func=_cmd_version)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())