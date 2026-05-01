"""
Audit report generation from the unified trace.

When you have a complete record of every data operation and every LLM
call, generating a human-readable audit report becomes a query rather
than a documentation effort. This module provides report generation in
multiple formats; the regulatory templates are user-supplied.

In v0.0.1, ``generate_audit_report`` returns a structured dict. v0.2
adds Markdown and PDF rendering.
"""

from __future__ import annotations

from typing import Any

from rudriq.core.tracker import get_tracker


def generate_audit_report(
    template: str = "default",
    output_format: str = "json",
) -> dict[str, Any]:
    """
    Generate an audit report from the current unified trace.

    Parameters
    ----------
    template:
        Which template to use. v0.0.1 supports only "default".
        Future templates may include user-supplied regulatory or
        internal-compliance templates.
    output_format:
        One of "json", "markdown", "pdf". v0.0.1 supports only "json".

    Returns
    -------
    Dictionary representation of the audit report. Each section maps
    to either a structural summary or a list of trace operations
    relevant to that section.
    """
    if template != "default":
        raise NotImplementedError(
            f"Template '{template}' not supported in v0.0.1. "
            "User-supplied templates land in v0.2."
        )
    if output_format != "json":
        raise NotImplementedError(
            f"Output format '{output_format}' not supported in v0.0.1. "
            "Markdown and PDF rendering land in v0.2."
        )

    tracker = get_tracker()
    graph = tracker.get_full_graph()

    return {
        "report_version": "0.0.1",
        "template": template,
        "trace_summary": {
            "node_count": graph["node_count"],
        },
        "data_operations": [
            n for n in graph["nodes"] if n.get("domain") == "data_lineage"
        ],
        "llm_operations": [
            n for n in graph["nodes"] if n.get("domain") == "llm"
        ],
        "cross_domain_links": [
            n for n in graph["nodes"]
            if n.get("lineage_parents") and len(n["lineage_parents"]) > 0
        ],
        "notes": [
            "RudriQ v0.0.1: report contains structural summary only. "
            "Full content rendering lands in v0.2.",
        ],
    }