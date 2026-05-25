"""PDF export for audit reports via fpdf2 (pure Python, no system libs).

Renders from the structured report dict (the same source the JSON
exporter uses), NOT by parsing the Markdown — more robust and
sidesteps Markdown edge cases entirely.

Determinism (load-bearing for the audit-grade claim): PDF metadata
embeds a creation timestamp by default. fpdf2's
``set_creation_date(None)`` does NOT suppress the timestamp — it sets
it to *current time* (verified against fpdf2 2.8.7 docstring). We
instead pin it to the run's own ``created_at`` so two exports of the
same run produce byte-identical PDFs. The Title/Subject/Producer
fields are likewise pinned to fixed strings (no wall-clock).

The fpdf2 dependency is an optional ``[pdf]`` extra. If absent, the
exporter raises a clean ``RuntimeError`` pointing at the install
command — same graceful-degradation pattern as ``[evaluate]``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

_LOG = logging.getLogger("rudriq.export.pdf")

# Colours — RudriQ audit visual identity (matches the Markdown emoji
# traffic lights and the README accent).
ACCENT = (11, 79, 108)        # 0B4F6C
DARK = (26, 26, 26)
GRAY = (102, 102, 102)
TRAFFIC = {
    "green": (40, 160, 70),
    "yellow": (200, 150, 0),
    "red": (200, 50, 50),
    "gray": (150, 150, 150),
}

# fpdf2's built-in "Helvetica" is a Latin-1 core font (no TTF). The
# audit body sometimes carries Unicode chars (em dash, ellipsis,
# typographic quotes) from evaluator explanations / RCA notes. Map
# them to ASCII equivalents at the renderer boundary so we don't
# either crash OR ship a multi-megabyte TTF dependency for a
# compliance-buyer PDF.
_UNICODE_TO_ASCII = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "−": "-",   # minus sign
    "…": "...",  # ellipsis
    "“": '"',   # left double quote
    "”": '"',   # right double quote
    "‘": "'",   # left single quote
    "’": "'",   # right single quote
    " ": " ",   # nbsp
    "→": "->",  # rightwards arrow
    "Δ": "delta",  # Greek capital delta (used in eval explanations)
    "α": "alpha",
    "β": "beta",
}


def _safe(text: str) -> str:
    """Map characters fpdf2's core Helvetica can't encode to ASCII.

    Anything still non-Latin-1 after the substitutions is replaced with
    ``?`` rather than raising — better a fuzzy character in a compliance
    PDF than a hard error mid-render. The substitution table is
    deliberately small (typography + a handful of common Greek letters
    used in evaluator explanations); add more entries here if a new
    code path emits them.
    """
    if not isinstance(text, str):
        text = str(text)
    for src, dst in _UNICODE_TO_ASCII.items():
        if src in text:
            text = text.replace(src, dst)
    # Drop anything still outside Latin-1.
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_audit_pdf(
    run_id: str,
    output_path: str,
    *,
    include_evals: bool = False,
    eval_metrics: list[str] | None = None,
    baseline_graph: Any = None,
    include_rca: bool = False,
    rca_target: str | None = None,
) -> str:
    """Generate a PDF audit report at ``output_path``. Returns the path.

    Raises:
        ValueError: if ``run_id`` is not in storage.
        RuntimeError: if the ``[pdf]`` extra (fpdf2) is not installed,
            with an explicit install hint.
    """
    from rudriq.export.audit import build_audit_report_dict

    report = build_audit_report_dict(
        run_id,
        include_evals=include_evals,
        eval_metrics=eval_metrics,
        baseline_graph=baseline_graph,
        include_rca=include_rca,
        rca_target=rca_target,
    )

    try:
        from fpdf import FPDF  # noqa: F401  (imported by _RudriqPDF too)
    except ImportError as exc:
        raise RuntimeError(
            "PDF export requires the 'pdf' extra. Install with: "
            "pip install 'rudriq[pdf]'. Alternatively, export Markdown "
            "(--format markdown) and convert with your own tool."
        ) from exc

    pdf = _build_renderer(report)
    pdf.render()
    pdf.output(output_path)
    return output_path


def _build_renderer(report: dict[str, Any]):
    """Construct the FPDF subclass lazily so importing this module
    doesn't require fpdf2 to be installed (tests + the rest of the
    package still load on a vanilla install)."""
    from fpdf import FPDF

    class _RudriqPDF(FPDF):
        def __init__(self, report: dict[str, Any]) -> None:
            super().__init__(orientation="P", unit="mm", format="Letter")
            self.report = report
            self.set_auto_page_break(auto=True, margin=18)
            self.set_margins(18, 18, 18)

            # DETERMINISM: pin PDF metadata to the run, not wall-clock.
            # ``set_creation_date(None)`` sets it to current time per
            # fpdf2 2.8 (verified against the docstring); pass the run's
            # ``created_at`` explicitly.
            run_block = report.get("run") or {}
            created_iso = run_block.get("created_at", "")
            try:
                dt = datetime.fromisoformat(created_iso) if created_iso else None
            except (TypeError, ValueError):
                dt = None
            if dt is not None:
                self.set_creation_date(dt)

            self.set_producer(_safe("RudriQ audit exporter"))
            self.set_title(_safe(
                f"RudriQ Audit Report - {run_block.get('run_id', '')}"
            ))
            self.set_subject(_safe(
                f"schema {report.get('schema_version', '')}"
            ))

        # Single-point sanitizer for every text-bearing call. fpdf2's
        # ``cell`` / ``multi_cell`` both accept a positional ``text``
        # arg or a keyword; we normalize it before delegating to super
        # so individual call sites don't have to remember _safe().
        def cell(self, *args, **kwargs):
            if len(args) >= 3:
                args = (args[0], args[1], _safe(args[2]), *args[3:])
            elif "text" in kwargs:
                kwargs["text"] = _safe(kwargs["text"])
            return super().cell(*args, **kwargs)

        def multi_cell(self, *args, **kwargs):
            if len(args) >= 3:
                args = (args[0], args[1], _safe(args[2]), *args[3:])
            elif "text" in kwargs:
                kwargs["text"] = _safe(kwargs["text"])
            return super().multi_cell(*args, **kwargs)

        # -- header / footer -----------------------------------------
        def header(self) -> None:
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.cell(0, 5, "RudriQ - Audit Report", align="R")
            self.ln(8)

        def footer(self) -> None:
            self.set_y(-12)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.cell(0, 5, f"Page {self.page_no()} of {{nb}}", align="C")

        # -- typography helpers --------------------------------------
        def h1(self, text: str) -> None:
            self.set_font("Helvetica", "B", 16)
            self.set_text_color(*ACCENT)
            self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*ACCENT)
            self.set_line_width(0.5)
            y = self.get_y()
            self.line(self.l_margin, y, self.w - self.r_margin, y)
            self.ln(3)
            self.set_text_color(*DARK)

        def h2(self, text: str) -> None:
            self.ln(2)
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(*ACCENT)
            self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(*DARK)

        def kv(self, key: str, value: Any) -> None:
            # fpdf2's multi_cell defaults new_x=RIGHT, which leaves x at
            # the right margin after rendering. The next cell() call
            # then has no horizontal space and crashes. We explicitly
            # set new_x=LMARGIN so x resets each row.
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "B", 9)
            self.cell(45, 5, str(key))
            self.set_font("Helvetica", "", 9)
            self.multi_cell(
                0, 5, str(value),
                new_x="LMARGIN", new_y="NEXT",
            )

        # -- main render ---------------------------------------------
        def render(self) -> None:
            self.alias_nb_pages()
            self.add_page()
            r = self.report

            self.h1("RudriQ Audit Report")
            run_block = r.get("run") or {}
            self.kv("Run ID:", run_block.get("run_id", ""))
            self.kv("Created:", run_block.get("created_at", ""))
            self.kv("Schema:", r.get("schema_version", ""))
            self.ln(3)

            self._render_summary(r.get("summary") or {})

            ev_summary = r.get("evaluation_summary")
            if ev_summary:
                self.h2("Quality Evaluation")
                self._traffic_table(ev_summary)

            rca = r.get("root_cause_analysis")
            if rca and rca.get("target_node_id"):
                self.h2("Root Cause Analysis")
                self.set_x(self.l_margin)
                self.set_font("Helvetica", "", 9)
                self.multi_cell(
                    0, 5,
                    f"Diagnosing node: {rca.get('target_node_id', '')}",
                    new_x="LMARGIN", new_y="NEXT",
                )
                self.set_font("Helvetica", "I", 8)
                self.set_text_color(*GRAY)
                self.multi_cell(
                    0, 4, rca.get("note", ""),
                    new_x="LMARGIN", new_y="NEXT",
                )
                self.set_text_color(*DARK)
                self.ln(1)
                if rca.get("candidates"):
                    self._rca_table(rca["candidates"])
                else:
                    self.set_font("Helvetica", "I", 9)
                    self.multi_cell(
                        0, 5,
                        "No upstream operations linked to the target.",
                        new_x="LMARGIN", new_y="NEXT",
                    )

            chains = r.get("lineage_chains") or []
            if chains:
                self.h2("Lineage Chains")
                for ch in chains[:30]:
                    self._render_chain(ch)
                if len(chains) > 30:
                    self.set_x(self.l_margin)
                    self.set_font("Helvetica", "I", 8)
                    self.set_text_color(*GRAY)
                    self.multi_cell(
                        0, 4,
                        f"(+{len(chains) - 30} more chains; see JSON export.)",
                        new_x="LMARGIN", new_y="NEXT",
                    )
                    self.set_text_color(*DARK)

            nodes = run_block.get("nodes") or []
            if nodes:
                self.h2("Operations Appendix")
                self._ops_table(nodes)

        # -- section renderers ---------------------------------------
        def _render_summary(self, summary: dict[str, Any]) -> None:
            self.h2("Summary")
            # Pick the high-signal fields a human reader cares about,
            # in a deliberate order. Skip ``run_id`` (already shown in
            # the title block) and ``libraries_seen`` (list — render
            # separately).
            key_labels = [
                ("started_at", "Started"),
                ("ended_at", "Ended"),
                ("duration_ms", "Duration (ms)"),
                ("total_nodes", "Total operations"),
                ("data_nodes", "  Data ops"),
                ("llm_nodes", "  LLM ops"),
                ("other_nodes", "  Other ops"),
                ("total_edges", "Total edges"),
                ("direct_edges", "  Direct (intra-domain)"),
                ("lineage_links", "  Lineage (cross-domain)"),
                ("causal_edges", "  Causal"),
                ("linked_llm_calls", "LLM calls linked to upstream"),
                ("unlinked_llm_calls", "LLM calls unlinked"),
            ]
            for key, label in key_labels:
                if key in summary:
                    v = summary[key]
                    if isinstance(v, float):
                        v = f"{v:.1f}"
                    self.kv(f"{label}:", v)
            libs = summary.get("libraries_seen")
            if libs:
                self.kv("Libraries observed:", ", ".join(libs) or "none")

        def _traffic_table(self, evs: dict[str, dict[str, Any]]) -> None:
            headers = ["Metric", "Status", "Mean", "Evaluated", "Notes"]
            widths = [50, 22, 22, 35, 45]
            self._table_header(headers, widths)
            self.set_font("Helvetica", "", 8)
            for metric in sorted(evs):
                s = evs[metric]
                light = s.get("traffic_light", "gray")
                mean = s.get("mean_score")
                mean_str = f"{mean:.2f}" if isinstance(mean, (int, float)) else "N/A"
                applicable = s.get("applicable_total", s.get("total", 0))
                evaluated = s.get("evaluated", 0)
                na = s.get("not_applicable", 0)
                notes = f"{na} not applicable" if na else ""

                self.set_text_color(*DARK)
                self.cell(widths[0], 6, metric, border=1)
                # Colour-code the status cell only — rest stays dark
                # so the row stays readable.
                self.set_text_color(*TRAFFIC.get(light, GRAY))
                self.cell(widths[1], 6, light.upper(), border=1)
                self.set_text_color(*DARK)
                self.cell(widths[2], 6, mean_str, border=1)
                self.cell(widths[3], 6, f"{evaluated}/{applicable}", border=1)
                self.cell(
                    widths[4], 6, notes, border=1,
                    new_x="LMARGIN", new_y="NEXT",
                )
            self.ln(2)

        def _rca_table(self, candidates: list[dict[str, Any]]) -> None:
            headers = ["#", "Operation", "Score", "Hops upstream"]
            widths = [10, 100, 25, 35]
            self._table_header(headers, widths)
            self.set_font("Helvetica", "", 8)
            for i, c in enumerate(candidates, 1):
                op = f"{c.get('library', '')}.{c.get('operation', '')}"
                self.cell(widths[0], 6, str(i), border=1)
                self.cell(widths[1], 6, op[:60], border=1)
                self.cell(widths[2], 6, f"{c.get('score', 0):.3f}", border=1)
                self.cell(
                    widths[3], 6, str(c.get("chain_distance", "")),
                    border=1, new_x="LMARGIN", new_y="NEXT",
                )
            self.ln(2)

        def _ops_table(self, nodes: list[dict[str, Any]]) -> None:
            headers = ["Kind", "Library", "Operation", "Node ID"]
            widths = [42, 32, 60, 36]
            self._table_header(headers, widths)
            self.set_font("Helvetica", "", 7)
            for n in nodes:
                self.cell(widths[0], 5, str(n.get("kind", ""))[:24], border=1)
                self.cell(widths[1], 5, str(n.get("library", ""))[:18], border=1)
                self.cell(widths[2], 5, str(n.get("operation", ""))[:38], border=1)
                self.cell(
                    widths[3], 5, str(n.get("node_id", ""))[:18],
                    border=1, new_x="LMARGIN", new_y="NEXT",
                )
            self.ln(2)

        def _render_chain(self, ch: dict[str, Any]) -> None:
            self.set_x(self.l_margin)
            self.set_font("Helvetica", "B", 8)
            target = f"{ch.get('llm_library', '')}.{ch.get('llm_operation', '')}"
            self.multi_cell(
                0, 5,
                f"LLM call: {target} (node {ch.get('llm_node_id', '')[:16]}, "
                f"upstream depth {ch.get('chain_length', 0)})",
                new_x="LMARGIN", new_y="NEXT",
            )
            self.set_font("Helvetica", "", 8)
            for step in (ch.get("chain") or [])[:10]:
                depth = int(step.get("depth", 1) or 1)
                indent = "    " * max(0, depth - 1)
                line = (
                    f"{indent}depth {depth}: "
                    f"{step.get('library', '')}.{step.get('operation', '')} "
                    f"({step.get('kind', '')})"
                )
                self.multi_cell(
                    0, 4, line,
                    new_x="LMARGIN", new_y="NEXT",
                )
            self.ln(1)

        # -- low-level helpers ---------------------------------------
        def _table_header(
            self, headers: list[str], widths: list[float],
        ) -> None:
            self.set_fill_color(*ACCENT)
            self.set_text_color(255, 255, 255)
            self.set_font("Helvetica", "B", 8)
            for h, w in zip(headers, widths):
                self.cell(w, 6, h, border=1, fill=True)
            self.ln()
            self.set_text_color(*DARK)
            self.set_font("Helvetica", "", 8)

    return _RudriqPDF(report)
