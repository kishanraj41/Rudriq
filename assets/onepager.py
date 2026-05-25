"""Generate the RudriQ design-partner one-pager (PDF).

The 30-second attachment for design-partner outreach. A compliance
officer or ML lead reads this before deciding whether to reply, so
every line earns its place — problem, what-it-does, why-different,
proof, the-ask, on one Letter page.

Visual identity reuses the same accent and Latin-1 sanitization as
``rudriq.export.pdf`` so the brief and the audit report look like
they come from the same tool — they do.

Run:

    python assets/onepager.py             # writes ./rudriq_onepager.pdf
    python assets/onepager.py path.pdf    # writes to a specific path
"""

from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF

# Match rudriq.export.pdf for visual consistency.
ACCENT = (11, 79, 108)       # 0B4F6C
DARK = (26, 26, 26)
GRAY = (102, 102, 102)
RULE = (220, 220, 220)


# The exporter sanitizes Unicode to Latin-1 because fpdf2's built-in
# Helvetica is core, not TTF. Keep the substitution table aligned with
# rudriq.export.pdf so the two outputs handle the same character set.
_UNICODE_TO_ASCII = {
    "—": "-",
    "–": "-",
    "−": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    " ": " ",
    "→": "->",
    "•": "-",
}


def _safe(text: str) -> str:
    for src, dst in _UNICODE_TO_ASCII.items():
        if src in text:
            text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


class _OnePager(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="Letter")
        # Single-page document — no auto-break. The layout is sized to
        # fit; if a future edit overflows we want a hard failure rather
        # than a silent overflow onto page 2.
        self.set_auto_page_break(auto=False, margin=10)
        self.set_margins(15, 14, 15)
        # Determinism (matches rudriq.export.pdf): no wall-clock in
        # metadata so two builds produce byte-identical PDFs.
        from datetime import datetime, timezone
        self.set_creation_date(datetime(2026, 5, 25, tzinfo=timezone.utc))
        self.set_producer(_safe("RudriQ one-pager generator"))
        self.set_title(_safe("RudriQ - Design partner brief"))
        self.set_subject(_safe("AI audit infrastructure for regulated AI"))

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

    # -- typography ------------------------------------------------
    def title_block(self) -> None:
        self.set_xy(self.l_margin, self.t_margin)
        self.set_font("Helvetica", "B", 26)
        self.set_text_color(*ACCENT)
        self.cell(0, 10, "RudriQ", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(*DARK)
        # The locked one-sentence headline. Audit-led, not debug-led:
        # the wedge is "we produce the artifact a regulator wants",
        # not "we help you debug your model." Order matters here.
        self.multi_cell(
            0, 4.8,
            "RudriQ is self-hosted audit infrastructure for clinical AI: "
            "it traces why an answer happened, proves whether it was "
            "grounded, and produces a compliance-ready report — with "
            "zero data leaving your environment.",
            new_x="LMARGIN", new_y="NEXT",
        )
        # Accent rule.
        y = self.get_y() + 1.5
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_y(y + 2.5)

    def section(self, heading: str, body_lines: list[str]) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*ACCENT)
        self.cell(0, 5, heading.upper(), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "", 9.5)
        for line in body_lines:
            if line.startswith("- "):
                self.set_x(self.l_margin + 4)
                self.cell(3, 4.6, "-")
                self.multi_cell(
                    self.w - self.r_margin - self.l_margin - 4 - 3,
                    4.6, line[2:],
                    new_x="LMARGIN", new_y="NEXT",
                )
            else:
                self.set_x(self.l_margin)
                self.multi_cell(
                    0, 4.6, line,
                    new_x="LMARGIN", new_y="NEXT",
                )
        self.ln(2)

    def footer_block(self) -> None:
        # Pinned to ~25mm from the bottom so the layout is stable.
        self.set_y(self.h - 25)
        self.set_draw_color(*RULE)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*ACCENT)
        self.cell(0, 5, "THE ASK", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(
            0, 4.6,
            "We're looking for 2-3 design partners in regulated AI "
            "(healthcare first) to shape RudriQ before GA. Free during "
            "the partnership. You get the tool, direct input on the "
            "roadmap, and audit artifacts for your own compliance needs.",
            new_x="LMARGIN", new_y="NEXT",
        )
        self.ln(1)
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*GRAY)
        self.cell(
            0, 4,
            "Kishan Raj VG  -  github.com/kishanraj41/rudriq  -  "
            "AutoLineage paper: papers.ssrn.com (search 'AutoLineage')",
        )


def build_pdf(output_path: str | Path = "rudriq_onepager.pdf") -> str:
    pdf = _OnePager()
    pdf.add_page()
    pdf.title_block()

    pdf.section(
        "The problem",
        [
            # Audit-led framing: 'compliance exposure' before 'bug'.
            # Note 'most LLM observability tools', not 'every' — the
            # claim has to survive a skeptical eng-lead with a Datadog
            # subscription.
            "When a clinical AI system gives a wrong or unsupported "
            "answer, regulated teams can't currently prove why it "
            "happened or trace it back through the pipeline - and they "
            "can't do it without sending data to a vendor cloud. For a "
            "HIPAA-bound or EU AI Act-regulated team, \"we can't "
            "explain what our AI did\" is a compliance and liability "
            "exposure, not just an engineering gap. Most LLM "
            "observability tools begin at the model call, which misses "
            "the upstream data and retrieval failures that actually "
            "cause bad answers - and they log to the cloud, which PHI "
            "can't.",
        ],
    )

    pdf.section(
        "What RudriQ does",
        [
            # Reordered audit-first, debugging-second. The deliverable
            # (compliance artifact) is bullet one; the root-cause hook
            # is the deliberate 'and-also' at the end.
            "- Produces a deterministic, tamper-evident audit report "
            "(Markdown / JSON / PDF) for every run - a compliance "
            "artifact you store and verify, generated entirely "
            "on-premise.",
            "- Traces every answer back through retrieval and data "
            "preparation to its source - the full provenance chain "
            "behind any output.",
            "- Proves whether each answer was grounded in its retrieved "
            "context, scored locally with no cloud calls.",
            "- And when an answer is wrong, ranks the likely upstream "
            "cause - so the same tool that audits also tells you why "
            "it broke.",
        ],
    )

    pdf.section(
        "Why it's different",
        [
            "- Self-hosted. Zero outbound network calls in core. Runs "
            "inside your environment, air-gapped capable. Built for "
            "teams who cannot send PHI to a vendor cloud - the buyers "
            "cloud-first observability tools structurally can't serve.",
            "- OpenTelemetry-compatible, not OpenTelemetry-bound - it "
            "expresses cross-domain data-to-LLM lineage that pure-OTel "
            "tools can't represent.",
        ],
    )

    pdf.section(
        "Proof",
        [
            # Linkage metric reframed as completeness, not a raw
            # fraction. Same underlying truth (every linkable call was
            # linked; standalone query embeddings have no upstream),
            # but phrased so the reader doesn't pattern-match 23/43 as
            # 'partial coverage'.
            "On a realistic clinical-style RAG pipeline, RudriQ linked "
            "every answer-generation call that had upstream data "
            "provenance back to its source, and correctly left "
            "standalone query embeddings unlinked. Drift detection "
            "validated (1.000 identical vs 0.820 perturbed). Audit "
            "reports byte-reproducible. Built on AutoLineage - "
            "published research (SSRN).",
        ],
    )

    pdf.footer_block()
    out = str(output_path)
    pdf.output(out)
    return out


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "rudriq_onepager.pdf"
    path = build_pdf(out)
    print(f"One-pager written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
