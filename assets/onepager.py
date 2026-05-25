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
        self.set_subject(_safe("Self-hosted AI lineage and audit"))

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
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*ACCENT)
        self.cell(0, 11, "RudriQ", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*DARK)
        self.multi_cell(
            0, 5,
            "Self-hosted, audit-grade evidence of why AI systems fail.",
            new_x="LMARGIN", new_y="NEXT",
        )
        # Accent rule.
        y = self.get_y() + 2
        self.set_draw_color(*ACCENT)
        self.set_line_width(0.6)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_y(y + 3)

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
            "Production AI fails because of upstream data, not the "
            "model. When a RAG system returns a wrong answer, the cause "
            "is usually the retrieval or the pipeline - but LLM "
            "observability tools start at the model call and can't see "
            "it. For regulated teams, \"we don't know why it failed\" "
            "is a compliance and liability exposure, not just a bug.",
        ],
    )

    pdf.section(
        "What RudriQ does",
        [
            "- Links every data operation to every LLM call - one "
            "causal trace across the seam no other tool covers.",
            "- Scores answer quality locally: groundedness, retrieval "
            "relevance, drift, consistency, coherence. No cloud calls.",
            "- Ranks the likely root cause when an answer is wrong "
            "(honest framing: ranked suspects, not a causal proof).",
            "- Produces a deterministic, hashable audit report "
            "(Markdown / JSON / PDF) for EU AI Act, AI-liability "
            "underwriting, and litigation defense.",
        ],
    )

    pdf.section(
        "Why it's different",
        [
            "- Self-hosted. Zero outbound network calls in core. Runs "
            "inside your VPC, air-gapped capable. Built for buyers who "
            "can't send PHI to a vendor cloud.",
            "- OpenTelemetry-compatible, not OpenTelemetry-bound - "
            "expresses cross-domain lineage that pure-OTel tools "
            "structurally cannot.",
        ],
    )

    pdf.section(
        "Proof",
        [
            "On a realistic 250-operation RAG pipeline: 23 of 43 LLM "
            "calls linked to their upstream data (the rest are fresh-"
            "string query embeddings, correctly unlinked). Drift "
            "detection validated (1.000 identical vs 0.820 perturbed, "
            "with new behavior flagged). Audit reports byte-"
            "reproducible across exports - hash them as evidence. Built "
            "on AutoLineage, published research (SSRN).",
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
