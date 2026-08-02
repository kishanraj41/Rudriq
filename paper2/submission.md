# Submission metadata — "Measuring the RudriQ Linker"

Paper 2 of the RudriQ/AutoLineage line. Release decision: **preprint now, venue later** — Arm B is
non-blocking for the preprint. Targets: **arXiv** (citable, venue-compatible) and **SSRN** (where
Paper 1 lives, abstract_id=6683825).

Both accept prior/dual posting; confirm the eventual venue permits preprinting before submitting there
(most CS venues do).

---

## Title

Measuring the RudriQ Linker: Accuracy, Calibration, and a Trust Envelope for Cross-Domain Data→LLM Provenance

## Author

Kishan Raj VG (full legal name: Kishan Raj Vandhavasi Goutham Kumar; publishes as "Kishan Raj VG")
University of the Cumberlands, PhD Program, Graduate Information Technology, Williamsburg, Kentucky, USA
kishanraj41@gmail.com

## Abstract — plain text

SSRN's abstract field does not render LaTeX or Markdown. Paste this version, not the one from `paper.tex`.

> Modern ML systems increasingly cross a domain boundary that existing observability tools do not: data
> prepared by pandas/Spark flows into LLM calls (embeddings, chat completions). The RudriQ linker attempts
> to recover this data-to-LLM lineage automatically, attaching a confidence-scored provenance edge from an
> upstream data artifact to each LLM call. The central question for an audit tool is not whether such a
> linker exists but whether its links are correct — and that has gone unmeasured, because cross-domain
> ground truth is hard to establish.
>
> We contribute a construction-based benchmark that makes ground truth tractable: pipelines in which we
> assign every node identity, so the true data-to-LLM edge set is known by construction, paired with an
> adversarial arm built to make the linker fail. All findings below are established on this synthetic and
> adversarial corpus; ecological validation on real third-party pipelines is in progress, and we mark
> throughout which quantities are portable linker properties and which are corpus-composition-dependent.
>
> On the constructed corpus we find: (1) the linker's per-link confidence scale is calibrated in structure —
> measured precision falls monotonically as stated confidence falls, a structural property (as distinct from
> the specific per-band precision values, which are composition-dependent); this is the keystone result.
> (2) the exact-match strategy (object-identity) is near-perfectly precise (precision 1.000, 95% CI [0.987,
> 1.000], n=300), and no false positives were observed on representative new data (0 of 200; 95% CI upper
> bound below 2%). (3) the false-positive surface comprises two bounded, characterized mechanisms that
> respect their boundaries, yielding a confidence-threshold deployment guideline. (4) common-path
> instrumentation overhead is below 0.1% of a single LLM call, with two characterized and mitigable
> fallback costs.
>
> This is Paper 2 of the RudriQ/AutoLineage line. AutoLineage (Paper 1) contributes the import-time capture
> mechanism; this paper measures the cross-domain linker's accuracy and calibration. Same system, different
> contribution.

## Keywords

data lineage; provenance; LLM observability; cross-domain lineage; confidence calibration; benchmark
methodology; retrieval-augmented generation; ML pipelines; audit; Python instrumentation

## arXiv

- **Primary category:** cs.SE (Software Engineering). Cross-list: cs.DB (Databases), cs.LG.
- **Author metadata:** given name `Kishan Raj`, family name `VG`.
- **License:** choose at upload (CC BY 4.0 keeps the widest reuse; arXiv's non-exclusive default is the
  conservative option).
- **Endorsement:** first-time `cs` submitters may need an endorsement. Paper 1 is on SSRN, not arXiv, so
  this is likely a first arXiv submission — allow time for it.
- **Upload:** `paper.tex` + `refs.bib`. arXiv runs BibTeX itself, but also accepts the generated `paper.bbl`;
  including the `.bbl` avoids build surprises.

## SSRN

- Upload the **compiled PDF** (SSRN does not build LaTeX).
- Paste the plain-text abstract above.
- Link to Paper 1 (abstract_id=6683825) as related work by the same author.

## Pre-upload checklist

- [x] `paper.tex` + `refs.bib` + `figs/` compile cleanly. Built with Tectonic 0.17.0 → `paper.pdf`,
      13 pages. No overfull boxes, no undefined references or citations. `authblk` + `\thanks` render
      the affiliation and the legal-name footnote correctly.
- [x] All 17 citations resolve, numbered [1]–[17] in the built reference list, each with a year.
- [x] Both figures embed as vector art and place cleanly (Figure 1 p.6, Figure 2 p.9).
- [x] PDF metadata (title, author) embedded via `hypersetup`.
- [x] Date reads August 2026.
- [x] No em dashes in the prose (0 in the rendered PDF; the 7 en dashes are numeric ranges).
      See "Open item" below re: the overhead figure legend.
- [x] Numbers verified against `corpus/artifacts/scale_up_n900.json` (N=900 re-run reproduces §5.1–5.5
      exactly) and `trust_map.json` (matches §5.7 row-for-row).
- [ ] `paper2/` committed to git.
- [ ] Rebuild after any further edit: `tectonic paper.tex` (or Overleaf: `pdflatex` → `bibtex` →
      `pdflatex` ×2). `paper.bbl` is checked in so arXiv can skip its BibTeX pass.

## Known divergence

`paper.md` titles the work "Measuring a Cross-Domain Lineage Linker…" while `paper.tex` (the submission
artifact) uses "Measuring the RudriQ Linker…". Pre-existing; the LaTeX title is the one that ships.

## Deliberately deferred (not blockers)

- **Arm B**: two annotated third-party pipelines with κ. Externally gated, user-sourced. Until it lands,
  inferred-strategy precision *values* and the pre-emption *correctness* claim stay scoped as
  on-corpus/future-work. The paper already says this in §4.2, §5.5, §6, and §8.

## Open item

The `overhead.pdf` figure's **legend** contains em dashes (`object_identity — flat O(1)`), baked into the
vector art. The paper prose is em-dash-free, but the figure is not. Fixing it needs the plotting script,
which is not in this repo — the two figure PDFs arrived as finished artifacts. If that script surfaces,
commit it under `paper2/figs/` so the figures become reproducible rather than one-off files.
