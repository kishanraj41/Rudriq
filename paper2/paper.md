# Measuring a Cross-Domain Lineage Linker: Accuracy, Calibration, and a Trust Envelope for Data→LLM Provenance

> **Draft.** Every number in §5 traces to a versioned artifact (`paper2/corpus/artifacts/*.json`) or a
> re-runnable harness (`paper2/corpus/*.py`); accuracy figures are deterministic and reproducible.
> Numbered citations `[n]` resolve to the References section. `[ARM-B]` marks where ecological-validity
> results land. References marked † need final venue/page confirmation at submission.

---

## Abstract

Modern ML systems increasingly cross a domain boundary that existing observability tools do not: data
prepared by pandas/Spark flows into LLM calls (embeddings, chat completions). *RudriQ's linker* attempts
to recover this **data→LLM lineage** automatically, attaching a confidence-scored provenance edge from an
upstream data artifact to each LLM call. The central question for an audit tool is not whether such a
linker *exists* but whether its links are *correct*, and that has gone unmeasured, because cross-domain
ground truth is hard to establish.

We contribute a **construction-based benchmark** that makes ground truth tractable: pipelines in which we
assign every node identity, so the true data→LLM edge set is known *by construction*, paired with an
adversarial arm built to make the linker fail. **All findings below are established on this synthetic and
adversarial corpus; ecological validation on real third-party pipelines is in progress** `[ARM-B]`, and we
mark throughout which quantities are portable linker properties and which are corpus-composition-dependent.

On the constructed corpus we find: **(1)** the linker's per-link confidence scale is **calibrated in
structure**: measured precision falls monotonically as stated confidence falls, a *structural* property
(as distinct from the specific per-band precision *values*, which are composition-dependent); this is the
keystone result. **(2)** the exact-match strategy (object-identity) is
near-perfectly precise (precision 1.000, 95% CI [0.987, 1.000], n=300), and **no false positives were
observed on representative new data** (0 of 200; 95% CI upper bound below 2%). **(3)** the false-positive
surface comprises **two bounded, characterized mechanisms** that respect their boundaries, yielding a
confidence-threshold *deployment guideline*. **(4)** common-path instrumentation overhead is **below 0.1%
of a single LLM call**, with two characterized and mitigable fallback costs.

This is **Paper 2** of the RudriQ/AutoLineage line. **AutoLineage (Paper 1)** contributes the import-time
*capture mechanism* [6]; **this paper measures the cross-domain *linker's
accuracy and calibration*.** Same system, different contribution.

---

## 1. Introduction

Data lineage and ML observability tools trace operations *within* a domain (DataFrame transformations,
training steps, or, separately, LLM call traces) but not *across* the boundary where prepared data enters
a language model. In a retrieval-augmented generation (RAG) or ML+LLM pipeline, the provenance question a
practitioner and an auditor both ask is: *which upstream data artifact produced the input to this LLM
call?* Answering it requires linking a data-domain node to an LLM-domain node, a **cross-domain lineage
link** that no general-purpose lineage system recovers automatically [1, 4, 7].

RudriQ's linker recovers these links at import time, with no user annotation, by maintaining process-local
registries of tracked objects and correlating each LLM call's input against them. It applies an *ordered*
set of matching strategies and attaches a fixed per-strategy confidence to each emitted edge.

The problem is **measurement**, not mechanism. A mechanism paper is accepted if the idea is novel and
works; an empirical paper is accepted only if the *evaluation methodology is sound*. For a cross-domain
linker, soundness turns on a single hard question: **what is ground truth, and how do you establish it?**
"Correct" is exactly the thing that is difficult to pin down for cross-domain lineage, and a flawed
ground-truth definition invalidates every downstream number.

Our contributions:

1. A **construction-based benchmark methodology** in which ground truth is known *by construction* (we
   assign node identities and choose which data feeds which LLM call), with a deliberately **adversarial
   negative arm** that exists to make the linker fail and to supply the false-positive-rate denominator.
2. A **scoring harness** that scores production (first-match-wins) and per-strategy *shadow* outcomes
   against ground truth, computing precision, recall, FPR, and **confidence calibration** with Wilson
   intervals, and that *structurally enforces* the honesty distinctions an empirical reviewer checks.
3. Four empirical findings (calibration, per-strategy accuracy, a characterized FP surface, overhead),
   each reported with its uncertainty and an explicit portable-vs-composition-dependent split.
4. A **trust-envelope map**: the failure surface connected to the calibration result as an actionable
   confidence-threshold deployment guideline.

---

## 2. Related Work

**Data lineage / provenance.** Open standards and platforms such as OpenLineage [1] and DataHub [2]
capture lineage *within the data domain*: datasets, jobs, and column-level edges (DataHub derives the
latter by SQL parsing), on foundations formalized by Cheney et al. [3]. These establish lineage *graphs*,
and some even *infer* edges, but they remain inside the data domain and report lineage as a catalog
feature; none crosses the data→LLM boundary or *measures* an inferred linker's accuracy and calibration
against constructed ground truth with adversarial negatives.

**ML experiment + pipeline tracking.** MLflow [4] and Weights & Biases [5] capture runs, parameters,
metrics, and artifacts via **explicit instrumentation**; lineage is recorded when the developer logs it.
AutoLineage [6] (Paper 1) removes the instrumentation via import-time hooking, but, like the above, is a
*capture* contribution. None evaluates a cross-domain linker's accuracy; capture and measured-inference are
different contributions.

**LLM observability / tracing.** OpenLLMetry/Traceloop [7], LangSmith [8], Langfuse [9], and Arize Phoenix
[10] trace LLM calls (model, tokens, latency, prompt/completion), typically over OpenTelemetry. They treat
the *data that produced the prompt* as out of scope; the data→LLM edge our linker recovers is precisely the
gap these tools leave.

**RAG / retrieval evaluation.** RAGAS [11], ARES [12], and TruLens [13] evaluate *answer and retrieval
quality*: faithfulness, groundedness, context/answer relevance. This is an **orthogonal axis** to
*provenance-link correctness*: they ask whether the answer is well-grounded, not whether a recovered
data→LLM lineage edge is the right edge.

**Confidence calibration.** Reliability diagrams and empirical calibration evaluation are established
[14, 15]. We apply the same diagnostics (precision conditioned on stated confidence, monotonicity, Wilson
intervals) to a *rule-based* linker whose confidence is **assigned per strategy** rather than learned,
which makes calibration a falsifiable claim about a fixed five-value scale.

**Closest prior work, and the boundary of our novelty.** Two recent works sit nearest. Yin et al. [16]
benchmark language models on extracting *schema lineage* from multilingual pipeline scripts, scoring with
their Schema Lineage Composite Evaluation (SLiCE) metric, which combines structural correctness with semantic fidelity,
over four components (source schemas, source tables, transformation logic, aggregation operations). It
shares our *measure-the-extraction* stance but differs on the two axes that define our contribution: it
stays **within the data domain** (schema-to-schema lineage in code, not data→LLM-call edges), and it
benchmarks **LLMs as extractors**, not a *rule-based* linker's precision/recall and **confidence
calibration** under adversarial negatives. Zhao [17] (XProv), a vision paper, proposes a common
parameterized representation for **cross-library** lineage in data-science workflows, linking materialized
lineage graphs to abstracted logical patterns; it is the closest work on *crossing boundaries*, but the
boundary it crosses is library-to-library *within* the data domain, it is architectural rather than an
evaluated system, and it neither links to LLM calls nor measures accuracy. We therefore do **not** claim
that lineage extraction is unmeasured (it is, in-domain [16]) nor that cross-boundary lineage is
unconsidered (it is, across libraries [17]). We claim novelty for a specific, and to our knowledge
unoccupied, combination: crossing the **data→LLM-call** boundary, with a **rule-based linker** whose edges
are **measured for accuracy and calibration** against a **construction-based benchmark with adversarial
negatives**.

---

## 3. The Linker Under Test

The linker's public entry point is `correlate(llm_input, span_attributes) → (parent_id, method,
confidence)`. It applies an **ordered** tuple of strategies and returns the **first** non-`None` match
(*first-match-wins*). The strategies and their **fixed** confidences:

| Order | Strategy | Confidence | Mechanism |
|---|---|---|---|
| 1 | `object_identity` | 1.0 / 0.95 | Python `id()` match against the in-process object registry (1.0 for the input itself; 0.95 for a registered element of a list/tuple input). |
| 2 | `content_hash` | 0.8 | SHA-256 of the whole input against persisted node hashes (storage-backed). |
| 3 | `substring` | 0.7 / 0.5 | Full-containment match (length-floored at 20 chars) between input strings and registered upstream strings. |
| 4 | `name_match` | 0.5 | Heuristic by variable/column name; **a stub returning `None`** in the evaluated version. |

Two properties drive the evaluation. **First, confidence is assigned per strategy/branch, never learned**:
there are exactly five discrete values {1.0, 0.95, 0.8, 0.7, 0.5}, which makes precision-conditioned-on-
confidence a *falsifiable* calibration claim. **Second, first-match-wins makes cost and behavior bimodal**:
a high-precedence strategy can pre-empt a lower one, and the expensive strategies run only on a miss of the
cheap one (§5.4, §5.5).

---

## 4. Methodology

### 4.1 The trichotomy (the crux)

Every candidate LLM call is labeled into one of three ground-truth classes, because **a non-link is not
always an error**:

- **LINKED-TRUE**: the input derives from a specific registered upstream node; the linker *should* emit
  the correct link. Contributes to recall.
- **TRUE-NEGATIVE**: the input is genuinely new data with no in-scope provenance (e.g. a fresh user
  query); the linker *should* stay silent. A fire here is a false positive; this pool is the FPR
  denominator.
- **OUT-OF-SCOPE**: provenance exists but outside what the linker can observe; excluded from recall *and*
  from the FPR pool, reported only as a coverage gap.

Recall is computed over LINKED-TRUE only; FPR over TRUE-NEGATIVE only. Conflating "should link and didn't"
with "correctly didn't link" would make recall arbitrary; this is the single most important methodological
decision in the benchmark.

### 4.2 Three arms

- **Arm A (synthetic, control).** Pipelines where we assign every `node_id`, so the true edge set is known
  by construction. Each LINKED-TRUE link is tagged with the **mechanism class** it should be caught by:
  *verbatim-object, element-of-collection, exact-copy, templated-containment, paraphrased-derived*, plus
  deliberate **strategy-collision** pipelines (one input satisfying two strategies at *different* true
  parents) to make pre-emption measurable.
- **Arm C (adversarial, negative).** Built to make the linker fail and to supply the TRUE-NEGATIVE pool:
  clean true-negatives, near-misses just below threshold, generic-content substring collisions, and
  content-hash aliasing traps.
- **Arm B (real, ecological).** Annotated third-party pipelines, with inter-annotator agreement (κ). **In
  progress** `[ARM-B]`; supplies the representative input distribution and the pre-emption *correctness*
  evidence (§5.5, §6).

### 4.3 Registered-vs-inferred and the self-proving gate

For object-identity links the upstream object *is* registered (the control). For inferred strategies
(content-hash, substring) the dataflow genuinely exists but the matching object is **not** identity-
registered; the linker must infer it. A per-pipeline **construction-consistency gate** runs each pipeline
through the real linker and asserts the *fired* strategy matches the *constructed* mechanism class (a
substring pipeline where object-identity fired is a corpus bug, hard-failed at generation). Assembly
**refuses to emit a corpus containing a failing gate**, so a shipped corpus provably tests what it claims.

### 4.4 Determinism and pre-registration

Corpora are seeded and serialize **byte-identically** across runs (verified by diff); LLM HTTP is mocked,
so the linker always sees the real Python object.

**Pre-registered vs. implementation-resolved (stated plainly, because the distinction is the point).** We
pre-registered the methodology before any corpus existed (`methodology.md` §12/§13): the trichotomy and its
scoring roles, the three-arm design, the κ protocol, the symmetric-collision decision, and the
mechanism-class tagging. Implementation then **forced refinements we did not foresee**, recorded in full and
dated in `methodology.md` §14–15 and explicitly **not** backdated into the pre-registration. Three reshape
how results are *scored*: **(S1)** a wrong-parent link is simultaneously a precision-error, a recall-miss,
and *not* a false-positive-rate event, so precision/recall/FPR are computed over three different
populations; **(S2)** the FPR denominator is the *representative* clean true-negative pool, with adversarial
decoys reported separately (§5.3–5.4); **(S3)** blended precision/FPR are corpus-composition-dependent and
so are not headline numbers. A fourth, surfaced only at scale-up (§14.12), bounds what may be *quoted*:
tightening an interval does not make a composition-dependent value a linker property, so we report the
monotonic *structure* and the composition-robust quantities as findings while treating per-band *values* as
on-corpus, a distinction §5 applies throughout. (The OUT-OF-SCOPE pooling rule of §4.1 is likewise an
implementation-resolved clarification, §14.11.) We state this split in the paper rather than only in the
doc, because presenting implementation-forced decisions as foresight would claim a rigor we did not have;
the honest admission is the credibility gain.

---

## 5. Results

Unless noted, figures are from the harness at scale (Arm A: n=100 per mechanism class; Arm C: 200 clean
true-negatives, 25 per decoy kind; 900 calls total), run against the real linker. Accuracy figures are
deterministic. Intervals are Wilson 95%. **All results below characterize the linker *on the constructed
corpus*; we do not claim they generalize to arbitrary real workloads until ecological validation lands
(§8). Every headline figure should be read with that "on the constructed corpus" tether attached.**

### 5.1 Calibration (the keystone)

Precision conditioned on stated confidence is **monotonic** (precision does not increase as confidence
falls) at N=900:

| Stated confidence | Measured precision | 95% CI | n | Carried by |
|---|---|---|---|---|
| 1.00 | 1.000 | [0.963, 1.000] | 100 | object_identity (verbatim) |
| 0.95 | 1.000 | [0.981, 1.000] | 200 | object_identity (element) + collision |
| 0.80 | 0.800 | [0.721, 0.861] | 125 | content_hash (exact-copy + aliasing) |
| 0.70 | 0.800 | [0.721, 0.861] | 125 | substring (templated + collision) |

On the constructed corpus, the confidence scale is **calibrated in structure**: the high bands show no
observed errors (precision 1.000, lower bound > 0.96); the false-positive surface lands precisely in the
lower bands where the riskier inferred strategies live. This is the assertion→measured-result transformation the paper sets out to
demonstrate.

**Honesty boundary (portable vs. composition-dependent).** The *monotonic structure* (precision not
increasing as confidence falls) is what we report as the calibration finding; the *values* of the lower
bands (0.800) are **corpus-composition-dependent** (0.800 is exactly the 100:25 true-positive :
adversarial-decoy ratio chosen for the corpus) and would become real linker properties only against a
representative input distribution `[ARM-B]`. We observed the structure on this corpus at N=900; we did not
run a composition sweep, so we claim the structure as a *reported finding*, not as composition-invariant.
We therefore do not quote 0.800 as a bare linker property; we quote the *monotonicity*.

### 5.2 Per-strategy accuracy (composition-robust)

| Strategy | Precision | 95% CI | n |
|---|---|---|---|
| `object_identity` | **1.000** | [0.987, 1.000] | 300 |
| `content_hash` | 0.800 | [0.72, 0.86] | 125 |
| `substring` | 0.800 | [0.72, 0.86] | 125 |

`object_identity` precision (>0.98, lower bound) is **composition-robust**: identity matching is exact, so
it produced no observed false positives across the clean/adversarial mix tested. The content-hash and
substring precision *values* inherit the §5.1 composition caveat.

**Per-mechanism recall** (strategy-appropriate; conditional on the mechanism class, hence composition-
robust), N=900:

| Mechanism class | Recall | 95% CI | n |
|---|---|---|---|
| verbatim_object | 1.000 | [0.963, 1.000] | 100 |
| element_of_collection | 1.000 | [0.963, 1.000] | 100 |
| exact_copy | 1.000 | [0.963, 1.000] | 100 |
| templated_containment | 1.000 | [0.963, 1.000] | 100 |
| strategy_collision | 1.000 | [0.963, 1.000] | 100 |
| **paraphrased_derived** | **0.000** | [0.000, 0.037] | 100 |

The catchable mechanism classes are recovered with lower-bound recall >0.96. **Paraphrased-derived is a
recall ceiling, not a misfire** (§5.6).

### 5.3 False-positive rate (representative denominator)

On the constructed corpus, over the **representative clean true-negative pool** (n=200): **FPR = 0.000,
95% CI [0.000, 0.019]**: **no false positives were observed** on genuinely-new data (0 of 200), with a
quotable upper bound below 2%. The adversarial
decoys are deliberately **excluded** from this denominator (§5.4): folding hard-by-construction decoys into
FPR would inflate the headline rate, the precise overclaim our methodology forbids. Two distinct notions of
false positive are kept separate: a *precision-FP* (any incorrect emitted link, including a wrong-parent
link on a positive) versus an *FPR-FP* (a link on a true-negative). A wrong-parent link is a precision-FP
and a recall-miss but is **not** in the FPR pool.

### 5.4 The false-positive surface (Arm C)

Arm C is designed to make the linker fail. It surfaces **two bounded, characterized mechanisms**:

- **content-hash aliasing** (confidence 0.8, *wrong-parent*): two upstream nodes with identical content;
  the linker returns the most-recent, mis-attributing provenance.
- **substring generic-collision** (confidence 0.7, *spurious-link*): a generic ≥20-char phrase incidentally
  fully contained in an unrelated prompt links the call to that phrase.

Critically, the **near-misses stayed silent**: an 18-char below-floor substring and a one-character-off
hash both produced no link. The linker's boundaries (the 20-char floor, exact-hash matching) **hold**; it
fails *only* where it is genuinely vulnerable. A bounded, explained failure surface is a stronger result
than reported perfection, which would read as a rigged adversarial arm.

### 5.5 Pre-emption: two claims, separate evidence

First-match-wins means a high-precedence strategy can mask a lower one. We report this as **two distinct
claims with two distinct evidence bases, never fused**:

> **Claim 1: behavioral masking (measured, Arm A, high-N).** In strategy-collision pipelines, production
> masks the substring-parent in favor of the object-identity parent at a rate of **1.000** (95% CI [0.963,
> 1.000], n=100 collisions at N=900). This is a *behavioral* statement (*what gets masked*) and carries
> no claim about which parent is "more correct."

> **Claim 2: correctness cost (deferred, Arm B, no synthetic evidence by design).** Whether pre-emption
> ever masks the *better* link is a correctness judgment that synthetic construction cannot honestly make
> (in our symmetric collisions both edges are genuinely true). The harness's sharpest check, cases where
> production scored *wrong* while a shadow strategy would have scored *right*, returns **0** at N=900,
> empirically confirming the symmetric construction produced symmetric outcomes. The correctness claim
> therefore has **zero synthetic evidence by design** and awaits annotated real pipelines `[ARM-B]`.

We deliberately keep these in separate paragraphs: fusing them into "pre-emption causes errors" would
inherit Claim 2's (absent) evidence onto Claim 1's broad behavioral result.

### 5.6 Overhead (distributional; the bimodal insight)

Overhead is the only *non-deterministic* measurement; we report it distributionally (warmup discarded,
median + tail, noise floor measured at ~300ns, GC disabled). **Environment-specific** (Win11, Python
3.12.4, 8-core Intel); the **portable findings are the ratios and scaling shapes**, not the absolute µs.

| Strategy | Scaling shape (10→10⁴ entries) | Median @1k | p99 @1k |
|---|---|---|---|
| `object_identity` | **flat O(1)** | 3.2µs | 19.7µs |
| `content_hash` | flat, but **≈1000× identity** (storage I/O) | 3.50ms | 6.37ms |
| `substring` | **O(n)**: 11µs → 13.75ms | 1.15ms | 2.15ms |

Because the strategies are first-match-wins, the expensive ones run **only on object-identity miss**. The
**common-path** end-to-end cost (register + correlate, identity hit) is **9.5µs ≈ 0.095% of a ~10ms
embeddings call (0.0019% of a ~500ms chat call)**. The two fallback costs are *characterized and
mitigable*: content-hash's per-query storage I/O can be batched/deferred; substring's O(n) scan is an
inverted-index roadmap item (the strategy to watch as the registry approaches its cap). Accuracy says the
linker is *correct*; overhead says it is *affordable in the common case, with bounded fallback costs*.

### 5.7 The trust-envelope map (the applied payoff)

Because the FP mechanisms concentrate in the lower confidence bands, **a confidence threshold is a
precision/recall knob with known content.** This table is from the **same N=900 corpus** as §5.1–5.4
(`trust_map.json`); its *Precision* **and** *Recall* are both **cumulative**, computed over all links (resp. all
positives) at confidence ≥ the threshold, a different cut from §5.1's *per-band* precision and §5.2's
*per-mechanism* recall. So both columns differ from those sections by construction, not by inconsistency:
the cumulative ≥0.80 precision (0.941) versus the per-band 0.80 precision (0.800); and the cumulative ≥1.00
recall (0.160 = verbatim's 100 correct of all 625 positives) versus verbatim's per-mechanism recall (1.000
= 100 of 100 verbatim positives). Different denominators, distinct quantities.

| Threshold | Precision (cumulative) | Recall (cumulative) | Admits FP modes |
|---|---|---|---|
| conf ≥ 1.00 | 1.000 [0.96, 1.00] | 0.160 [0.13, 0.19] | (none) |
| conf ≥ 0.95 | 1.000 [0.99, 1.00] | 0.480 [0.44, 0.52] | (none) |
| conf ≥ 0.80 | 0.941 [0.91, 0.96] | 0.640 [0.60, 0.68] | hash-aliasing |
| conf ≥ 0.70 | 0.909 [0.88, 0.93] | 0.800 [0.77, 0.83] | hash-aliasing, substring-collision |

- **FP-intolerant deployments (audit/regulated): threshold at confidence ≥ 0.95** → precision 1.000 (95% CI
  [0.99, 1.00]), excluding *both* characterized FP modes.
- **Recall-maximizing deployments: ≥ 0.70** → recall 0.800, admitting the two (now named) FP modes.
- **Structural blind spot at any threshold:** paraphrased-derived links are unrecoverable until
  `name_match` is implemented.

The **threshold→FP-mode mapping** (which modes appear at which threshold) is **composition-robust**: fixed
by which confidence band each mechanism occupies, and unchanged from the smaller pilot run; only the values
shift. The *recall values* remain composition-dependent (they track the positive/negative mix) and are read
as on-corpus, not as portable rates.

---

## 6. Threats to Validity

Pre-registered (`methodology.md` §9) and restated honestly:

- **T1: constructing the test to flatter the system.** Mitigated by Arm C (adversarial, by design) and by
  reporting per-arm; we refuse a single blended headline figure.
- **T2: ground-truth annotation error (Arm B).** A frozen annotation guide, two blind annotators, κ
  reported, third-party adjudication; pre-committed κ thresholds and the response to missing them.
- **T3: self-authored pipelines.** Arm B requires ≥2 third-party pipelines we did not write `[ARM-B]`.
- **Small per-band N / composition-dependent values.** The per-band precision *values* for inferred
  strategies are corpus-composition-dependent; only the monotonic structure, object-identity precision,
  per-mechanism recall, and the representative FPR are quotable as bare linker properties (§5.1).
- **Synthetic-only ecological validity.** The headline accuracy is established on constructed pipelines;
  real-pipeline validation is in progress `[ARM-B]` and is required before the inferred-strategy precision
  *values* are claimed for real workloads.
- **Single-environment overhead.** Absolute latencies are environment-specific; the ratios (content-hash
  ≈1000× identity) and shapes (identity flat, substring O(n)) are portable.

Stating these *is* the credibility: an evaluation that reports 1.0 everywhere with no stated limits invites
the suspicion it earns.

---

## 7. Discussion: Deployment Guidance

The trust-envelope map (§5.7) converts the calibration finding into an operational knob. A deployer chooses
a confidence threshold knowing *exactly* which failure modes each setting admits: an FP-intolerant audit
deployment thresholds at ≥0.95 and accepts reduced recall; a recall-maximizing deployment accepts the two
named, bounded FP modes. The overhead profile (§5.6) supports leaving instrumentation **on in production**:
the common path costs <0.1% of an LLM call, and the two fallback costs are mitigable by batching
(content-hash) and indexing (substring). Together these empirically back leaving instrumentation
**always-on** for the path that dominates, with its costs and limits named rather than hidden. (The
*zero-egress* property, meaning the linker runs in-process and no data leaves the host, is **architectural**, not
a result of this study; we note it as deployment context, not as something measured here.)

---

## 8. Limitations and Future Work

- **Ecological validity (Arm B).** The benchmark's most important next step: ≥2 annotated third-party
  pipelines to (a) supply the representative input distribution that turns the inferred-strategy precision
  *values* into real numbers, and (b) furnish the pre-emption *correctness* evidence currently absent by
  design (§5.5). Arm B slots into a proven scoring contract.
- **Substring scaling.** O(n) registry scan (13.75ms at 10⁴ entries); an inverted index would flatten it.
- **Recall ceiling.** Paraphrased/derived links are unrecoverable until the `name_match` strategy ships;
  this is a roadmap item, not a bug.
- **One link per call (multi-parent provenance).** The linker emits a single best link per LLM call, so for
  a call whose input genuinely derives from *multiple* upstream artifacts (e.g. a RAG prompt assembled from
  several retrieved chunks) it recovers *a* correct parent but not the *complete* parent set. We report
  **link-level recall** (did it find a correct parent?) as the headline; **provenance-completeness** (the
  fraction of all true parents recovered) is structurally capped for multi-parent calls and is the natural
  target of a future multi-link mode.
- **Causal root-cause analysis** over the recovered lineage is a v1.0+ track beyond this paper's accuracy
  scope.

---

## 9. Conclusion

We measured a cross-domain data→LLM lineage linker with a construction-based benchmark that makes ground
truth tractable and an adversarial arm that makes failure measurable. The linker's confidence scale is
calibrated in structure; its exact-match strategy is near-perfectly precise; it shows no observed false
positives on representative new data; its false-positive surface is two bounded, characterized mechanisms;
and its common-path overhead is below 0.1% of an LLM call. The result is not a claim of perfection but a **map of
when the linker should and should not be trusted**, which, for an audit tool, is the more useful and the
more credible contribution. Real-pipeline ecological validation is the next step `[ARM-B]`.

---

### Reproducibility

Methodology and all decisions: `paper2/methodology.md` (§1–15). Corpus generators, scoring harness,
overhead harness, and trust-map analysis: `paper2/corpus/*.py`, released at
<https://github.com/kishanraj41/rudriq> (package `rudriq` on PyPI). The scored N=900 corpus behind every
figure in §5.1–5.5: `paper2/corpus/artifacts/scale_up_n900.json`, regenerated by `python -m
paper2.corpus.scale_up`; the §5.7 sweep: `trust_map.json`. Smaller per-arm verification corpora
(`{arm_a,arm_c}.json`) ship alongside for the construction-consistency gate (§4.3). Accuracy results are
deterministic and re-run against the real linker; overhead is distributional and environment-specific (§5.6).

---

## References

> All references below were verified against a primary source (arXiv/ACL Anthology, official repo, or
> vendor docs). Items marked † need final author/venue/page confirmation at submission.

[1] OpenLineage: An Open Standard for Lineage Metadata Collection. LF AI & Data Foundation, 2021–.
https://openlineage.io

[2] DataHub: A Metadata Platform for the Modern Data Stack. LinkedIn / Acryl Data (open source), 2020–.
https://datahub.com

[3] J. Cheney, L. Chiticariu, W.-C. Tan. "Provenance in Databases: Why, How, and Where." *Foundations and
Trends in Databases*, 1(4):379–474, 2009.

[4] M. Zaharia, A. Chen, A. Davidson, et al. "Accelerating the Machine Learning Lifecycle with MLflow."
*IEEE Data Engineering Bulletin*, 41(4):39–45, 2018.

[5] Weights & Biases, Inc. Experiment tracking and artifact versioning platform. https://wandb.ai

[6] K. R. Vandhavasi (VG). "AutoLineage: Operation-Level Data Lineage for Python ML Pipelines via
Import-Time Hooking." SSRN preprint, 2026. abstract_id=6683825. *(Paper 1.)*

[7] Traceloop. OpenLLMetry: OpenTelemetry-based observability for LLM applications. Apache 2.0.
https://github.com/traceloop/openllmetry

[8] LangChain, Inc. LangSmith: LLM and agent observability and evaluation.
https://docs.langchain.com/langsmith

[9] Langfuse. Open-source LLM engineering platform (tracing, evaluation, prompt management).
https://langfuse.com

[10] Arize AI. Phoenix: open-source AI observability and evaluation (OpenInference / OpenTelemetry).
https://phoenix.arize.com

[11] S. Es, J. James, L. Espinosa-Anke, S. Schockaert. "RAGAS: Automated Evaluation of Retrieval Augmented
Generation." *EACL 2024* (System Demonstrations). arXiv:2309.15217.

[12] J. Saad-Falcon, O. Khattab, C. Potts, M. Zaharia. "ARES: An Automated Evaluation Framework for
Retrieval-Augmented Generation Systems." *NAACL 2024* (Long Papers). arXiv:2311.09476.

[13] TruLens: feedback functions for LLM/RAG evaluation (the "RAG triad"). TruEra / Snowflake.
https://www.trulens.org

[14] A. Niculescu-Mizil, R. Caruana. "Predicting Good Probabilities with Supervised Learning." *ICML 2005*,
pp. 625–632.

[15] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. "On Calibration of Modern Neural Networks." *ICML 2017*,
PMLR 70:1321–1330. arXiv:1706.04599.

[16] J. Yin, Y.-W. Chen, M.-L. Lee, X. Liu. "Schema Lineage Extraction at Scale: Multilingual Pipelines,
Composite Evaluation, and Language-Model Benchmarks." arXiv:2508.07179, 2025. *(Verified against the arXiv
abstract; closest prior work, distinguished in §2.)*

[17] J. Zhao. "Learning Lineage Constraints for Data Science Operations" (XProv). Vision paper,
arXiv:2506.18252, 2025. *(Verified against the arXiv abstract; nearest cross-boundary work, distinguished
in §2.)*
