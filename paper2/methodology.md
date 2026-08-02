# Paper 2 — Cross-Domain Linker Accuracy Benchmark: Methodology (Phase 1)

> **Status:** design, pre-implementation. This document defines *what* we measure and
> *how we establish ground truth* before any harness code is written. Everything in
> Phases 2–5 is downstream of the decisions here; a flaw in this document invalidates
> the experiment no matter how clean the harness is.
>
> **Scope decision (2026-06-13):** full publication. This methodology is written to
> survive adversarial peer review, not just to produce a pitch number.

---

## 0. The one-sentence claim under test

> *RudriQ's cross-domain linker correctly identifies data→LLM lineage links in
> instrumented Python pipelines, with measurable and well-characterised precision,
> recall, and false-positive behaviour — and its per-link confidence scores are
> meaningful, not arbitrary.*

Paper 1 (AutoLineage) was a **mechanism** paper: novel + works ⇒ accept. Paper 2 is an
**empirical** paper: it is accepted only if the *measurement methodology* is sound. A
reviewer will not reject the linker; they will reject our *measurement of it*. So the
adversary we design against is **a skeptical reviewer of the evaluation**, not a skeptical
user of the tool.

---

## 1. The system under test (grounded in real code)

Not a reconstruction — these are the exact entry points the harness calls.

| Component | Location | What it does |
|---|---|---|
| `correlate(llm_input, span_attributes, strategies)` → `(parent_id, method, confidence)` | [linker.py:465](../rudriq/linker.py#L465) | **First-match-wins** over an *ordered* strategy tuple. Returns the first strategy that yields a non-`None` parent. |
| `link_by_object_identity` | [linker.py:244](../rudriq/linker.py#L244) | `id()` against `_object_registry`. **1.0** if the input itself is registered; **0.95** if any element of a list/tuple input is registered (the slice case). |
| `link_by_content_hash` | [linker.py:286](../rudriq/linker.py#L286) | SHA-256 of the *whole* input vs persisted node hashes. **0.8**. |
| `link_by_substring` | [linker.py:350](../rudriq/linker.py#L350) | Either-direction containment between input strings and registered upstream strings. **0.7** if combined score ≥ 0.6, else **0.5**. |
| `link_by_name_match` | [linker.py:449](../rudriq/linker.py#L449) | **Stub** — always returns `None` in v0.0.2. |
| `register_object_identity(obj, node_id)` | [linker.py:181](../rudriq/linker.py#L181) | Populates both registries; the harness uses this to *assign* node_ids and thereby define ground truth. |
| `TraceEdge(parent_id, child_id, kind, confidence, link_method, metadata)` | [schema.py:205](../rudriq/core/schema.py#L205) | The persisted link. `kind == LINEAGE_LINK` is the cross-domain edge we evaluate. |

Three properties of this code that **drive the entire methodology**:

1. **First-match-wins + fixed ordering.** `correlate` stops at the first strategy that
   fires. The order is `object_identity → content_hash → substring → name_match`. This is
   a *design choice that is itself measurable*: a higher-precision strategy placed first
   can pre-empt a lower one that would have linked to a *different* (possibly more correct)
   parent. We must therefore evaluate **both** the production `correlate` (ordered, the
   product) **and** each strategy in isolation (the science). See §6.

2. **Confidence is fixed per strategy/branch — never learned.** There are exactly five
   confidence values in the system: {1.0, 0.95, 0.8, 0.7, 0.5}. This is a gift: confidence
   is a *discrete, pre-declared* variable, so **precision conditioned on confidence band**
   is directly testable and a calibration claim is falsifiable (§7). The substring 0.5/0.7
   split is where false positives will concentrate and is the number a reviewer scrutinises
   hardest.

3. **Content-hash is whole-object-exact.** `compute_content_hash`
   ([schema.py:291](../rudriq/core/schema.py#L291)) is type-tagged and recurses over the
   entire value. It fires only when the *entire* LLM input hash-equals a registered node —
   i.e. you embedded *exactly* the registered object, unmodified. Wrapping upstream data in
   a prompt template (`f"Context:\n{context}..."`) changes the hash. Consequence:
   content-hash precision is ≈1.0 by construction (SHA-256 collision is not a real-world
   event) but its **recall is structurally bounded** to the "passed verbatim" case. The
   benchmark must report this per-strategy, never blended into one aggregate number.

---

## 2. Metrics — precise definitions

Let the universe be the set of **(upstream_node, llm_call)** *candidate pairs* in a
benchmark pipeline. For each pipeline we have, by construction or annotation, a
**ground-truth link set** `G` (§4 defines how). The linker, run over the pipeline, emits a
**predicted link set** `P` (each predicted link carries `method` and `confidence`).

A predicted link `(u, c)` is **correct** iff `(u, c) ∈ G` *and* it satisfies the link
correctness definition in §3 (a link to the *right* parent, not merely *a* parent).

```
TP = |{ p ∈ P : p is correct }|
FP = |{ p ∈ P : p is incorrect }|        # linked, but to the wrong/no true parent
FN = |{ g ∈ G : no correct p links it }| # should-have-linked, didn't

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2·P·R / (P + R)
FPR (audit-critical) = FP / (FP + TN)
```

where **TN** = candidate pairs that are *truly unrelated* and that the linker *correctly
left unlinked* (§4.3 builds these deliberately — without a designed negative arm, TN is
undefined and FPR is unreportable).

**Why FP is weighted heavily.** RudriQ is an *audit* tool. A wrong link asserts a data
provenance that does not exist — in a regulated setting (the healthcare positioning), that
is worse than a missing link, which merely under-claims. We therefore report FPR as a
first-class metric and, in analysis, treat a single high-confidence false positive as more
damaging than several false negatives.

**Reported breakdowns (non-negotiable for the empirical core):**

- Overall P / R / F1 / FPR for the production `correlate` (ordered).
- **Per-strategy** P / R / FPR (object_identity, content_hash, substring), each evaluated
  in isolation (§6).
- **Per-confidence-band** precision: precision among links emitted at each of
  {1.0, 0.95, 0.8, 0.7, 0.5} (§7). This is the calibration result.
- Per-pipeline-archetype P / R (§4) so we can say *where* it works, not just *whether*.

---

## 3. What counts as a "correct" link — the definition that does the work

This is the subtle core. Three problems must be resolved *before* counting anything.

### 3.1 A non-link is not always a false negative

In [realistic_rag_pipeline.py:230](../examples/realistic_rag_pipeline.py#L230), query
embeddings call `client.embeddings.create(input=query)` where
`query = f"Tell me about topic {i}"` — a string **created at query time, derived from no
registered upstream artifact**. The linker correctly emits no link. Counting that as a
false negative would *punish correct behaviour* and make recall meaningless.

**Resolution — every candidate LLM call is ground-truth-labelled into one of three classes:**

| Class | Meaning | Linker should | Counts toward |
|---|---|---|---|
| **LINKED-TRUE** | The call's input *is derived from* a specific registered upstream node | emit the correct link | TP if right, FN if missed, FP if wrong parent |
| **TRUE-NEGATIVE** | The call's input is *genuinely new data* (a fresh user query, a constant prompt) with no upstream provenance in-scope | emit *no* link | TN if silent, FP if it links anything |
| **OUT-OF-SCOPE** | Provenance exists but outside what RudriQ can see (data read by a library we don't hook, env vars, etc.) | excluded from G entirely | neither — reported separately as coverage gaps |

Recall is computed over **LINKED-TRUE only**. FPR is computed using **TRUE-NEGATIVE** as
the negative pool. This separation is the single most important methodological decision in
the paper and the first thing a reviewer will check.

### 3.2 "Right parent," not "a parent" — provenance is a chain

A query's prompt may contain *retrieved documents* (true parent = the corpus chunk) while
the *query string itself* is new. If the linker links the prompt to the corpus chunk:
correct. If it links the prompt to, say, an unrelated earlier document that happens to
share boilerplate ("Answer based on the provided context"): a false positive, even though
*a* link was warranted. Correctness is **identity of the parent node**, compared against
`G`'s designated parent for that call — not the mere existence of any link.

**Multi-parent calls.** A RAG prompt with 5 retrieved chunks has 5 true parents. The
linker, being first-match-wins, emits **at most one** link per call. We therefore define,
per LINKED-TRUE call, a **set** of acceptable true parents `G(c) ⊆ nodes`, and score:

- **TP** if the single emitted link ∈ `G(c)`.
- **Partial-recall metric (secondary):** fraction of `G(c)` covered = 1/|G(c)| when one
  correct link is emitted. We report a **link-level recall** (did it find *a* true parent?)
  and a **provenance-completeness** number (of all true parents, how many surfaced?) — the
  latter honestly exposes that a one-link-per-call linker cannot fully reconstruct
  many-to-one provenance. Hiding this would be the kind of omission that sinks an empirical
  paper at review.

### 3.3 Confidence-as-correctness ambiguity

Is a confidence-0.5 substring link that is *technically* real but *practically* useless a
TP or FP? **Decision:** correctness is binary on *identity* and independent of confidence.
A real link at 0.5 is a TP. Confidence is then evaluated *separately* as a calibration
question (§7): "of links emitted at 0.5, what fraction are TP?" This cleanly separates
"did it find the right edge" from "did it know how sure to be," which are two distinct
claims the paper makes.

---

## 4. The corpus — how ground truth is established

Three arms, in increasing ecological validity and decreasing control. A strong empirical
paper uses **all three**; each answers a reviewer objection the others cannot.

### 4.1 Arm A — Synthetic pipelines with known lineage (control)

Construct pipelines where we *assign* every `node_id` via `register_object_identity` and
*choose* exactly which registered object feeds which LLM call. Ground truth `G` is known
**by construction** — there is no annotation error.

These are **parameterised generators**, not hand-written scripts, spanning the dimensions
that stress each strategy:

- **Pass-shape:** verbatim object (hits object_identity 1.0) · slice/batch (0.95) ·
  exact-copy-different-object (content_hash 0.8) · embedded-in-template (substring) ·
  paraphrased/truncated (should *miss* — tests the FN frontier).
- **Input modality:** single `str` · `list[str]` batch embeddings · chat `messages` dicts ·
  multimodal content parts (exercises `_extract_llm_input_strings`,
  [linker.py:308](../rudriq/linker.py#L308)).
- **Distractor density:** N unrelated registered docs sharing boilerplate, to provoke
  substring false positives at controlled rates.
- **Chain depth:** data → transform → transform → LLM, to test linking through multi-step
  lineage.

> **Reviewer objection answered:** "Synthetic ⇒ unrealistic." → Arm A claims *controlled
> precision/recall on declared mechanisms*, not ecological validity. That is what Arms B/C
> are for. We never over-claim from Arm A alone.

### 4.2 Arm B — Instrumented real pipelines (ecological validity)

Take **real, third-party RAG/ML code** (e.g. a public LangChain or LlamaIndex example, a
Kaggle notebook) and instrument it. Ground truth here is **manually annotated**: a human
traces, for each LLM call, which upstream artifact its input derives from.

The existing [realistic_rag_pipeline.py](../examples/realistic_rag_pipeline.py) is a seed —
but it is *our own* code and so cannot be the only Arm B source (see threat T3). We need
≥2 pipelines we did **not** write, annotated independently.

**Annotation protocol (to make manual ground truth defensible):**

- A written **annotation guide** operationalising §3's three classes and the "right parent"
  rule, frozen before annotation begins.
- **Two independent annotators**, blind to the linker's output. Disagreements adjudicated
  by a third; **inter-annotator agreement (Cohen's κ) reported.** Low κ is itself a finding
  about how ambiguous cross-domain lineage is.
- Annotators label *candidate pairs*, never see `P` — so annotation cannot be contaminated
  by what the linker happened to do.

### 4.3 Arm C — Adversarial / negative corpus (the anti-flattery arm)

The sharpest reviewer objection is **"you built a benchmark your own linker is guaranteed
to win."** Arm C exists to answer it. It is *designed to make the linker fail* and to
populate the TRUE-NEGATIVE pool (without which FPR is undefined):

- **Near-miss distractors:** registered docs that share a long boilerplate prefix with an
  LLM prompt but are *not* its true source — designed to trip substring's containment test
  above `min_match_length=20`.
- **Boilerplate collisions:** many prompts sharing system messages ("Answer based on the
  provided context") to test that the 20-char floor actually holds.
- **Genuinely-new inputs:** fresh user queries with no upstream (the TRUE-NEGATIVE
  backbone).
- **Hash-adjacent inputs:** objects that are *almost* equal (one token changed) to confirm
  content_hash does *not* fire on near-matches (precision guard).
- **Aliasing traps:** two distinct upstream docs with identical content — does the linker
  attribute to the right *one*, and is "either" acceptable? (Forces a `G(c)` set decision.)

Arm C is where we *expect* and *want* to see FPs and FNs. A benchmark that reports 1.0
across all three arms is not a strong result — it is a sign the negative arm was too weak,
and we will say so.

---

## 5. Determinism, isolation, and reproducibility

A reviewer may re-run the harness. It must be **deterministic and hermetic**:

- **No network.** LLM HTTP is mocked exactly as
  [realistic_rag_pipeline.py:77](../examples/realistic_rag_pipeline.py#L77) does (httpx
  `MockTransport`). Embeddings/completions are canned; the linker's input is the *real*
  Python object regardless, so accuracy is unaffected by mock responses.
- **Registry isolation between pipelines.** `clear_object_registry()`
  ([linker.py:222](../rudriq/linker.py#L222)) + `clear_input_registry()` between every
  pipeline, mirroring the `isolated_storage` fixture at
  [test_processor.py:12](../tests/test_processor.py#L12). Cross-pipeline registry leakage
  would manufacture phantom links and silently corrupt FPR.
- **Cache size pinned.** `_set_cache_size_for_tests()` set high enough that LRU eviction
  never fires *within* a benchmark pipeline — otherwise eviction, not matching logic, drives
  recall, confounding the measurement. (A *separate* experiment may vary cache size to
  characterise the eviction frontier, but the headline accuracy numbers run uncapped.)
- **Fixed seeds** for any synthetic generation; the harness records seed + linker version +
  strategy ordering in every result file.
- **`Date.now()`/random caveat:** timestamps are injected, not sampled, so results diff
  cleanly across runs.

---

## 6. Per-strategy vs. production evaluation

We run **two evaluation modes** over the same corpus:

1. **Production mode** — `correlate(...)` with the default ordered strategy tuple. This is
   the number the *product* delivers and the pitch quotes.
2. **Isolation mode** — each strategy invoked alone (`correlate(..., strategies=(s,))`).
   This is the *science*: it yields per-strategy P/R/FPR and exposes **pre-emption** — cases
   where `object_identity` fired first and returned parent X while `substring` alone would
   have returned parent Y. When X ≠ Y and Y ∈ `G(c)`, ordering *cost* a correct link; when
   X ∈ `G(c)` and Y ∉, ordering *saved* one. Quantifying net pre-emption effect is a
   genuine, publishable finding about the first-match-wins design.

---

## 7. Confidence calibration

Because confidence ∈ {1.0, 0.95, 0.8, 0.7, 0.5} is discrete and pre-declared, calibration
is a clean, falsifiable claim:

- For each band, report **empirical precision** = TP / (TP+FP) among links emitted at that
  band, with **Wilson confidence intervals** (counts will be small in some bands; naive
  proportions would over-claim).
- **Monotonicity check:** does empirical precision *increase* with stated confidence? If
  0.95 links are empirically *less* precise than 0.7 links, the confidence scale is
  miscalibrated — an honest, valuable negative result that directly informs whether
  downstream consumers should threshold on confidence.
- **Reliability diagram** (stated vs. empirical precision) as the headline calibration
  figure.

This section turns "we attach a confidence" from an assertion into a measured property —
exactly the assertion→result transformation that is the point of Paper 2.

---

## 8. Runtime overhead (secondary, but expected)

Buyers and reviewers ask "what does instrumentation cost?" Measured separately from
accuracy:

- **Per-operation linker latency**: wall-clock of `correlate` per call, by strategy and by
  registry size (substring is O(candidates × registry × upstream-strings) — its cost grows
  with corpus size and must be characterised, not assumed cheap).
- **End-to-end overhead**: instrumented vs. uninstrumented pipeline runtime at the
  ~250-operation scale the realistic example already exercises.
- Reported as median + tail (p95/p99); tails matter for the substring scan.

This is *less central* than accuracy and is explicitly framed as such in the write-up.

---

## 9. Threats to validity (pre-registered)

Stating these *before* running the experiment is what separates a rigorous empirical paper
from a marketing benchmark.

| # | Threat | Mitigation |
|---|---|---|
| **T1** | **Constructing the test to flatter the system.** Synthetic corpora can be built so the linker can't lose. | Arm C (adversarial/negative) by design; report per-arm so Arm A's high numbers can't mask Arm C; refuse to headline a single blended figure. |
| **T2** | **Ground-truth annotation error** in Arm B. | Frozen annotation guide, two blind annotators, κ reported, third-party adjudication; annotators never see `P`. |
| **T3** | **We wrote the example pipeline**, so it encodes our mental model of how linking works. | Arm B requires ≥2 pipelines authored by *others*; the existing example is labelled as such and not counted as independent. |
| **T4** | **Mock LLM responses** differ from real ones. | Linker input is the *real* Python object; responses never reach the linker. Documented explicitly. |
| **T5** | **Single ordering** evaluated. | Isolation mode (§6) decomposes the ordering effect; we report pre-emption rate. |
| **T6** | **Confidence bands with tiny N** over-claim precision. | Wilson intervals; bands with N below a pre-set floor reported as "insufficient data," not as a point estimate. |
| **T7** | **Provenance is many-to-one**, but the linker emits one link/call. | §3.2 link-level recall *and* provenance-completeness both reported; the limitation is stated, not hidden. |
| **T8** | **In-scope ≠ real-world scope** (OUT-OF-SCOPE class). | Coverage gaps reported as a first-class number, not silently dropped from G. |
| **T9** | **LRU eviction confounds recall.** | Headline runs uncapped; eviction studied as a separate, labelled experiment. |

---

## 10. Experimental procedure (the runbook Phases 2–4 implement)

1. **Generate Arm A** (parameterised synthetic) + **assemble Arm B** (annotated real) +
   **author Arm C** (adversarial). Freeze the corpus; version it.
2. For each pipeline: clear registries → run the pipeline under `RudriQSpanProcessor` →
   collect emitted `TraceEdge`s with kind `LINEAGE_LINK`.
3. Score `P` against `G` using §2/§3 definitions, in **both** production and isolation modes.
4. Emit machine-readable results (per-link verdicts + aggregate metrics + Wilson intervals +
   seed/version provenance) so the run is re-deriveable.
5. **Failure characterisation** (the part reviewers trust most): cluster every FP and FN by
   cause — boilerplate collision, paraphrase miss, pre-emption, eviction, out-of-scope — and
   report counts and representative examples per cluster. The "20 unlinked query embeddings"
   observation is the template: *where* it fails and *why*, honestly.
6. **Overhead** (§8) as a separate harness pass.

---

## 11. What this enables for Phases 2–5

- **Phase 2** builds the corpus generators (Arm A), assembles + annotates Arm B, authors
  Arm C — all against the real `register_object_identity` / `record_llm_input` API.
- **Phase 3** builds the scoring harness implementing §2/§3, production + isolation modes.
- **Phase 4** runs it, produces the calibration + failure-cluster analysis, measures
  overhead.
- **Phase 5** writes it up: related work (lineage/observability eval methodology — OpenLineage,
  data-provenance benchmarks, RAG-eval frameworks), methodology (this document, hardened),
  results, the threats above, honest limitations.

---

## 12. Open methodological questions to pressure-test *now* (before building)

> **RESOLVED 2026-06-13 — all five closed. See §13 for the pre-registered resolutions and
> the two corpus-structure changes (§12.2, §12.4) they force. This section is preserved as
> the original open-questions record.**

These are the decisions I want adversarially challenged before a single generator is
written — getting them wrong is cheap to fix here and expensive to fix after the corpus
exists:

1. **`G(c)` for multi-parent RAG calls — set membership or ranked?** I propose "any true
   parent = TP, completeness tracked separately." Is link-level recall the honest headline,
   or does a reviewer demand provenance-completeness as *the* recall number?
2. **Is content_hash's structural low recall a *limitation* or *by-design correctness*?** I
   lean: report it as a precision/recall *trade-off per strategy*, not a defect — but the
   framing matters for how the abstract reads.
3. **How many Arm B pipelines is "enough" for ecological validity?** Two feels like the
   floor; is it the ceiling a venue will accept?
4. **Should pre-emption (§6) be a headline finding or a footnote?** I think it's genuinely
   novel (most lineage tools don't have ordered multi-strategy linkers) and deserves a
   figure — but it risks diluting the precision/recall story.
5. **κ target for Arm B.** What inter-annotator agreement do we treat as "ground truth is
   reliable enough to publish"? Below some κ, the honest move is to *narrow* the link
   definition until annotators agree — which changes §3.

---

## 13. Resolutions of §12 (pre-registered, 2026-06-13)

These were settled *before* any corpus generator was written. Decisions that the corpus
structure depends on are marked **[CORPUS]**; they amend the named earlier sections.

### 13.1 — Multi-parent recall (§12.1, amends §2 and §3.2)

**Decision.** Two separately-reported numbers; neither is "found *any* parent":

- **Link-level recall (headline)** = of LLM calls with ≥1 true parent, the fraction for
  which the linker emitted a link to **a *correct* parent** (∈ `G(c)`). A wrong-parent link
  is *never* a recall hit — it is an FPR event. This measures the linker against its stated
  one-best-link-per-call objective.
- **Provenance-completeness (secondary)** = of all true (call→parent) edges in the corpus,
  the fraction recovered. Structurally capped at 1/N for an N-parent call; that cap is
  *reported as a strength* — the limitation named before the reviewer names it.
- **The gap** between the two on the multi-parent arm is **promoted to a reported finding**:
  it quantifies how much provenance the one-link design leaves on the table — a concrete
  input to the future "should the linker emit N links?" question.

Frozen framing sentence: *"RudriQ's linker is designed to emit the single highest-confidence
provenance link per call, not exhaustive multi-parent provenance. We report link-level
recall as the primary measure of that objective, and provenance-completeness as a secondary
measure that makes the one-link-per-call architectural limitation explicit and quantified."*

### 13.2 — content_hash recall framing (§12.2, **[CORPUS]** amends §2 and §4.1)

**Decision.** Low aggregate recall for content_hash is a *measurement artifact* of blending
across mechanism classes, not a defect. Recall is decomposed per mechanism class.

- **[CORPUS] Every LINKED-TRUE link is tagged with the mechanism class it should be caught
  by:** `verbatim-object` (→ object_identity 1.0) · `element-of-collection` (→ 0.95) ·
  `exact-copy` (→ content_hash 0.8) · `templated-containment` (→ substring) ·
  `paraphrased-derived` (→ no live strategy; an honest recall-ceiling FN motivating the
  v0.1 name_match work).
- Report **strategy-appropriate recall** (recall over the class a strategy owns; content_hash
  over `exact-copy` ≈ 1.0) *and* **strategy-marginal recall** (contribution to overall
  recall; low for content_hash) — never one blended figure.
- Abstract framing: content_hash = high-precision, narrow-coverage; broader noisier coverage
  delegated to substring. Provable from the per-class table, not asserted.

### 13.3 — Arm B sizing (§12.3, amends §4.2)

**Decision.** Binding constraint is **mechanism coverage, not count.** Rule: **≥2 third-party
pipelines we did not author, collectively exercising all three live strategies on real code**
(target mix: one substring-heavy LangChain/LlamaIndex RAG + one object-identity/slice-heavy
pandas→embeddings ML pipeline). The existing [realistic_rag_pipeline.py] is the non-counted
third (threat T3: self-authored). Count is a function of coverage; ~3–4 third-party is the
ceiling before annotation labor dominates. *(The labor commitment of sourcing + annotating
two third-party pipelines is the one item that is genuinely a project-owner decision, not a
methodological one.)*

### 13.4 — Pre-emption prominence (§12.4, **[CORPUS]** amends §6)

**Decision.** Prominence is **data-driven against a pre-registered threshold**, reported
either way:

- **Pre-emption correctness cost** = fraction of links where first-match ordering produced a
  wrong/missed link that a later strategy would have scored to a correct parent.
- **> 2% ⇒ headline finding** (figure): a novel result on ordered multi-strategy linkers.
- **~0% ⇒ positive footnote**: "first-match ordering costs no measurable correctness."
- **[CORPUS] Arm A must include deliberate strategy-collision pipelines** — calls whose input
  simultaneously satisfies two strategies pointing at *different* true parents (e.g. contains
  a registered object **and** embeds a different doc's text). Without these, pre-emption cost
  is structurally unmeasurable. The harness captures all strategies in shadow per call (§3
  Phase 3 runbook) so the cost is computed, not estimated.

### 13.5 — κ target (§12.5, amends §4.2 and §3)

**Decision.** Threshold **and the response to missing it** pre-committed before any κ is seen:

- **κ ≥ 0.80** → ground truth reliable; publish as-is.
- **0.67 ≤ κ < 0.80** → publishable; κ reported prominently, all disagreements third-
  adjudicated, and the disagreement set reported as a finding about where cross-domain
  lineage is inherently ambiguous.
- **κ < 0.67** → do **not** publish real-arm numbers as ground truth; **narrow §3's "right
  parent" definition until annotators converge, then re-annotate.** (Narrowing likely means
  excluding marginal-confidence ambiguous cases from `G`.)

### 13.6 — Net effect on the Phase 2 build

Two corpus-structure requirements are now frozen and must be honored by the Arm A generator:
1. **Per-link mechanism-class tags** (§13.2).
2. **Strategy-collision / pre-emption-bait pipelines** (§13.4).

Plus the §3.1 trichotomy labels (LINKED-TRUE / TRUE-NEGATIVE / OUT-OF-SCOPE) and seeded
determinism (§5). With these, the generator's output contract is fully specified and Phase 2A
can be built without further methodological ambiguity.

---

## 14. Implementation-resolved decisions (dated 2026-06-13)

> **Integrity boundary — read this first.** The decisions in §12/§13 were **pre-registered**:
> settled *before* any corpus or harness existed, before any result was seen. The decisions in
> **this section were *not* foreseen — they were forced by the implementation** (the Phase 1→3
> build) and resolved as the code and the live linker taught us things the methodology had left
> implicit. They are recorded here, dated, **explicitly distinct from the pre-registration**, and
> must never be backdated into §12/§13. The paper's credibility rests on "we decided the
> methodology before we saw the results"; the honest record therefore separates *what we
> foresaw* (§12/§13) from *what the build taught us* (§14). A reviewer respects that distinction —
> blurring it would undercut the very pre-registration discipline that makes the work strong.
>
> Each item notes how it was discovered, because "discovered by probe/harness, not reasoned" is
> itself the methodological point.

### 14.1 — The linker has two registration surfaces, not one *(discovered: layer-1 probe)*

`link_by_content_hash` reads **persisted storage** (`find_nodes_by_hash`), not the in-process
registries that feed `object_identity`/`substring`. `register_object_identity` never touches
storage. A constructor that knew only the identity surface *cannot* build a content-hash
pipeline. The corpus primitives therefore carry **two surfaces**: identity (`register_object_identity`)
and storage (`save_node` of a `TraceNode` with `content_hash`). Invisible from the API names;
found only by running the real linker.

### 14.2 — Verbatim/element classes are co-satisfiable with substring *(discovered: layer-1 Proof A)*

A registered doc lives in the content registry, so for a verbatim-object input substring
*correctly* co-fires in isolation to the same parent. That is the linker working, not a bug. The
construction-consistency gate for these classes therefore asserts **`production == intended`**, not
isolation-silence. A naive "only the intended strategy is satisfiable" gate would hard-fail correct
pipelines. (Only the storage-exclusive exact-copy class is genuinely single-satisfiable.)

### 14.3 — Element-of-collection requires explicit element registration *(discovered: probe row b2)*

`register_object_identity(L, id)` registers `id(L)` and `id(L[0])` **only**; the 0.95 branch scans
input elements. A non-zero-aligned slice (`L[2:4]`) therefore **silently fails to fire 0.95**. The
constructor registers the **specific target element**, never relies on slice alignment — which is
also more faithful to AutoLineage's real per-element registration. Zero-alignment was a trap that
works by accident of including index 0.

### 14.4 — Paraphrased-derived needs construction-provable derivation-truth *(discovered: self-review of a shipped flaw)*

The first paraphrased-derived constructor produced *generic prose with no source-specific content*
(`pd-0`/`pd-1` were identical) — the silence gate passed on what were effectively two unrelated
strings, an FN claim with no genuine edge behind it. **Silence is necessary but not sufficient.**
Fix: the prompt's distinguishing content is now a deterministic, injective transcoding of the
source datum (digits→words), asserted at construction (datum present in source as digits, in prompt
as words, disjoint across surfaces). Also confirmed: the substring matcher is **full containment**
length-floored at 20 chars, *not* n-gram overlap — so a clean paraphrase trivially avoids firing.
Free-form semantic paraphrase (derivation established by annotation, not construction) is deferred
to Arm B.

### 14.5 — Arm C uses an *inverted* three-regime gate *(discovered: Arm C design)*

Unlike every Arm A gate (fire-where-silence-expected = corpus bug), Arm C has opposite pass
conditions per kind: **CLEAN_TN / NEAR_MISS** (`expect_fire=False`) — a fire is a **measured false
positive to record, never a hard-fail** (hard-failing would delete the FP events FPR exists to
count); **SHOULD_TRIP** (`expect_fire=True`) — a **non-fire is the corpus bug** (weak decoy). For the
first kind a fire is a *result to measure*; for the second a non-fire is a *bug to fix*.

### 14.6 — S1: three distinct error roles for one event *(forced by the harness)*

A wrong-parent link on a positive (e.g. content-hash aliasing) is simultaneously a **precision-FP**
(it drags band/strategy precision), a **recall-FN** (the true edge wasn't recovered), and **not an
FPR event** (FPR counts only links on true-negatives). A naive single 2×2 confusion cell conflates
these. Precision, recall, and FPR are therefore computed over **three different populations**.

### 14.7 — S2: the FPR denominator is the representative clean-TN pool *(forced by the harness)*

FPR is computed over **CLEAN_TN only**; the adversarial decoys (built to be hard) are reported
**separately** as "FP surface exists." Folding decoys into FPR would inflate the headline rate —
the exact overclaim the "0.400 is a construction artifact" caveat warned of. This makes the
inflation *structurally impossible*, which is the right way to honor a caveat.

### 14.8 — S3: blended precision/FPR are composition-dependent, not headline *(forced by the harness)*

A single blended precision (or FPR over a mixed pool) is a function of corpus composition, not a
linker property. The harness flags blended rates as composition-dependent and presents
**per-strategy and per-confidence-band** results as the real findings. *(Scale-up corollary, §14.12
when added: even per-band precision *values* for the inferred strategies remain composition-
dependent; only the monotonic structure, per-mechanism recall, representative FPR, and
object-identity precision are quotable as bare linker properties.)*

### 14.9 — Pre-emption is reported as two claims, structurally separated *(decision, enforced in code)*

The harness emits **two rows, two N's, two evidence types**, never fused: (1) **behavioural** masking
rate (Arm A, high-N synthetic — "precedence masks the substring-parent at rate X", no correctness
judgment); (2) **correctness** rate (Arm B, small-N annotated — "the masked parent was the true
source"). The write-up must keep them separate; building the separation into the harness prevents
the prose from collapsing them.

### 14.10 — The symmetric-default decision is empirically validated *(confirmed by the harness)*

§13.4 chose symmetric Arm A collisions (the finding is *what gets masked*, not *whether the better
link got masked*), arguing that designating the masked substring-parent "more correct" would put a
thumb on the scale (identity-presence is stronger provenance than substring-presence). The harness's
**sharpest check returns 0** — there is *no* synthetic case where production scored wrong while a
shadow strategy would have been right. So the symmetric construction provably produced symmetric
outcomes, and the correctness claim correctly has **zero synthetic evidence** — sending it to Arm B
is the predicted, honest result, not a gap.

### 14.11 — OUT-OF-SCOPE: excluded from G *and* from the FPR pool *(confirmed)*

OUT-OF-SCOPE calls are neither links-that-should-exist (excluded from recall's denominator) nor
representative-new-data-that-should-stay-silent (not in the FPR negative pool). They are a third
category, reported only as a **coverage gap**. With 0 such cases currently it is moot in the numbers,
but the logic is present and correct for scaled corpora.

### 14.12 — Interval tightness ≠ quotability for composition-dependent values *(discovered: scale-up to N=900)*

Scaling Arms A+C (n_per_class=100, clean-TN=200, decoys=25/kind) tightened the intervals enough to
quote the composition-**robust** quantities — `object_identity` precision **1.000 [0.987, 1.000]**
(n=300), representative **FPR 0.000 [0.000, 0.019]** (n=200, "<2%"), catchable-class **recall
[0.963, 1.000]** (n=100 each), paraphrase **recall-ceiling [0.000, 0.037]** ("<4%"), and the
**monotonic structure** (1.00 ≥ 1.00 ≥ 0.80 ≥ 0.80, holding at N=900). The symmetric-construction
check stayed **0** at scale.

But scaling **revealed a trap**: the inferred-strategy per-band precision *values* tightened to
**0.800 [0.721, 0.861]** — and `0.800` is exactly the `100/125` decoy:TP ratio chosen for the
corpus. **A tight interval around a composition-determined number is still not a linker property.**
The honest partition, now empirical not merely argued:

| Quotable as bare linker properties (composition-robust) | NOT quotable without a representative distribution |
|---|---|
| monotonic precision-vs-confidence *structure* | inferred-strategy (`content_hash`, `substring`) per-band precision *values* |
| `object_identity` precision (identity matching is exact) | — these are `(linker × corpus composition)` |
| per-mechanism *recall* (conditional on the class) | → require a **representative input distribution**, |
| representative FPR over the clean-TN pool | → which is **Arm B's** job (ecological validity) |

This is *why* Arm B is methodologically necessary and not optional: the controlled corpus establishes
the calibration **structure** and the composition-robust rates with tight intervals, but the
**absolute precision values for the inferred strategies** are only meaningful against a representative
mix of clean-vs-adversarial inputs — which only annotated real pipelines (Arm B) can supply. Found on
cheap construction-provable work, before Arm B's annotation labor, exactly as the build discipline
intends.

*Performance note:* the harness materialises one isolated DuckDB per pipeline; 900 calls took ~185s.
Acceptable at this scale; the per-pipeline DB connect is the bottleneck if the corpus grows much
larger (an in-memory or pooled-storage path would be the optimisation).

---

## 15. Overhead (§8): measurement methodology and findings (dated 2026-06-13)

> **This is the first *non-deterministic* measurement in Paper 2**, so its rigor is the opposite
> of the byte-identical discipline used everywhere else — it is **distributional**. The
> *methodology of the measurement* is itself part of the result (unlike the deterministic work,
> where "what we measured" was self-evident from the artifact), so it is recorded here while exact.

### 15.1 — Measurement methodology (distributional rigor)

- **Steady-state only:** a disclosed warmup window (200 calls) is discarded before sampling.
- **Median + tail, never the mean alone:** p95/p99 reported because for an always-on audit tool the
  *tail* (worst-case under load) is what a buyer cares about; the mean is the least useful statistic.
- **Noise-floor-aware:** the timing instrument's own cost is measured (an empty callable). On this
  environment the floor is **median ~300ns**; every strategy measured is well above it, so all
  numbers are real signal, not instrument noise. (A strategy at/below the floor would be reported as
  "below our ability to measure" — a strong result, not a gap.)
- **GC disabled during the sampling loop** so a GC pause is not mis-attributed to the operation.
- **Two baselines, two questions:** absolute per-strategy `correlate()` latency (the *mechanism*
  cost) **and** end-to-end marginal cost the linker adds per call (the *buyer-facing* cost).
- **Swept by strategy and by registry size** across four orders of magnitude (10→10⁴) so the scaling
  *shape* is shown, not asserted.

### 15.2 — Portable vs. environment-specific (the load-bearing honesty — read FIRST)

A reviewer correctly distrusts absolute latencies from one machine. So the split is explicit and the
paper **leads with the portable findings**:

| **PORTABLE (the findings)** | **ENVIRONMENT-SPECIFIC (illustrative only)** |
|---|---|
| Strategy **ratios**: `content_hash` ≈ **1000×** the in-process strategies | Absolute values: 9.5µs common-path, 3.5ms content_hash |
| Scaling **shapes**: `object_identity` flat **O(1)**, `substring` **O(n)** | Measured on: Win11, Python 3.12.4 (CPython), 8-core Intel, perf_counter res 100ns |
| The **bimodal** cost structure (below) | noise floor median ~300ns |

Quoting an absolute as if it were a portable claim is the overhead-section equivalent of "don't quote
0.727 as a calibration result." The absolutes are illustrative-on-this-environment; the ratios and
shapes are the claims.

### 15.3 — Findings

- **`object_identity` — O(1), flat, effectively free.** 2.3µs→3.2µs median across 10→10⁴ entries;
  p99 ~20µs. It is the common path (the RAG pattern registers objects), and it is near-free.
- **`content_hash` — flat but ~3.5ms (≈1000× identity).** Flat because the cost is per-query DuckDB
  round-trip, *not* data-scaling. **Characterised, mitigable property, not a buried weakness:**
  in-process strategies dominate latency-sensitive paths; content_hash can be **batched/deferred**,
  and the per-call fresh SQL execute is a concrete optimisation target (prepared-statement / cache).
- **`substring` — O(n) in registry size.** 11µs @10 → **13.75ms @10⁴**, growing ~1200×. A **genuine
  scaling limitation**, reported plainly: at the 50k LRU cap it reaches tens of ms. Roadmap mitigation:
  an inverted index would flatten it. The strategy to watch for large pipelines.
- **The bimodal insight (what makes this an argument, not just a measurement):** because `correlate`
  is first-match-wins, the expensive strategies run **only on object_identity miss**. So the
  *common-path* cost is **9.5µs end-to-end = 0.095% of a ~10ms embeddings call, 0.0019% of a ~500ms
  chat call**, and the millisecond fallback costs are paid only on fallthrough. "Always-on,
  leave-it-on-in-production" is empirically backed for the path that dominates, with the fallback
  costs honestly characterised and mitigable.

The accuracy results said the linker is *correct*; the overhead results say it is *affordable in the
common case, with two characterised and mitigable fallback costs* — a stronger, more credible claim
than a flat "it's fast."
