# MEMPLANT / DREAMGRAFT-001 v3.1 — The Chimaera Garden: Counterfactual Alleles

**Subtitle:** The Stolen Dreams Foundry becomes an experimental ecology of inheritable, editable, recombinant memories.  
**Status:** second independent-review amendments integrated; **proposal only — implementation NOT AUTHORISED**.  
**Date:** 2026-11-15. **Prior reviewed heads:** v1 `c17c2f8ec3891134825b9a18d3db02b7d565c37d`; v2 `fa442a190f8a33a50b0bd7eeaf9c8729e7dcf68c`. **Independent v2 review verdicts:** G0/G1 conditional; DG-2–DG-5 rework. This document is a proposed resolution, not independent reapproval.  
**Location:** techrote/neurounner/MEMPLANT/. **Parent:** [MEMPLANT README](README.md), [concept](CONCEPT.md), [pseudo-embeddings](PSEUDO_EMBEDDINGS.md), [research programme](EXPERIMENTS.md), [reference catalogue](REFERENCES.md).  
**Authority:** documentation revision only; no model training, implementation, schedule, cross-repository work or merge is approved.

> **The ambitious question:** Can a donor-derived, numerically transmitted memory be interrogated and *selectively edited*, can two donor memories cause **joint** consequences, and can a descendant acquire information unavailable to its ancestor—without providing the receiver with the source text?

This is deliberately a playful computational metaphor. No claim is made about consciousness, subjective experience or biological memory. A successful computation does not mean an entire mind, conversation, reasoning process or identity has migrated.

## Contents

1. **The Chimaera Garden** — generative worlds whose inherited laws can mutate and recombine.
2. **Novelty boundary** — what prior latent-memory and causal-intervention research already covers.
3. **The Dream Genome** — opaque donor-derived state, operational invariants and an addressable-KV frontier.
4. **Five experiments and a moonshot** — imprint, causal law, Counterfactual Allele, Chimaera and descendants.
5. **Mechanism and feasibility** — actual donor state, model interfaces, two positive oracles and CPU cost controls.
6. **Causal evaluation** — paired interventions, negative controls and separate S0–S5c claims.
7. **Generative outputs** — a provenance-backed atlas with simulation hooks and beautiful failures.
8. **Execution gates** — G0 feasibility only first; independently reviewed later milestones.
9. **Resolution and remaining risks** — explicit treatment of all nine second-review findings.
10. **Review request** — separate re-review of G0/G1 and the aspirational horizon.

---

## 1. The Chimaera Garden

### The experience

Donor A spends a few minutes in a fictional world. It encounters invented physical laws, odd materials, events, exceptions and consequences. Recipient B has **not read the world**, and is given neither its story nor its labels. Instead, an encoder captures features of A's genuine computational state and a model-specific compiler delivers a small, continuous, recipient-native memory prefix.

B is then invited to:
- describe a consequence of that unfamiliar physics;
- invent a machine, organism, ruin, ritual or material that could exist there;
- handle a new counterfactual event that the donor never explicitly narrated;
- continue exploring the world for several turns, retaining its laws;
- grow an independent descendant world from the same original memory.

In the full speculative vision, memory capsules can be grafted from one world into another. The worlds evolve like an **ecology of acquired memories**. Some inheritance is faithful, some is distorted, and some becomes a genuinely new invention. The machine exposes both the creative artefact and the genealogy of the state that produced it.

**Illustrative fiction; not an observed neural result:** A privately reads that a subterranean coral grows toward mechanical vibrations and releases buoyant glass spores after three minutes of silence. B, seeing only a numeric capsule and an unrelated fixed instruction, invents a railway fungus whose spores rise whenever the engines stop. A second donor remembers that metals change their hardness when exposed to moonlight. A proposed *Chimaera* splice asks whether a receiver invents a coherent machine obeying **both** laws rather than a generic random collage.

The previous proposal's three semantic labels are retained as a **sanity instrument**, not the creative destination. The destination is a reusable, selectively editable *world law*: one thing can happen differently and the recipient's imagination responds coherently while unrelated inherited constraints remain stable.

### Central creative object: an evolving Dream Atlas

Each dream entry has:
- raw donor world text and a sealed, authoritative synthetic law graph;
- an opaque donor-state fingerprint and a small compiled latent capsule, with model/adapter provenance;
- one or more recipient continuations, creative inventions and counterfactual predictions;
- source/child relationships, mutation/splice operations and per-branch lineage;
- matched no-memory, shuffled-memory and compact-text baselines;
- a reader-facing "field guide" interpretation with explicit raw-versus-editor distinction.

The deliverable should be browsable as Markdown/JSONL, with optional SVG/HTML visualisation *only if* it adds value. A collection of 25 strange but coherent mechanisms is useful on its own. A latent memory that survives one unfamiliar task is a legitimate research result on its own. Neither substitutes for the other.

## 2. What is novel, and what is not?

MEMPLANT is **not** the first attempt at latent model communication, KV-state conversion, learned pseudo-tokens, query-conditioned memory or multi-sharer fusion. The relevant preceding work includes [KV-Lingo](references/02-kv-lingo.md), [Cache-to-Cache](references/05-cache-to-cache.md), [XKV](references/06-xkv.md), [ICAE](references/08-icae.md), [C²KV](references/11-c2kv.md) and [Memorilla: Latent Semantic Memory for LLMs](https://arxiv.org/abs/2610.21375) (October 2026). Memorilla already investigates learned query-dependent memory tokens for a frozen LLM and agent experience; a memory-token decoder alone is not sufficient novelty.

Nor are *causal variables in neural networks* a new idea. [Causal Abstractions of Neural Networks](https://papers.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html), [Interchange Intervention Training](https://proceedings.mlr.press/v162/geiger22a.html) and [Distributed Alignment Search](https://proceedings.mlr.press/v236/geiger24a) motivate explicit source-to-base intervention tests. MEMPLANT should borrow this discipline; it cannot claim to invent causal abstraction.

**Potential distinctive contribution, conditional on empirical demonstration:** an *inherited donor-derived latent state* supports **operationally predictable semantic edits**, **interaction-dependent fusion** and **descendant-specific acquisition** across frozen, unrelated model substrates. The key is that these claims are separately falsifiable. A source-specific soft-prefix improves on generic conditioning, but is not by itself a novel form of transferable causality.

| Capability already studied elsewhere | Proposed discriminating question in DREAMGRAFT |
| --- | --- |
| Cache translation and repeated handoff | Can a transferred *rule* be selectively edited in a different receiving world? |
| Learned continuous memory tokens and compressed context | Does a rule-specific numerical edit generalise beyond lexical/template changes? |
| Multi-sharer cache fusion | Can two donors jointly determine a **new interaction**, and do bidirectional interventions preserve each source's independent contribution? |
| Model-generated experience storage | Can C recover a new consequence learned **only by B**, separating it from B's printed tokens? |
| Causal alignment/interchange training | Can interventional invariants persist *across donor/recipient model boundaries*, and what is lost during transfer? |

**Scientific hierarchy:** contextual feature communication is easier than useful state transfer; a V-derived fingerprint is easier than addressable K/V memory; recipient-conditioned source-fact retrieval is easier than law transfer; law transfer is easier than transferable latent edits; two facts coexisting are easier than interaction-dependent recombination; a second transfer is easier than genuinely new information acquired and inherited. The proposal treats each as a different experimental claim.

Two cautionary baselines: [When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches](https://arxiv.org/abs/2608.04893) and [Draft-KV](https://arxiv.org/abs/2609.34754). They motivate **mismatched-donor** and counterfactual identity tests instead of inferring communication merely because any prefix improves average performance.

All claimed originality remains *hypothetical*. Priority and novelty will require a fresh comparison against the literature at publication time.

## 3. The Dream Genome — a source-derived latent with testable operations

A **Dream Genome** is the deliberately narrow name for a versioned, opaque memory object *obtained from donor model computation*. It is not yet a universal representation, a natural causal graph, an editable self-describing program, or a faithful copy of an LLM's full conversation.

~~~text
private synthetic world + authoritative evaluator-only causal graph
                       |
               frozen donor A prefill
                       |
        source K/V, positions and hidden states
                       |
        E_A: initially pooled V feature baseline
                       |
                versioned latent z
                       |
             optional branch / edit / cross
                       |
        D_B: frozen-model-specific compiler
                       |
     B-native soft embeddings (or later deep KV)
                       |
   fixed recipient query -> frozen B -> evaluated output
~~~

**Mandatory source-origin constraint:** the learned numerical payload must originate in A's actual computational state. The evaluator's law graph, target labels, summaries, original text and source identifiers never enter the recipient-facing payload or prompt at inference. If we directly encode the law graph, we have demonstrated *symbolic-to-latent communication*, not the proposed memory transplant. Retain symbolic-ground-truth evaluation separately.

### Minimum manifest / provenance

- Source hash, donor model/tokenizer revision, layer-selection and positional metadata.
- Encoder revision; payload bytes/dtype/shape/hash; recipient-specific compiler revision.
- Parent capsule hashes and operation log: *imprint, branch, independently reimprint, latent edit, cross, descend*.
- User/tenant/source permissions; synthetic-only pilot; immutable lineage and run/seed record.
- Evaluator-only world graph and answers in an isolated record, **not** a field delivered to B.
- Native recipient compatibility, attention mask/positions, fixed task template and fallback path.

### Operational invariants (hypotheses, not schema guarantees)

| Invariant | Operational test |
| --- | --- |
| **Source identity** | A real donor prefix follows its donor's held-out truth; a swapped prefix follows the *new* donor's truth. |
| **Rule selectivity** | Editing heat behavior flips heat-dependent answers while preserving vibration and exception answers. |
| **Composition** | Two parents produce correct *interaction-dependent* consequences neither parent can determine alone. |
| **Edit transfer** | An edit derived in world X works on a separately encoded world Y with different nuisance details. |
| **Independence** | Sibling branches cannot modify one another's source manifest or recipient state. |
| **Inheritance** | A descendant passes along B-only new information, not merely A's prior fact or B's echoed words. |
| **Substrate robustness** | The same immutable z yields semantically corresponding outcomes when separately compiled to two recipients; **unproven** until tested. |

These are **behavioral invariants**, not a claim that every latent coordinate maps to a human-labelled causal variable. “Same operation” is evaluated by changes to the receiver's decisions, with a symbolic intervention oracle and paired controls.

### A deliberate representation ladder

**R0 / cheapest baseline:** 4 selected donor layers × 4 temporal bins × pooled V heads. The proposed Qwen geometry yields ~2,048 scalars; use a 32/64-wide adapter to compile 8 recipient input embeddings. Useful to prove an *information-bearing contextual feature channel*.

**R1 / matched controls:** uncontextualised token embeddings and contextual hidden-state summaries at matched feature/prefix budget and training effort. Since V is derived from contextual hidden states via learned projections, **V cannot contain information absent from the complete source hidden state**. If R0 beats R1 in a particular task, it shows a useful bottleneck or inductive bias, *not* intrinsically richer information or a unique KV advantage.

**R2 / optional addressable memory (requires a new gate):** retain a small sequence of source K/V with **key-dependent retrieval** by learned queries or a cross-attention readout, rather than averaging away addressing and order. Compare under matched total retained bytes, target-token positions, adapter capacity, and training/compile costs; include position-order/exception tests. This is the appropriate place to examine whether *the associative structure of KV* offers a benefit beyond ordinary contextual representations.

**R3 / speculative portable causal genome:** a shared z with model-specific decoders and edit/cross operations that satisfy the invariants above; no obligation to implement until R0–R2 evidence justifies it.

A capsule is not “editable” because its bytes can be changed. It is editable only if a controlled operation makes the **intended causal consequence** change and unrelated consequences remain stable. Likewise, a stored clone is not live KV cloning until model-native state restoration has been validated.

## 4. The five dream experiments, plus the Counterfactual Allele

### DG-1 — Imprint: source-specific latent communication

**Question:** Can A's actual contextual state cause an unrelated frozen B to recover donor-specific information without seeing A's text?

Use a tiny 2–3 memory interface smoke; then 24–48 examples with short, supervised, teacher-forced answers. Donor A's V fingerprint is the primary representation; compare uncontextualised token embeddings, contextual hidden-state summary and zero/random/donor-swapped prefixes. Add B's direct fitted-prefix **interface oracle** and B's **full-text task-capability oracle** (distinct tests).

**Early assay:** small *material / trigger / consequence* tuples, but describe their recovery as **cross-model contextual feature communication**, not as transferable causal memory. Report source-following on a swapped genuine capsule relative to both the original source and the substituted source.

**Science:** S0–S3; no claim of editable, addressable, cross-task or multi-generation memory.  
**Creative use:** preserve 6–12 raw recipient inventions and short editor-independent dream postcards.

### DG-2 — Causal Dream Physics: laws and exceptions

**Question:** Does the same inherited capsule support deductions **not explicitly answered in the donor's story**, under an instruction unseen during adapter fitting?

The source is a small symbolic-grounded fictional world rendered into multiple, genuinely varied diaries. Vary heat/hardness, vibration/buoyancy, material exemption, temporal ordering and interactions. Hold out entire *factorial combinations*, source-world families and paraphrase templates from the sealed test; group all counterfactual twins into the same split. Balance lexical cues, sentence positions and document lengths.

The deterministic evaluator knows the world rules; **it is forbidden to serve those rules to the donor-state compiler or recipient at inference**. It asks fixed-choice/answer-likelihood questions about unseen interventions first, then evaluates open-ended inventions through derived rule constraints. A positive result is a correct *novel consequence*, not recognition of a label.

**Science:** S4 requires source-specific rule application on the held-out intervention distribution, with matched text/KV/embedding controls. This remains *operational* causal-rule transfer, not a proof of an identifiable causal model inside B.

### DG-3 — Dream Surgery: three fundamentally different operations

1. **Source reimprinting control:** edit the donor's source text, re-run A, compile a new capsule and start B at a clean branch. This proves source sensitivity, **not latent surgery**.
2. **Latent-only edit:** derive an operation from already-created latent capsules. Apply it to an *unseen, separately encoded world*, without re-reading that world's text or giving B labels. Test changed-law fidelity and unchanged-law preservation. This is the proposed genuinely experimental surgery.
3. **True mid-session recipient-state modification:** edit B's already-populated live cache, carefully accounting for dependent keys/values, positions, token history and downstream recomputation. This is **not** demonstrated by a fresh branch using a different prefix and requires an independent later approval/contract, informed by [AKV](references/12-akv.md).

**Science:** S5a only if a *latent-only* edit is selective and transfers to held-out worlds. Fresh-branch source reimprinting never counts as S5a.

#### DG-MOONSHOT-01 — The Counterfactual Allele (priority after DG-1)

A direct and unusually revealing experiment in **portable numerical changes to learned world rules**. Construct a factorial quartet:

| Capsule | Heat law | Vibration law |
| --- | --- | --- |
| \`z00\` | heat hardens | vibration sinks |
| \`z10\` | heat softens | vibration sinks |
| \`z01\` | heat hardens | vibration floats |
| \`z11\` | heat softens | vibration floats |

Compute an edit from one pair *without knowing or transmitting the encoded world's text*:

\[
\Delta_{\text{heat}} = z_{10} - z_{00},\qquad
\widetilde z_{11}=z_{01}+\Delta_{\text{heat}}.
\]

Evaluate \`D_B(ẑ11)\` against the **separately encoded** \`z11\` on held-out consequences, and against \`z01\` on unchanged vibration constraints. Do not judge success by cosine similarity to \`z11\`: judge *semantic actions and predictions*. Apply a heat edit learned from one quartet to different material names, episode ordering and world templates; keep all query, task and adapter configurations fixed.

- **No-new-network first:** direct vector delta; if that fails, report failure. A trained low-rank edit operator is a *separate* optional experiment with its own held-out split and compute gate.
- **Controls:** direct source re-encoding, shuffled heat deltas, random/norm-matched shifts, embedding-derived deltas, other-law deltas, no edit, textual edit reference and “heat only” recipient prompts.
- **Invariance:** the edited heat consequence changes in the correct direction; vibration consequences remain fixed; unintended label changes are measured.
- **Commutation correction:** fixed additive edits commute identically: `(z + Δh) + Δv = (z + Δv) + Δh`; reversing them is therefore **not** a meaningful scientific test. Report joint semantic rule satisfaction separately. To test nontrivial operator order, a separately approved future experiment must use independently specified **non-additive or context-dependent** edit maps, evaluate recipient-side causal outcomes on independent worlds, and predeclare whether the modeled rules ought to commute.
- **Avoid leakage:** law labels appear only in the evaluator and offline experiment construction; B sees fixed prompts and numerical capsules.
- **Uncertainty:** near-linear structure may be absent, and a context-dependent edit may not transport. That is a valid negative discovery.

**Decision:** the first proposed moonshot *after* source-specific DG-1, under a separate bounded DG-2/DG-3 research gate. It must not be silently included in G0/G1 training.

**Creative output:** a family of strange “before/after” worlds where one impossible physical law changes and machines, organisms and architecture adapt accordingly. Keep failed edits in the atlas as failures, not imagined successful mutations.

### DG-4 — Chimaera: joint, intervention-sensitive recombination

**Question:** Can two separately acquired donor memories jointly determine a consequence **neither one alone can answer**, with independent editability of each contribution?

Start with two memories from one donor model (same-A, different source) before admitting heterogeneous source models. Construct a task requiring A's heat transformation **and** B's vibration rule to determine a single combined event. Use a deterministic symbolic evaluator with known joint predictions and explicit contradictions/exception handling.

**Arms:** A-only, B-only, true A+B fusion, wrong-but-plausible A′+B, A+B′, naive latent mean/concatenation, **independent decoding of each donor plus downstream combination**, direct recipient full-text two-source reference and short text postcards. Hold fixed memory budget and report extra fusion/compute overhead.

**Bidirectional intervention criterion:** swap or edit A while B remains fixed; only A-dependent predictions should change. Then edit B while A remains fixed; only B-dependent predictions change. A result where both parent facts merely co-occur in fluent prose is **not** joint causal inheritance. Report dual-source *interaction* performance and unrelated-law preservation.

**Science:** S5b requires correct withheld **joint consequences** and source-specific interventions above one-source and wrong-source controls. Independent-decode-and-combine is a practical **comparator, not a mandatory victory condition**. Report scientific fusion fidelity separately from speed, cost and modular-baseline accuracy; claim superior integration only with matched-resource evidence. Latent fusion itself is not novel.  
**Creative output:** chimaera machines, ecological hybrids and explicit paradox notes, with source-specific lineage.

### DG-5 — Ghost Descendants: newly acquired B information

**Question:** Can C inherit something **B newly learned or derived** that donor A never knew, via B-derived numerical computational state and no B→C text handoff?

A originally gives B latent rule R. B privately encounters new event E (absent from A's source) and may infer new conclusion J from R+E. Extract B's native state and send it through a separately trained B→C (or B→B) compiler while C sees a fixed query. **No printed B continuation or A source is passed directly to C**. Excluding original prefix positions from B's state readout is necessary to test some alternatives but **not sufficient**: downstream states can still encode A and generated tokens.

**Additional acquisition-versus-exposure comparison (independent v3 review):** Capture B's numerical state **T1 immediately after B receives event E**, then **T2 after a controlled inference step** intended to derive previously unstated conclusion J. Independently verify B can derive J. Compare recipient C's answer under T1 and T2 with an identical C prompt, plus a matched-compute/position control in which B advances through neutral steps. Match external source E and any generated/teacher-forced tokens across paired conditions to the extent the runtime permits; disclose unavoidable token-position differences. Retain E/E′ counterfactual swaps and A→C ancestral controls. A T2-specific gain on J strengthens attribution but does **not prove** internal thought inheritance: C may receive a better numerical encoding of E and derive J itself. Without this incremental controlled gain, report only input-state transfer, not a computed-memory transplant.

**Predeclared inheritance arms:**

| Arm | B's additional experience | Purpose |
| --- | --- | --- |
| Silent/controlled state | E is processed without sampling new B output, then snapshot | isolates newly processed B experience from freely generated-token contents |
| Matched-token controls | hold generated visible token sequence fixed across E vs control E′, or teacher-force identical continuation | tests whether E changes state beyond the printed token string |
| Natural continuation | B receives E and generates normally before state extraction | ecologically valid but confounded by B's generated text |
| A-only direct | A→C, without B experiencing E | ensures new information was not already available to ancestor |
| Counterfactual/swap | substitute E′ or unrelated B donor history | recipient must follow B-specific event identity, not a generic B cache |
| Plain text relay | C receives available B text/summary | practical comparison; explicitly **not** a non-textual transplant |

The evaluator tests J and source-specific consequences, plus whether a new rule was **acquired by B** versus repeated from A. Even the silent arm can transport contextual representations of E rather than some special non-textual discovery; distinguish *B-specific information inheritance* from proof of a unique introspective thought representation.

**Science:** S5c's basic finding is B-specific information transfer through numerical state. The stronger claim of **B-derived conclusion J inheritance** requires an improvement from the post-inference state over the immediate-after-E state, beyond equal-step/position and matched-token controls, with independent verification that B derives J. Otherwise report B-input/experience transfer only.  
**Creative output:** genealogy of mechanisms that inherit some features and develop others across incompatible neural substrates.

**Stop rule:** DG-3–DG-5 are an ambitious horizon, not automatic follow-on tasks. A useful gallery can survive a failed latent-channel experiment, but the provenance must say “no demonstrated transplant.”

## 5. Mechanism, model capability and resource feasibility

### Initial model pair — instrumental, not a universal result

| Role | Candidate | What must be demonstrated |
| --- | --- | --- |
| Donor A | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | pin config/weights/tokenizer; extract **actual**, non-mislabeled layer K/V tensors under \`no_grad\` |
| Recipient B | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | native text **task** ability, differentiable \`inputs_embeds\` path, correct masks/positions, cache continuation, native fitted-prefix **interface** ability |
| Optional editor | neurounner pinned LFM2.5-1.2B-Instruct | polishing *only after* raw scoring; no access to donor text or sealed law labels |
| Future receiver C | unselected | new model/adapter subject to separate approval, budgets and scientific controls |

A and B are unrelated **and not similar in parameter count**. This is a cheap first communication experiment, not proof of peer-sized model-state interchange. Existing neurounner GGUF/llama.cpp throughput and LoRA runs do not establish PyTorch gradient-through-receiver speed or KV insertion support. Pin an isolated compatible Transformers/PyTorch environment without modifying existing host workflows.

### Two positive oracles that answer different questions

**Oracle I — numeric interface:** optimise a recipient-native soft prefix *directly on a few known examples* while B's weights remain frozen. Verify valid gradients into the prefix, observable output dependence on its values, and inference continuation equivalence. This memorising diagnostic is **excluded from generalisation scoring**.

**Oracle T — task competence:** give B **the complete true synthetic world as text** and ask held-out fixed-choice causal questions and short answers. Determine whether this tiny receiver can actually solve the task. If B cannot solve tasks with the source available, do not interpret its failure with compressed latents as failure of memory transfer. Simplify G1 tasks or propose a stronger B with a separate resource estimate.

Neither oracle substitutes for the other. A positive interface oracle does not imply causal reasoning capacity; a positive text oracle does not imply useful soft-prefix conditioning.

### Concrete first mechanism

1. With frozen A in \`no_grad\`, prefill each donor memory and extract selected layer states. Initial **R0**: 4 × layers × 4 length-normalised bins × 2 KV heads × 64 head dimensions = 2,048 scalar value features, subject to actual configured head geometry.
2. **Precompute once and checksum every donor feature vector**, and log model revision, tokenizer, source hashes and native tensor shape. Unload A **before** fitting B's adapter to avoid overlapping resident model weights. Generate matched feature baselines on the same source corpus and budget.
3. Start with \`LayerNorm → 32/64 bottleneck → 8×576 soft input embeddings\` (~431k dense weights for the 64-wide case; confirm actual package implementation). Normalize output magnitude relative to recipient embedding statistics. Freeze A/B weights; train only this small adapter.
4. Train B under teacher forcing with a **single masked cross-entropy objective** for the short target answer; the shared instruction is masked out of loss. B's parameters are frozen but B's computations remain differentiable: do **not** wrap B forward in \`no_grad\`, and disable receiver generation/inference caching during gradient training.
5. At inference separately test the same \`inputs_embeds\`+task prefix route with a proper attention mask, BOS/template, position sequence and B-native KV continuation. Verify deterministic repeated continuation at identical seeds. Inspect actual receiver inputs for donor labels, source names and unintended text leakage.
6. Favor **forced-choice target-answer log-likelihood** for early science and raw sampled prose for the separate dream atlas. No KL distillation, deep-KV injection or multiple fusion networks in G0.

### Cost and overfitting reality

Eight FP16 vectors × width 576 use ~9 KiB at the input; their recipient FP16 30-layer KV content is ~180 KiB (assuming 3 KV heads and 64 head width), excluding other sequence positions and overhead. A ~431k-weight FP32 adapter uses ~1.7 MB for parameters, plus optimizer states. The A source prefill, retained source-state bytes, feature extraction, trainable activations through 30 B layers, compilation and B decode all count in full resource accounting.

**24–48 training memories versus ~431k adapter parameters is a memorisation risk.** This tranche is a capability/smoke assay only. A held-out factorial world split or new independent world family is required before causal generalisation is claimed. No free-text sample score or dramatic visual output negates this.

**Five-step G0 measurement:** forward/backward wall time, peak RSS or VRAM, gradient norm and finite values, high-water memory across feature extraction, full-text/native-prefix oracle scores, exact source and runtime hashes. Do not infer that the existing two-runner-hour limit is enough; it is an exploratory cap, not an evidence-based forecast.

## 6. Causal evidence contract: what a transplant must actually do

### Required controls

| Method | Question it answers |
| --- | --- |
| Full source text to B | Can B solve the task with all information? **Reference, not a fixed upper bound.** |
| Short extractive text postcard | Can an ordinary cheap text channel do as well at a comparable latency/storage budget? |
| Uncontextualised A token embeddings, same budget | Can a lexical encoder recover labels without meaningful context processing? |
| A contextual hidden states, same budget | Do V-derived features offer a favorable bias compared to another contextual representation? |
| Actual pooled A values → B prefix | Can a true contextual donor-state feature reach B through a numerical channel? |
| Zero/static or norm/covariance-matched random prefix | Is any extra conditioning enough? |
| Mismatched real donor | Does **donor identity** determine the answer rather than generic latent activation? |
| Counterfactual twin source swap | Does B follow the new world's altered law while holding unaffected rules constant? |
| Directly optimized B native soft prefix | Is a numerical prefix *usable* at all? Diagnostic only, no generalisation claim. |
| Independent decode-and-combine | Does two-source fusion exceed merely extracting both rules separately? |
| Direct source reimprint and text edit | Is a latent numerical edit doing anything beyond re-running the encoder or writing a textual new source? |

For each donor swap, report **both original-source accuracy and swapped-source accuracy**. For each counterfactual intervention, report changed-rule fidelity *and* preservation of unrelated laws. For joint inheritance, ask unseen questions requiring **both** donors simultaneously. For descendants, distinguish information that A knew, information encoded in B's generated tokens, and information introduced only by B's new experience.

Controls are **paired by world, task, decoding seed and recipient checkpoint**. For numeric arms, B's fixed task prompt must not reveal donor ID, answer labels, training split, source filename or any lexical cue. Source truth and scorer-only data must be segregated from receiver input. Audit all prompt/tensor routes.

### Revised evidence hierarchy — disjoint research claims

| Level | Permitted finding | What it does **not** establish |
| --- | --- | --- |
| **S0** | Valid gradients, embeddings, positions and native cache interfaces | Any memory transfer |
| **S1** | A numerical prefix predictably changes B's outputs | Example-specific communication |
| **S2** | Correct donor identity changes source-specific recipient answers | KV-specific advantage or transferable causal models |
| **S3a** | Contextual source features outperform lexical/uncontextualised baselines | Value projections contain unique extra information |
| **S3b** | Pooled value-derived features beat matched contextual hidden-state features | Unique KV information or addressable KV memory |
| **S4** | Transplanted laws predict held-out counterfactual consequences | Identifiable internal causal mechanism in B |
| **S5a** | A latent-only rule edit is selective and portable | Live KV surgery unless already-decoded B state itself was edited |
| **S5b** | Two memory sources support an **interaction-dependent** held-out consequence | That B→C descendants can inherit B's new experience |
| **S5c** | C inherits information newly available to B through verified numeric state transfer | Consciousness, general learning or absence of possible token encodings |

**No automatic promotion between levels.** S3b only means the chosen value projection is a more useful feature encoding in this setting, not that it can exceed the information content of the full hidden state.

### Data, independence and uncertainty

- G1: 2–3 fixed instrument fixtures, then 24–48 tiny training cases. This is a **smoke test**, not confirmatory evidence.
- G2 candidate: ~240 synthetic worlds split 160/40/40, grouped by **scenario/world/template family** and by *counterfactual twins*. Cross-product factors include rule/exception/interaction order, materials and nuisance wording. Hold out specific combinations and intervention queries **not appearing as answers in training diaries**.
- A sealed 40-world test is imprecise (around a 50% proportion, naive 95% binomial uncertainty is roughly ±15 percentage points). Report actual paired differences and intervals, family-level grouping, failures, invalid rate, forced-choice likelihoods and sampled results. Multiple continuations from the same world are **not independent worlds**.
- If an apparently large scientific claim emerges, obtain a **fresh untouched confirmatory bank**, not repeated peeks at the existing sealed cases. Move exploratory probes to the wandering notebook and freeze new hypotheses before formal retesting.
- Avoid a huge model/template/slot/loss grid. One representation and one loss first, then the smallest evidence-driven ablation.

### Four independent outcome dimensions

1. **Information-specific:** source identity, counterfactual rule recovery and donor-swap following.
2. **Interventional:** rule-local edit success, unrelated-variable preservation, cross-world edit generalisation, bidirectional composition.
3. **Inheritance:** B-only acquisition versus ancestral echo or generated-text representation, matched token controls.
4. **Creative:** blind novelty, coherence, practical usability and fascinating failure selection. These artistic scores are **never** presented as evidence in dimensions 1–3.

Report storage positions, bytes, encoder cost, input/output bandwidth, native cache bytes, compute, real end-to-end latency, and calibration/adapter amortisation. A reduction in receiver context positions is not an actual total compression measurement.

## 7. Creative programme: wander deliberately, but preserve genealogy

The **Atlas of Stolen Dreams** is not a leaderboard: it is a research-backed foundry of impossible matter, machines, creatures, physical laws, contradicting worlds and strange failures. Its distinctive value is that interesting artefacts can be **reused, simulated and traced back to observed operations**.

Each selected artefact has four complementary **views**, kept explicitly distinct:

1. **Raw dream:** verbatim recipient continuation, model/adapter hash, source capsule, task, seed, method and all matched-control raw outputs.
2. **Field guide:** readable interpretation from a human or optional LFM editor; this may invent structure and must never replace raw evidence.
3. **Ancestry:** donor worlds, verified or failed latent operations, grafts, branches, original source hashes and descendants; distinguish a speculative family tree from a measured numerical inheritance.
4. **Simulation hook:** a minimal implementable trigger, exception, state update and consequence that a game/simulation designer can prototype (especially useful for materials and physics experiments).

Good artefacts include **before/after alleles** (a world whose heat law changes, with consequences for machines and ecology), **chimaera machines** (a device whose operation needs two independently inherited laws), and **ghost descendants** (characters who acquire new constraints and pass them along). A deliberately failed transplant can still produce striking art; the scientific ledger must plainly mark it as a failed transfer.

Use a small, browseable static Markdown index with JSONL raw data; optionally a lightweight SVG genealogy where it adds actual interpretability. No elaborate front-end, renderer, image generation campaign, bespoke scheduling framework or large issue forest is required. The novelty should live in *interventions and inheritance*, not dashboards.

Blinded reviewers choose interesting inventions without seeing whether they came from latent, text-summary, random-prefix or post-editor conditions. Record novelty, internal consistency, reusability and unexpected failure value separately. Never select appealing samples for scientific accuracy denominators; retain every raw generation including malformed and discarded ones.

**Failure-tolerant product:** if numerical transfer never reaches S2, the same synthetic world bank and text-postcard generator can still make a weird-physics atlas. That product is creative usefulness, not successful memory transplantation.

## 8. Execution gates, budgets and authorisation

| Gate | Work | Evidence required to request next gate | Authority |
| --- | --- | --- | --- |
| **G0 — instrumentation + two oracles** | Pin two model revisions; real A K/V extraction; feature hashing and A unload; B native soft-prefix/gradient, masks/positions/cache; native full-text **task competence**; five-step wall/RSS probe | Oracle I and Oracle T results, source provenance, input-leak check, gradient validity, deterministic continuation, realistic training-cost estimate | **Proposed first authorisation; owner has not approved it** |
| **G1 — bounded imprint** | Up to 24–48 first training cases, one CE objective, uncontextualised/hidden/V controls, zero/random/swapped arms, 6–12 raw creative postcards | Reproducible source-specific effect (or no-go); paired raw outputs; actual cost. No causal-generalisation claim | **Hold until G0 evidence and explicit owner approval** |
| **G2 — causal worlds** | Factorial synthetic worlds with withheld rule/exception/intervention combinations; B full-text capability; symbolic scorer; fresh-sealed-world policy | Source-specific withheld consequences and novel-task transfer. Define errors and uncertainty | **Separate owner approval after G1** |
| **G3 — Counterfactual Allele** | Delta from latent paired worlds applied to *independent target world*, compared to reimprint, random and text edits; test unchanged-law invariance | S5a selective latent-only edit; possible explicit no-go before training extra editor | **Separate owner approval after G2** |
| **G4 — Chimaera interaction** | Dual-source capsule with interaction-only test, bidirectional swaps, modular comparator and contradictions | S5b joint consequence above one-source and wrong-source controls; modular accuracy and efficiency reported separately | **Separate owner approval; not implied by G3** |
| **G5 — Ghost Descendants** | Immediate-after-E versus controlled post-inference B state, matched-step/position and emitted-token controls; qualified B→C translator | J-specific post-inference improvement above immediate and matched-step conditions needed for stronger acquired-conclusion claim | **REWORK; separate later research approval** |

**Compute discipline:** initial G0 only on existing hosted CPU when pinned models and isolated dependencies are available. G1 proposes *at most two aggregate hosted runner-hours of adapter fitting*, pending the five-step profile. That is a cap, not a throughput guarantee. No large training, GPU/TITAN use, autonomous tasks or extra providers without fresh authorization and a resource estimate.

**Storage discipline:** source data, prompt fixtures, raw records, tensor/corpus hashes, negative observations and small summaries follow neurounner [dataset policy](../docs/DATASET_FILING_SYSTEM.md). Big tensors/checkpoints remain outside ordinary Git with retrievable immutable manifests, never source-embedded secrets. Use wholly synthetic sources in the first pilot. Do not accept screenshots, models or datasets of uncertain ownership as anonymous input.

**Stop/repair:** interface or task oracle fails; mismatched donor does as well as true donor; apparent signal disappears against simple embeddings; gradients invalid; unreported corpus leakage; resource cap exceeded; inaccessible raw evidence. Each outcome can be recorded as a useful negative result. Only the owner can authorize the next gate, and **neither PR #229 nor the foundation PR #228 is approved for merge**.

## 9. Self-review, second independent review and resolution register

### Independent v2 review: verdicts

The reviewer assessed [v2 HEAD \`fa442a190f8a33a50b0bd7eeaf9c8729e7dcf68c\`](https://github.com/techrote/neurounner/commit/fa442a190f8a33a50b0bd7eeaf9c8729e7dcf68c). Verdict: **G0/G1 CONDITIONAL APPROVAL; DG-2–DG-5 REWORK**. The review does **not** authorize code or training. This v3 is a proposed response, not proof the reviewer approves the changes.

### Finding-by-finding amendments

| Finding | v3 disposition | Concrete change |
| --- | --- | --- |
| **P1-A** V-only is not uniquely KV memory | **Contract clarified; novel mechanism deferred** | R0 is contextual value-feature communication, R1 hidden/embedding comparators; R2 addressable K/V experiment separately gated; S3a/S3b narrower |
| **P1-B** lexical and causal shortcut risks | **Test contract expanded** | factorial withheld interactions and consequences, world/template-group splits, twin pairs, novel task instructions, no label leakage |
| **P1-C** native-prefix oracle not task oracle | **Both specified** | Oracle I tests numerical interface, Oracle T tests B's full-text rule-solving on held-out cases |
| **P1-D** Chimaera may merely co-list facts | **Joint intervention contract** | interaction-dependent question not solvable from one donor; A/B bidirectional edits; independent decode+combine baseline |
| **P1-E** descendants may relay B's generated tokens | **Controls expanded** | silent/controlled state, fixed identical generated sequences, natural generation, direct A→C, B-only experience and B→C state-swap |
| **P2-A** Dream Genome lacked invariants | **Operational invariants added** | source identity, edit specificity, composition, inheritance, restore/branch and substrate robustness as *tests*, not assumptions |
| **P2-B** tiny corpus and CPU cap inadequate for strong claims | **Claim levels bounded** | G1 smoke only, 40-world sealed test imprecision disclosed, fresh confirmation for strong claims, 5-step performance measurement |
| **P2-C** literature novelty comparison incomplete | **Expanded** | Memorilla, causal abstraction, interchange-intervention training, distributed alignment search; older cache/prefix papers retained |
| **P2-D** fresh re-encoding was mislabeled surgery | **Operations disambiguated** | DG-3 explicitly separates reimprinting, transferable latent-only edits and genuine mid-session KV modifications |

### Independent v3 re-review follow-up — 15 November 2026

Reviewed head: `24cee5390c85bed0d2e5b293223ddad9e5cc432f`. **Design verdicts:** G0 APPROVE; G1, G2/G3 and G4 CONDITIONAL APPROVAL; G5 REWORK. Three remaining findings are addressed in this v3.1 amendment: (1) additive commutation test is mathematically vacuous and withdrawn; (2) modular decode-and-combine is a comparator, not a scientific minimum for Chimaera; (3) G5 has immediate-versus-post-inference comparison, with matched processing controls and remaining input-encoding confounds. These are specification corrections **not new experimental results**. Approval of G0's design is **not owner authorization**. PRs #228 and #229 remain Draft/unmerged.

### The first moonshot — why the Allele comes before Chimaera

After G1, **Counterfactual Allele** is the strongest *bounded* test of whether a Dream Genome has useful manipulable structure. One inferred numerical law difference must work on a separately encoded world and preserve unrelated laws. This has a clear negative-result interpretation if it fails, generates appealing before/after dream artefacts, and is simpler to falsify than a generational ecology.

Only after a convincing causal law and selective edit should a two-donor chimaera be claimed as recombination, and only after that should B-specific descendants be given costly new cross-model adapters. The *research horizon* stays broad but the *evidence surface* stays discriminating.

### Remaining unresolved risks (not waived)

1. A 135M recipient may not be capable of multi-rule tasks even with explicit text. Oracle T must be checked before interpreting latent failure.
2. A pooled 2,048-dim V feature may lose keys/order/exceptions. R2 addressable KV may be scientifically required and more expensive.
3. A supervised ~431k-weight adapter with only 24–48 memories can learn labels/templates and fail out of domain. G1 is explicitly a pilot.
4. Counterfactual-delta addition may be non-linear and context-dependent; a failure is not evidence that all editable memory is impossible.
5. A “recombined” output may borrow generic knowledge from B's weights rather than source-provided interaction. Factorial/mismatched-source controls reduce but cannot eliminate this.
6. A silent B→C transfer can still encode tokens B *read* rather than a privileged hidden “thought”; source-specific interventions are essential.
7. A shared z decoded by two models does not imply identical cognitive state or universal address space.
8. Repeated creative cherry-picking creates gallery selection bias. Keep all generations and report blind aggregate metrics separately.
9. Newly added research frontiers may exceed the computational and engineering resources of neurounner; every gate is independently optional.

**Revised design principle:** maximize the ambition of *what could be learned*, not the breadth of code or number of simultaneously running jobs. There remains no approval to implement.

## 10. Request for independent technical review — v3

**Requested response:** a severity-ranked independent review, with explicit and **separate** APPROVE / CONDITIONAL APPROVAL / REWORK / REJECT verdicts for **G0**, **G1**, **G2/G3 Counterfactual Allele**, **G4 Chimaera**, and **G5 descendants**. The owner is requesting better science and more ambitious *research*, not implementation or a merge.

Please examine:

1. **Capability:** Is G0's dual oracle sufficient to distinguish a broken prefix interface from an incapable 135M causal-reasoning recipient?
2. **Attribution:** Does S2/S3a/S3b properly distinguish pairing-specific latent information, contextualisation, useful V-derived bias and genuinely addressable K/V behavior?
3. **Intervention:** Do the factorial worlds and independent \`Δheat = z10 − z00\` test adequately distinguish an actual selective latent edit from keyword decoding, averaging and fresh re-encoding?
4. **Causal abstraction:** Are the specified recipient-side invariants meaningful and precise enough? Would interchange intervention or distributed alignment techniques provide a stronger *minimal* test?
5. **Fusion:** Is a held-out **joint** consequence and bidirectional donor swap the decisive Chimaera assay, or can it still pass through isolated extraction plus composition in B's pretrained weights?
6. **Inheritance:** Do the silent/controlled-token descendant arms genuinely add evidence of **B-acquired** information passing to C, given that even silent B state can encode E?
7. **Novelty:** Given Memorilla, C2C, XKV, KV-Lingo and causal interchange training, is the combined intervention/genealogy programme potentially distinctive? Which claim should be most carefully restricted?
8. **Compute/engineering:** Are model pins, state features, precompute/unload, gradient-through-B, task assays and gating realistic on the declared CPU budget? What *smallest* improvement would materially raise interpretability?
9. **Artistic value:** Does the raw dream / field guide / ancestry / simulation-hook archive enable useful discovery even when scientific controls reject the transplant story?
10. **Simplification:** Which speculative feature is most likely to divert the programme without improving information transfer? Which one **additional** moonshot is worth keeping?

**Authority boundary:** Do **not** implement, train, spend compute, create schedules, merge PR #228/#229 or treat any design assertion as a successful runtime result. G0 instrumentation is the only **proposed** first implementation increment, pending explicit owner approval. Subsequent milestones require fresh owner authorization and appropriate evidence.

### Primary literature to check for novelty and controls

- [KV-Lingo](references/02-kv-lingo.md), [Cache-to-Cache](references/05-cache-to-cache.md), [XKV](references/06-xkv.md), [ICAE](references/08-icae.md), [C²KV](references/11-c2kv.md), [AKV](references/12-akv.md).
- [Memorilla (arXiv:2610.21375)](https://arxiv.org/abs/2610.21375) — query-conditioned memory tokens and frozen LLMs.
- [Geiger et al., Causal Abstractions of Neural Networks (NeurIPS 2021)](https://papers.neurips.cc/paper_files/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html).
- [Geiger et al., Inducing Causal Structure for Interpretable Neural Networks (ICML 2022)](https://proceedings.mlr.press/v162/geiger22a.html).
- [Geiger et al., Distributed Alignment Search (CLeaR 2024)](https://proceedings.mlr.press/v236/geiger24a).
- [When Does Latent Communication Pay? (arXiv:2608.04893)](https://arxiv.org/abs/2608.04893).

**All experiments described in this PR remain unexecuted.** The aim is an experiment capable of detecting an unexpected *causally meaningful* inheritance effect if one appears, while remaining scientifically useful and creatively interesting if it does not.
