# MEMPLANT / DREAMGRAFT-001 v2 — The Chimaera Garden

**Subtitle:** The Stolen Dreams Foundry becomes an experimental ecology of inheritable, editable, recombinant memories.  
**Status:** revised design for independent review; implementation **NOT AUTHORISED**.  
**Date:** 2026-11-15. **Previous proposal head:** c17c2f8ec3891134825b9a18d3db02b7d565c37d.  
**Location:** techrote/neurounner/MEMPLANT/. **Parent:** [MEMPLANT README](README.md), [concept](CONCEPT.md), [pseudo-embeddings](PSEUDO_EMBEDDINGS.md), [research programme](EXPERIMENTS.md), [reference catalogue](REFERENCES.md).  
**Authority:** documentation revision only; no model training, implementation, schedule, cross-repository work or merge is approved.

> **The ambitious question:** Can one neural model's private experience be made into an inheritable computational object—something another, independently trained model can *remember, reinterpret, mutate, branch, recombine, and pass on* without being handed the original text?

This is deliberately a playful computational metaphor. No claim is made about consciousness, subjective experience or biological memory. A successful computation does not mean an entire mind, conversation, reasoning process or identity has migrated.

## Contents

1. **The Chimaera Garden** — a generative project with dream worlds, inheritable rules and an atlas of descendants.
2. **What would be genuinely new** — the distinction between existing cache translators and MEMPLANT's ambitious research questions.
3. **The Dream Genome** — source-grounded latent capsules, target-native compilers and transparent lineage.
4. **The five experiments** — from small imprints to causal mutation, recombination, memory descendants and heterogeneous substrates.
5. **Mechanism and model feasibility** — precise initial adapter, models, injection semantics and capability controls.
6. **Causal evidence and controls** — guardrails against merely learning labels, accidental perturbation or prompt leakage.
7. **Generative outputs and the wandering laboratory** — atlas, dream ecology and what remains useful after weak results.
8. **Execution and resource gates** — concrete first steps and non-automatic expansion triggers.
9. **Self-review and professional-review resolution** — severity-ranked corrections and residual risks.
10. **Independent review request** — adversarial review questions and authority boundary.

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

## 2. What would be genuinely new?

**Not claimed as novel:** transferring KV states between models, soft prompts, learned continuous memory tokens, gated inter-agent cache fusion or compressed-cache recombination. These have relevant published precedents, including [KV-Lingo](references/02-kv-lingo.md), [Cache-to-Cache](references/05-cache-to-cache.md), [XKV](references/06-xkv.md), [ICAE](references/08-icae.md), and [C²KV](references/11-c2kv.md).

**MEMPLANT's proposed novel direction** is the *intersection* of these capabilities, experimentally tested as operations on living, lineage-bearing memories rather than as single question-answer channels:

1. **Counterfactual inheritance:** Does an inherited *causal rule* survive an unseen task, not just a memorised label?
2. **Differential memory mutation:** Can changing one source rule selectively alter an already-inherited recipient memory without reintroducing the original text?
3. **Memory chimaeras:** Can two independently acquired donor memories be fused into an interesting but still source-grounded new constraint system?
4. **Latent descendants:** Can a receiver's continuation become the source of a later transplantation, with measurable inheritance through generations and without any text relay?
5. **Substrate-independent dream identities:** Can one saved capsule be compiled into multiple unrelated model-native prefixes, measuring what stays invariant when the recipient model changes?

This is a **research proposal**, not a claim of academic priority or proven technical feasibility. Some individual operations may overlap existing work. The empirical contribution sought is *content-specific causal inheritance through edits, compositions and generations*, with the creative ecology as a secondary output.

A caution from independent literature: [When Does Latent Communication Pay? A Causal Audit of Relayed KV Caches](https://arxiv.org/abs/2608.04893) and [Draft-KV](https://arxiv.org/abs/2609.34754) report settings in which purported latent communication gains do not depend on the correct example's donated state. MEMPLANT must not make that mistake. **A recipient must demonstrably follow the donor's identity and counterfactual edits, not merely be improved by having any prefix.**

## 3. The Dream Genome: a real but minimal interface

A **Dream Genome** is a name for an opaque *source-derived latent capsule with explicit lineage*. It does **not** mean a universal embedding coordinate system. Initially, the format represents a frozen donor's compressed state; it is compiled into a recipient-specific soft prefix. The same capsule being re-used in multiple tasks constitutes a *postcard*, not yet model-independent memory.

~~~text
authoritative synthetic dream / source text / law graph
                             |
                   donor A prefill once
                             |
            actual layerwise A KV / other state
                             |
       E_A: fixed pooling + small learned encoder
                             |
          Dream Genome z (opaque, versioned)
                 /            |          \
                /             |           \
       branch A query   branch B query    edit / splice
                 \             |           /
                 D_B(z, task, receiver identity)
                              |
                 B-native soft prefix / KV
                              |
                    B continuation + cache
                              |
                    measured behavior + art
                              |
                 optional B→C descendant
~~~

### Capsule manifest (proposed, not a frozen public API)

- **Identity:** opaque capsule ID; source hash; model weight, tokenizer and inference-runtime identities; tensor-shape/layout version.
- **Payload:** externally stored tensor bytes with digest, dtype, projection/pooling recipe and encoder revision.
- **Lineage:** parent capsule hashes; operation (imprint / branch / counterfactual edit / splice / descend); stochastic seed and code revision.
- **Boundary:** source ownership and allowed recipients; no confidential user conversation; synthetic-only first pilot.
- **Scientific fields:** source labels, hidden truth graph and held-out answer keys kept in a *separate evaluator-only record*, never in the receiver-visible manifest.
- **Restore rule:** numerical capsule plus compiler, recipient model pin, positions/masks and full fixed task template; reject incompatible compilations.

**Important:** the *authoritative transcript and law graph* remain lossless, private to the scorer and archived with hashes; the capsule is an intentionally lossy derived view. The first experiment must prove that changing the capsule while holding the receiver's prompt and seed fixed changes source-specific behavior. It must not hide semantic clues in filenames, metadata, text prompts, or a manually selected donor-specific task.

A first-stage capsule can be nothing grander than a pooled V-state feature vector and one small adapter. The manifest exists to make later branching/editing/composition measurable, not to prematurely create a library framework.

## 4. The five experiments: increasingly alien memory

### DG-1 — Imprint: prove a source-specific memory channel

**Question:** Can A's contextualised KV-derived features cause frozen B to recover information from a synthetic experience without B seeing its text?

Keep the independently trained small models. Start with 24–48 tiny memory examples and one supervised training objective, **not** distillation. Fit a V-only donor-state adapter to 8 soft-prefix positions, evaluate zero, random, source-swapped, uncontextualised token-embedding, hidden-state, full-text and compact-text baselines. Verify a positive receiver-native soft-prefix capability control first.

Use simple material/trigger/effect labels *only* for the first plumbing assay. Follow immediately with an **unseen task instruction** (e.g., invent a device based on the inherited rule rather than listing labels). Never score an LFM-polished idea as raw memory transfer.

**Useful even if weak:** recoverable occasional motifs and an atlas of unfiltered generative outputs.  
**Scientific gate:** correct donor identity and counterfactual direction matter more than eloquence.

### DG-2 — Causal Dream Physics: transmit laws, not nouns

**Question:** Does a memory capsule convey a changeable, compositional *law of an imaginary world*?

Give donor A a short *multi-event world diary* (roughly 100–300 words) with two or three interacting physical rules and an exception. Train with short questions about unseen consequences and interventions, not only labels. A sealed evaluator has a deterministic mini rule graph for each world; it can answer whether a proposed mechanism is consistent without using another LLM as the primary oracle.

Construct minimal counterfactual twins: same words and world vocabulary, but reverse the direction of a relation, negate a consequence, change an exception, or change time ordering. Keep twins in the same data split. For example, "Copper becomes brittle **after** heating" versus "Heating prevents copper from becoming brittle" is not permitted to be solved just by a keyword bag.

Require B's *unseen* invention to remain consistent with at least one donor-only rule. Report both positive transfer (recovering a law) and negative transfer (inventing an incompatible rule). Have human readers score novelty independently from rule fidelity.

**Useful outcome:** compact weird-physics setting cards that can inspire playable simulation constraints.  
**Scientific gate:** task-transfer and counterfactual accuracy above donor-swapped and token-embedding baselines.

### DG-3 — Dream Surgery: mutate one memory without destroying the rest

**Question:** Can a **small, explicit change to donor state** cause a selective change in a recipient's interpretation?

For paired worlds differing in one causal law, compute separate source states and capsules. First test **branch re-compilation**: B runs from the same clean starting prompt with old versus new capsule; verify the changed rule flips while unrelated rules remain stable. Then test a delta representation (new capsule minus old, or a learned low-rank edit) applied to a stable latent, but compare against simply recompiling the new capsule. The former is not a novel improvement if it gives the same effect at higher cost.

**No false claim of in-place KV surgery:** replacement of a prefix at a fresh branch is *not* live editing of an already-decoded B cache. True live surgery would require correcting downstream dependencies/positions, and is a separately approved stretch target; compare against recomputation and cache-editing research such as [AKV](references/12-akv.md) and [PatchKV](https://arxiv.org/abs/2609.26219).

**Useful outcome:** "before/after" dreams—worlds where one law changes and the emergent inventions adapt.  
**Scientific gate:** causal edit specificity; measure changed-rule recovery and unaffected-rule preservation, not just that generations differ.

### DG-4 — Chimaera: graft two dreams into one mind

**Question:** Can *two unrelated donor memories* produce a coherent hybrid mechanism that obeys facts from **both**, without the recipient being shown the originals?

Start with two memories encoded by the **same donor model A**. Hold both capsules fixed and compare:
1. a single-donor capsule (A-only or B-only);
2. naive vector mean/interpolation (a deliberately weak baseline);
3. a small gated/fused latent compiler that combines capsules;
4. a full-text two-memory receiver reference;
5. compact written clue cards under a comparable input/byte/cost accounting.

A deterministic evaluator defines worlds where both source constraints are jointly possible and worlds where they conflict. A good chimaera identifies or resolves the conflict rather than simply emitting fashionable blended nouns. Source identity and edit trace must determine which constraints survive. Later, **only after** one working fusion, allow source A and unrelated donor C with different architectures; do not silently call a same-A two-memory merge "cross-model merging."

**Useful outcome:** surprisingly inventive hybrid materials, impossible machines and interacting ecologies with source lineage.  
**Scientific gate:** above single-source and mismatched-source controls on *both-source* causal questions; quantify conflict handling.

### DG-5 — Ghost Descendants: memory across generations and substrates

**Question:** Can one recipient's *experience of a transplant* become the source state for a new recipient, and what survives?

A receives a memory; B gets only a numeric graft and a fixed task. After B continues for one or more steps, snapshot B's native computational state. A **newly calibrated B→C or B→B transfer path** attempts to condition C without sending donor A's text **or B's generated text** to C. This tests transgenerational state inheritance rather than ordinary textual communication.

Compare first-generation and second-generation rule recall, novel invention quality and accidental drift. Use shuffled ancestor IDs and matched text-relay baselines; preserve exact generation, compiler, and parentage. Multiple recipients compiled from a **single shared z** are a separate portability challenge, not a result of having two independently trained pairwise adapters.

**Useful outcome:** family trees of alien ideas, with measurable fidelity versus creative mutation across descendants.  
**Scientific gate:** a second independent transfer actually consumes donor-derived computational state and retains source-specific information above mismatch controls.

**Stop rule:** DG-3–DG-5 are aspirational research fronts, *not* implied authorisation. Failure of DG-1 ends any scientific claim of inheritance; the standalone atlas generator can continue and must be labelled accordingly.

## 5. Mechanism and model feasibility

### First bounded candidate pair

| Role | Proposed model | Reason / limit |
| --- | --- | --- |
| Donor A | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Open weights, inspectable decoder-only Transformer state, 24 layers, width 896, 14 attention heads, 2 KV heads; independent family from receiver. Pin model/tokenizer revision. |
| Recipient B | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Frozen 30-layer, width-576, 9 query/3 KV-head Llama-style recipient with inexpensive relative compute; prior neurounner CPU training evidence but **not** this prefix method. |
| Optional text editor | Neurounner's pinned LFM2.5-1.2B-Instruct | Human-readable atlas editing **only after** raw scoring; must never see donor private memory/answers or be called part of the non-textual transfer. |
| Future recipient C | Model selected by feasibility and independent approval | Separate family preferred; CPU/GPU budget and decoder compatibility to be measured. Not a v2 prerequisite. |

The pair is deliberately **not parameter-count matched**: this is a cheap heterogeneous experimental path, not evidence for large/peer-sized interchangeability. The experiment requires pinned **full-precision or training-capable** PyTorch/Transformers models for autograd. The host's GGUF/llama.cpp inference results cannot be used as proof that it supports this training or directly injecting custom soft prefixes. Keep all experimental dependencies isolated from existing neurounner workflows.

### First-stage adapter

1. Freeze A. For each synthetic donor memory, perform genuine A prefill under no-grad and capture per-layer K/V states (and position/layout metadata).
2. Select four donor layers and four length-normalised time bins. Begin with values only: for the specified Qwen geometry this is approximately **2,048 pooled scalar features** (4 × 4 × 2 KV heads × 64 per head), subject to checked runtime shapes.
3. Fit a small LayerNorm → 32- or 64-wide MLP → **8 × 576** recipient embedding compiler. With 64-wide bottleneck and that feature count it has roughly **0.43 million** trainable dense weights, before biases and normalization. Do **not** claim it recreates A's full context.
4. Frozen B receives the generated vectors prepended to the embeddings of one identical task instruction; attention mask, positions, BOS/chat template and continuation semantics must be verified.
5. During adapter training, disable recipient inference-cache writes and preserve gradients **through B's operations back to the adapter**, even though B parameters are frozen. During generation, use validated native cache continuation and deterministic control comparisons. No donor text, labels or semantic metadata enter B's forward path.
6. After a working objective, optionally separate the source-state encoder E_A and the B-specific compiler D_B to preserve a reusable z and test alternate recipient decoders. That separation is not free and is not assumed successful.

**Distillation is deferred.** Start with one short supervised extraction/causal-response objective and gradient checks. Only consider KL distillation after a reproducible evidence-bearing implant.

### Positive soft-prefix capability oracle

Before judging cross-family transfer, directly optimise a tiny **B-native continuous prefix** on a few known calibration cases (using only their known target answers). Show B's output actually responds to the fitted vectors; compare to no-prefix and random prefix with the same input. This is a **diagnostic interface oracle**, not a generalisation result: its test answers were used during fitting, so it is excluded from sealed performance comparisons. If the native soft-prefix oracle fails, stop and repair recipient embedding/position/caching semantics or choose a more capable B with a newly approved compute budget.

### Practical cost honesty

Eight FP16 input vectors at width 576 occupy about 9 KiB, but B's **corresponding per-layer KV** is about 180 KiB at FP16 before other context, tensor overhead and source state. Include input-state bytes, cache bytes, all A prefill compute, transfer, B compilation and B decode costs. A smaller number of virtual positions is *not* a measured compression ratio.

## 6. Causal evidence: the transplant can be wrong

### Locked controls (addresses all independent review P1 findings)

| Condition | What enters B | What it rules out |
| --- | --- | --- |
| Full donor text | D itself | B's native full-text reference capability; **not** a strict ceiling |
| Equal-budget text postcard | source-grounded extractive/short textual clue | useful ordinary compression baseline |
| Uncontextualised donor token embeddings | a matched-budget pooled embedding fingerprint through same-capacity compiler | lexical/feature classifier versus contextual KV value |
| Donor hidden states | matched-budget hidden-state fingerprint | whether K/V contributes beyond simpler contextual representation |
| True donor V → B pseudo-prefix | real A context state, learned compiler | actual candidate |
| Zero/static prefix | constant | prompts and B prior knowledge |
| Norm/covariance-matched random prefix | nonsemantic vectors | generic perturbation/regularisation |
| Swapped donor prefix | different source's genuine capsule | whether correct *source identity* matters |
| Counterfactual twin swap | same lexicon, one reversed rule | whether B follows causal rule changes rather than word statistics |
| Recipient-native fitted prefix | B-optimised known case | validates that the virtual-token channel can be used at all |
| Optional no-adapter / random adapter | fixed random projection | trained mapping versus generic perturbation |

**Key outcome:** for a swap A→B, report both (i) performance relative to A's ground truth and (ii) performance relative to **B's swapped-in ground truth**. A valid source-sensitive implant should follow B after the swap. For counterfactual twins, test that precisely the edited relation changes and other source facts remain stable. This avoids a widely observed pitfall in latent communication: a generic prefix can improve a metric without communicating the actual donor's content.

**Condition-specific audit:** B's input record must contain the same fixed prompt, decode settings and nonsemantic indices across control arms; the only permitted semantic condition is the actual text in designated text baselines, and the only numeric condition in latent arms is the tensor. Inspect prompts/logs for label/source-name leakage. Source annotations and evaluator answers stay outside receiver work directories where practical.

### Data and preregistered splits

Begin with 2–3 examples for interface feasibility and **24–48 small training examples** for an early signal.

A later controlled bank may contain approximately **240 short structured memories (160 train / 40 validation / 40 sealed test)**, with templates and **counterfactual twins grouped into the same split**. Individual paraphrases of the same world/scenario must never cross train/test. Include lexical/causal adversarial cases: reversed dependencies, negation, swapped antecedents, changed thresholds, and exception-order changes. Balance material/trigger/effect frequencies and ensure the model cannot infer donor labels from prompt templates, length or position alone.

A small 40-case sealed test offers weak precision; report per-field accuracy, complete-tuple accuracy, causal-intervention consistency, invalid rate, paired differences and uncertainty intervals. Run a predeclared small number of decoding seeds (up to 3 if affordable) and treat repeated peeks as exploratory, **not** fresh holdouts. Avoid expanding formal test claims to the 12–24 open-ended dreams used for creative play.

### Measurements that matter

- **Semantic fidelity:** donor-specific facts, causal consequences, swap-following accuracy, edit specificity, unrelated-fact preservation, novel-task transfer and multistep retention.
- **Mechanism specificity:** real KV-derived features versus uncontextualised embedding and hidden-state features, same training budget and compiler capacity.
- **Task transfer:** the **same trained capsule**, without additional tuning, used for extraction and at least one held-out creative instruction.
- **Artistic utility:** blinded human selection of useful mechanisms, judged separately from faithful rule inheritance; raw outputs first, polished versions later.
- **Efficiency:** source prefill, adapter compile, recipient forward/backward, inference time, peak RSS/VRAM, tensor bytes and amortisation break-even versus text replay/retrieval.

**Scientific reporting categories:**
- **S0:** native embedding interface works;
- **S1:** prefixes perturb B reproducibly, but no source-specific content;
- **S2:** genuine source-specific, donor-swap-following semantic transfer above zero/random;
- **S3:** KV-derived contextualisation beats an equal-capacity simple token-embedding baseline on held-out causal tests;
- **S4:** tasks and causal edits preserve a significant subset of the source rule structure;
- **S5 (stretch):** multi-donor fusion or second-generation handoff with source-specific lineage.

Never report S1 as "memory transplanted"; S2 does not imply S3; S3 does not imply universal memory, and S5 is not guaranteed.

## 7. Creative programme: wander deliberately

The **Atlas of Stolen Dreams** is a companion creative research tool, not a benchmark leaderboard.

Proposed artefact families:
- **Impossible materials:** state-dependent hardness, gravity, temperature, sound, light, memory, decay.
- **Alien machines:** contraptions whose operations derive from one or two inherited world laws.
- **Living ecologies:** organisms that acquire materials, exchange energy or reproduce under impossible but internally testable laws.
- **Causal paradoxes:** two donor dreams conflict; the recipient either finds a reconciliation or documents a failure.
- **Memory family trees:** branch a capsule, mutate a law, splice a second source, observe descendants diverge.
- **Repair and invention postcards:** take a raw B concept and let a text-only editor rephrase it for readability, always preserving the raw evidence.

One dream can generate a *material / machine / ecology / failure* quartet under four independent receiver instructions. Another dream can be combined with a second donor to produce a coherent chimaera. Creative variation is welcome, but claim inheritance only when an intervention changes the correct inherited law with matched controls.

Keep a **wander notebook**: unplanned experiments, accidental breakthroughs, aesthetically interesting failures and exact seeds, openly marked exploratory. Curate a public gallery of non-sensitive synthetic outputs with a lineage graph (a simple table or static SVG suffices). Do not build a full game engine, a browser app, procedural animation, a sprawling agent framework or image-generation system merely to make the gallery look impressive.

**If latent transfer fails:** the same structured world bank and text-only postcard baseline can still drive a useful weird-physics foundry. Its output remains creative material, but all scientific labels must say *no demonstrated transplant*.

## 8. Execution gates and compute limits

| Gate | Minimum actual work | Evidence before expansion | Authority |
| --- | --- | --- | --- |
| **G0 — instrument and cost** | pin model revisions, download/hashes, true A K/V extraction, B inputs_embeds, mask/position checks, positive prefix oracle; 5 forward/backward updates | real gradient on adapter, deterministic continuation, peak RSS and step timing, no hidden source leakage | **first proposed authorisation only** |
| **G1 — first imprint** | 24–48 cases, one low-rank adapter, matched A-source vs swapped/zero/random, token-embedding and hidden-state controls; 6–12 raw dream cards | reproducible donor-sensitive result or precise negative explanation | separate approval after G0 |
| **G2 — causal worlds** | grouped 160/40/40 bank, counterfactual twins, fixed held-out tasks, native/text baselines | source-aware swap and unseen-rule consistency, formal/creative metrics separately | separate approval after G1 |
| **G3 — dream surgery** | branch-compile old/new capsules; measure rule-local specificity; optional learned delta | changed-rule flip, preserved unrelated rules and lower edit cost if claimed | separate approval after G2 |
| **G4 — chimaeras** | fuse two source memories; blind dual-rule and contradiction tests | both-source evidence beats single-source and mismatched controls | separate approval after G2/G3 |
| **G5 — ghost descendants** | B-generated native state → B/C recipient via new pairwise adapter or shared z; no text relay | second handoff verified, ancestry recovery above mismatched lineage | separate research programme/approval |

**Compute:** G0 begins CPU-only on existing hosted Linux if model/license and dependency checks succeed. The reviewer explicitly warned that prior neurounner LoRA measurements do **not** price a prefix adapter requiring backward activations through B. Inspect a 5-update profile (wall time, RSS, gradient norm, cache behaviour) *before* scheduling real training. Proposed G1 cap: **two aggregate hosted runner-hours of fitting** and at most three preregistered variants, subject to observed throughput. G2–G5 require new estimates. Local GPU/TITAN hardware may be proposed only after a recorded resource comparison and explicit owner approval.

**Storage:** source texts, prompts, raw generations, negative results, manifests and small metrics belong under neurounner's documented [dataset filing system](../docs/DATASET_FILING_SYSTEM.md). Store large tensors/weights/checkpoints externally with immutable checksums; do not commit private contexts, unreviewed external datasets, gigabyte blobs or temporary Actions-only evidence. Use public/synthetic materials in this pilot.

**Stop conditions:** bad masks/positions; unverified real cache extraction; native fitted prefix fails; missing donor identity effects; data leakage; resource budget exceeded; semantic label shortcut; or conflating editing/branching with true live KV patching. Do not automatically progress because the atlas looks entertaining.

## 9. Self-review and professional-review resolution

### Independent reviewer requirements incorporated

| Review | Earlier weakness | V2 amendment |
| --- | --- | --- |
| **P1-A** | three-label recovery could just be feature classification | same-capacity **uncontextualised token-embedding** baseline plus hidden-state control; S3 explicitly requires KV-derived advantage |
| **P1-B** | lexical/template shortcuts across split | group counterfactual twins, scenario families and paraphrases; include negation, role/temporal reversals, causal exception edits |
| **P1-C** | B may be incapable of using eight continuous tokens | recipient-native directly fitted soft-prefix **positive capability oracle** before any transplant conclusion |
| **P1-D** | swapped prefix scored against original only | report **original and swapped source** accuracy; change exactly one property to measure causal direction |
| **P2-A** | very small B may be too weak | native text and native-prefix baselines; explicit fallback to a larger B only after budget/approval |
| **P2-B/C** | LoRA evidence does not price gradient through frozen B; cache/autograd semantics | real five-step forward/backward profile; train with cache disabled; separate inference continuation validation |
| **P2-D** | combining labels and KL was too complex | one short supervised loss first; target distillation only after successful simpler trial |
| **P2-E** | positions are not total compression | account for bytes and full donor/adapter/recipient compute |
| **P2-F** | label-tuned prefix may fail outside label task | sealed **unseen creative task** with same prefix and no retuning |
| **P3** | excessive layer/bottleneck tuning | fixed four-layer/four-bin V baseline, 32/64-width bottleneck, no early grid search; full-text B is a reference, not guaranteed ceiling |

### Independent self-critique of this V2 ambition

1. **Novelty is a programme-level aspiration, not established.** A tiny V-fingerprint-to-prefix classifier may not be materially new; only causal edits, recombination and distinct task/substrate transfer give the concept additional substance.
2. **The weak link could be the recipient.** SmolLM2-135M may fail at imaginative, multistep causal reasoning even with ideal source information. Always measure native B and leave open a reviewed, stronger receiver.
3. **The donor state may not add information over token embeddings.** If S2 works but S3 fails, label the result "learned non-textual cross-model feature channel" and do not manufacture a KV advantage.
4. **Splicing numerical vectors is not automatically semantic composition.** It requires controlled paired worlds, dual-source oracles and perhaps a trained fusion module.
5. **A multi-generation result can be contaminated by text.** A B→C claim requires auditing every tensor/prompt path so C never sees A's original text or B's printed continuation.
6. **World rules are simplified synthetic truth, not unrestricted memory.** The physics-law mini-world enables causal verification but constrains generalisation; the creative dream bank probes outside it without overclaiming.
7. **The grand vision has major risk of scope explosion.** G0 and G1 are deliberately small and remain valuable. G3–G5 are independent optional frontiers, with no implied CI automation, new issue tree, model zoo or framework.
8. **Creative curation is selection-biased.** Keep every raw attempt; the gallery is chosen art, not a substitute for sealed scientific metrics.
9. **Privacy matters even for internal tensor state.** Use only synthetic sources in this pilot; donor tensors are not an anonymisation mechanism.

**Design verdict:** more ambitious **research horizon**, more discriminating **first experiment**. The earlier proposal was a small classifier-flavoured proof. This proposal makes causal inheritance, selective mutation, dual-donor recombination and generational transfer the explicit horizon, while retaining a genuinely executable initial mechanism.

## 10. Request for independent technical review

Reviewers should be adversarial, imaginative and precise. In particular:

1. Does the S0→S5 evidence hierarchy properly separate generic conditioning, source-specific feature communication, KV-specific advantage and actual causal inheritance? What additional negative control is essential?
2. Is the donor **V-derived** eight-slot compiler the most informative small first step, or would a simple deep-KV interface provide better evidence with similar cost?
3. Does the proposed split really separate compositional reasoning from memorised templates, when counterfactual twins have similar vocabulary? What minimum held-out worlds would be enough to expose the problem?
4. Can a tiny recipient with a native fitted-prefix oracle but poor reasoning meaningfully test rule transfer? Which fallback model/architecture is most informative under CPU and a modest local GPU budget?
5. Is **Dream Surgery** a technically honest edit assay, and what would be required to upgrade fresh-branch recompilation to *real* mid-session KV editing?
6. Is **Chimaera** materially more interesting than concatenating two text summaries or averaging two prefix vectors? How can dual-source inheritance be scored without an LLM judge?
7. Is the **Ghost Descendants** route plausible without the original text or B's output leaking into C, or does that require a much larger research programme?
8. Which one moonshot would maximize interesting, interpretable creative output under an approximately two-runner-hour first fitting budget?
9. Identify unjustified assumptions, training/serving incompatibilities, leakage channels, misleading novelty claims and opportunities to simplify without losing the weirdness.

**Requested decision:** APPROVE / CONDITIONAL APPROVAL / REWORK / REJECT, with explicit separate recommendations for **G0/G1** versus the aspirational **G2–G5**. The owner has requested greater ambition and novelty, **not implementation**. Neither this stacked documentation PR nor its parent #228 may be merged without explicit final owner approval.