# MEMPLANT / DREAMGRAFT-001 — The Stolen Dreams Foundry

**Status:** proposal for independent technical review; not approved for implementation.  
**Date:** 2026-11-15.  
**Host:** `techrote/neurounner`; proposed later implementation in `experiments/memplant/dreamgraft-001/`.  
**Parent research dossier:** [MEMPLANT](README.md), especially [CONCEPT](CONCEPT.md), [PSEUDO_EMBEDDINGS](PSEUDO_EMBEDDINGS.md), [EXPERIMENTS](EXPERIMENTS.md), and [REFERENCES](REFERENCES.md).  
**Authority:** research and design only. No trained adapter, running programme, successful transplant, speedup or model quality is claimed.

## 1. One-paragraph pitch

Build a tiny **generative idea foundry** in which donor model **A** reads a fictional micro-memory (a place, physical rule, material, mechanism, or bizarre experience), and frozen recipient model **B** tries to invent a new artefact influenced by it **without receiving the micro-memory text**. A small trainable adapter maps features extracted from A's *actual inference state* into 8 temporary soft-prefix positions that B can attend to. The emitted fragments become an **Atlas of Stolen Dreams**: odd physical processes, materials, mechanisms, props, ecology and game-ready hooks, with lineage to their donor sources and matched controls. The exploratory objective is *measurable donor-derived influence*, not faithful full-session restoration. A partial, eccentric, nonrandom transfer is already interesting; pure creative output without demonstrated transfer is still an artefact, but must be labeled **not evidence of transplantation**.

### A deliberately imaginary example (not an observed result)

**A reads, privately:** “A buried metallic garden grows toward vibrations. When silence lasts three minutes, its glass leaves become weightless and drift into the ceiling.”  
**B sees:** a fixed instruction plus **only A-state-derived continuous pseudo-tokens**.  
**B might invent:** “Hushglass moss: railway-side growth that climbs engine vibrations and sheds buoyant panes when a train stops.”  
**Optional editor:** a separate existing neurounner model can reformat that raw seed into an idea card; the editor is not part of the transplant measurement and may only see the recipient's output, not the hidden donor memory.

## 2. Useful even if it only slightly works

The product is a searchable, source-traceable **dream atlas** rather than another abstract accuracy chart. Each card records a generated mechanism, its visual cue, an unexpected failure mode, source identity, receiving model and method, with raw and polished versions. A human can harvest intriguing concepts for game materials, worlds, fiction or prompts without trusting any scientific claim.

Useful outcomes, in descending order:

- **Actual transplant signal:** B reliably inherits some donor motifs or constraints without seeing the text; usable low-bandwidth non-textual model-to-model conditioning.
- **Partial / strange transfer:** B inherits a tendency, material class or effect but recombines it into a novel mechanism; both a research observation and generative raw material.
- **No measurable state transfer but good generation:** the prompts/idea-bank/post-editor still form a standalone ideation generator; clearly **no-go on the claimed transplant**.
- **Failure to make useful cards:** retain an instrumented cache-extraction, prefix-injection, reproducible-control and failure-analysis fixture for future MEMPLANT work.

Do not conflate creative novelty with memory fidelity. The atlas can be artistically useful while the transplant hypothesis fails.

## 3. Proposed small model and runtime selection

| Role | Candidate | Why | Qualification still needed |
| --- | --- | --- | --- |
| **Donor A** | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Small independently trained Qwen-family Transformer; Hugging Face weights expose per-layer K/V; different tokenizer/attention geometry from B | Pin *exact* upstream weight and tokenizer revisions; verify extraction, licenses and resource fit |
| **Recipient B** | [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Cheap frozen recipient for gradient-through-model soft-prefix training; neurounner has previously qualified this family's CPU LoRA workflow | Use source pin from [TRAIN-001](../docs/TRAIN_001_CPU_LORA.md) only after confirming current artifact availability; verify `inputs_embeds`, position handling and continued generation |
| **Optional atlas editor** | neurounner's pinned LFM2.5-1.2B-Instruct | Turns raw, sometimes malformed tiny-model ideas into readable concepts; already used as a compact formatter in the host | Strict separation from evidence scoring; no access to donor source or hidden answer labels |

**Important:** this pair is *not* similar in parameter count; the smaller recipient is intentional for a cheap first CPU experiment. It is a **cross-family, nonidentical-tokenizer test**, but not a demonstration of large-model or peer-size interoperability. The recipient's low raw writing ability is part of the reason to separate transplant measurement from editorial output. The pinned GGUF inference server used elsewhere in neurounner is **not assumed to expose trainable autograd or KV tensor injection**. Use an isolated PyTorch/Transformers experimental environment after a small compatibility check; do not silently modify the host runner's pinned GGUF workflow.

## 4. The proposed experimental mechanism (V1, after self-review)

```text
                    donor-only fictional memory D
                                 |
               frozen Qwen prefill; collect source K/V
                                 |
       selected layer × time-bin × KV-head features
           (start with values; keys optional later)
                                 |
                small trainable A→B adapter
                       ~8 × d_model(B)
                                 |
              8 recipient-native soft tokens (z_B)
                                 |
   fixed B task instruction (NO D text; NO motif labels)
                                 |
                 frozen SmolLM2 recipient B
                                 |
                  raw invention + motif tags
                                 |
          evidence scoring     optional LFM editor
                 |                    |
         paired controls       dream-atlas card
```

The first technique is explicitly **KV-derived pseudo-embedding transplantation**. It is *not* a successful direct A-KV → B-KV replacement. This naming distinction is mandatory in results and project descriptions.

### State acquisition

For each donor memory, prefill frozen A, capture true K/V state, and pool **values** from a handful of selected layers (e.g. 4 evenly distributed layers) and 4 relative sequence bins. Keep head identity and fixed-length source-position bins. This yields a fixed-size A-state fingerprint, with an optional last-value/readout. Save the raw tensors outside source Git as permitted by the repository's storage policy, and store content hashes and shape metadata.

The simple first adapter is `LayerNorm → low-rank MLP (~128 bottleneck) → 8 × hidden_size(B)`, with normalization and a controlled output scale near B's embedding distribution. Only adapter parameters train. **Do not** claim it preserves A's entire attention memory: aggressive pooling is an intentionally lossy proof of a state information channel.

Ablations:
- source hidden-state pooling, to ask whether using KV adds value;
- zero/static prefix, random prefix, swapped-donor prefix;
- optional K+V fingerprint, with correct handling of position-dependent rotary keys, *after* V-only works.

### Receiver semantics

Feed B's input embeddings as `[8 generated virtual vectors ; embeddings(fixed instruction)]`, with a correct full-length causal mask, positions and cache continuation. Never concatenate Qwen token IDs onto B's input, and never let B see donor text at inference in the transplant arm. First validate that generated tokens **actually attend to the prefix**, persist through decode, and differ under controlled prefix swaps with all other inputs and seeds fixed.

### Training signal

Generate a fixed, synthetic donor corpus with three discoverable traits: **material**, **trigger**, **consequence**. Train against (1) supervised short motif tags and (2) if compute allows, KL/next-token distillation from **native B given the donor text** on matched continuation questions. These are separate losses/ablations; a tag-only adapter demonstrates compressed feature transfer, **not** all-purpose conversational memory equivalence. Both A and B remain frozen. Log whether target B itself can recover the intended traits *when it is given D in text*; a deficient native full-text teacher is not a fair ceiling for a graft.

### Why not start with direct KV→KV?

Qwen and SmolLM2 have different tokenizers, KV-head layouts, layer counts and embedding coordinates; direct cache injection would introduce alignment, RoPE, sequence-position and optimizer issues before any interesting generation exists. The virtual-prefix technique lets B construct its own valid native attention cache from continuous conditioning vectors. **Direct learned B-native deep KV prefixes** can be an opt-in follow-up, with a separate injection-correctness test and no obligation to proceed if the first result is useful.

## 5. Data, tasks and creative wanderings

**Controlled synthetic bank (proposed):** 240 short fictional memories drawn from 8 materials × 8 triggers × 8 consequences, with varied wording and settings. Use 160 training, 40 validation and **40 sealed held-out combinations**. All component labels should occur in training; hold out exact combinations and paraphrase styles to avoid confusing unseen vocabulary with transfer failure. A corpus generator must retain the exact RNG seed, permutations and every source string. Do not use real user conversations or private source documents.

**Fixed blind receiver query:** “Recover the observed material, trigger and effect; invent one strange tool or natural mechanism based on them.” The instruction is identical across donors and conditions. Raw B outputs may be malformed; preserve them rather than repairing before scoring.

**Open-ended creative bank:** a separate, explicitly unscored set of ~12 micro-memories. These can be generated by Sprocket (already used for eccentric mutations in neurounner), hand-authored, or recombined from public-domain fictional motifs. They must never be added to training after their outputs are examined without a new dataset revision. Use them to see what the model invents, not to establish statistical transfer.

**Generative play:** for each donor memory, branch the *same stored A-state fingerprint* into 2–3 fixed tasks (“invent a substance”, “invent a machine”, “invent a failure”). Compare raw variations and, optionally, the output from LFM text-only polishing. This tests task-conditioned use of the memory and creates a gallery. Do **not** call it independent session cloning unless an actual branch-local model state is created and validated.

**Atlas card:** immutable donor-ID/hash, source license/type, adapter and model revisions, branch/query ID, raw recipient continuation, parsed motif tags or `invalid`, blind condition ID, optional editor output, and a human note about novelty/usefulness. Store an easily browsable Markdown index with JSONL raw data; use README/retrieval links according to neurounner's dataset filing policy. Never overwrite failures or only keep “good” outputs.

## 6. Controls and success criteria

| Condition | B gets | Interpretation |
| --- | --- | --- |
| **Full text oracle** | actual donor D as text | target's capability ceiling on this task |
| **Equal-budget textual hint** | terse extractive / summary clues of similar token or byte budget | practical compact-text baseline |
| **Zero/static prefix** | no donor-specific information | detects prior associations and prompt bias |
| **Shuffled donor prefix** | a real prefix from *another* source | detects superficial prefix activation and leakage |
| **Random norm-matched prefix** | noise with donor-like statistics | detects nonsemantic decoding perturbation |
| **Genuine KV-derived prefix** | A-state-derived 8 virtual vectors | candidate transplant |
| **Hidden-state-derived prefix** | A hidden-state fingerprint, otherwise same architecture/budget | identifies whether K/V extraction specifically helps |

Keep sampling temperature, decode budget, receiver instruction and seed *matched* for paired comparisons. A positive scientific signal requires a held-out increase in **source-appropriate trait recovery** over zero/shuffle/random controls, not just fluent text or generic similarity. Report mean tag accuracy, complete triple accuracy, invalid rate and paired uncertainty intervals, alongside qualitative blind human comparisons of novelty/utility. Forty sealed records is a *small exploratory sample*; do not oversell statistical certainty. A strong equal-budget text summary can win without invalidating a weaker but real non-textual transfer signal.

**Exploratory levels, not release gates:**

- **L0, instrument works:** verified cache extraction and a soft prefix B demonstrably consumes, with native decode unaffected by shape/position errors.
- **L1, perturbation works:** different true donor states cause repeatable, attributable output changes; shuffled/random controls distinguish association from mere instability.
- **L2, weak memory works:** on sealed cases, donor-specific traits survive above controls, though they may be sparse or distorted.
- **L3, useful grafts:** the atlas contains blind-selected reusable concepts and at least some inherited traits; **scientific transfer quality and creative utility reported separately**.
- **L4, extended research (optional):** direct deep-KV graft, multi-turn persistence, shared latent `z`, or other recipient family. **Not required** for a good pilot.

An L1 result is interesting but is *not* evidence of semantic memory transfer. A curated gallery is a practical deliverable even at L0 or L1 if its limitations are labeled accurately.

## 7. Execution plan — small start, wandering allowed

**P0 / preflight:** Verify exact model downloads, tokenizer/config compatibility, use-cache extraction, prefix injection and deterministic continuation with a 2–3 item fixture. Verify no donor text/labels leak to B. Conduct a **5-update training and memory/time probe**, and inspect the actual peak RSS/step duration before choosing training budget. If training is prohibitively slow on hosted CPU, reduce batch/adapter or request explicit local GPU authorization; do not launch an unbounded job.

**P1 / first artificial graft:** Fit the low-rank V-only adapter on a tiny 24–48-item calibration subset, produce the first matched no-prefix/shuffle/real-prefix output triplets, and record actual B inputs. If B receives the prefix correctly, create 6–12 initial dream cards (clearly exploratory, not scored).

**P2 / bounded proper trial:** Freeze a corpus split and training recipe. Run 160/40/40 protocol as resource permits; limit the first hosted pilot to a declared, measured CPU-job quota (suggested initial maximum **two aggregate runner-hours of adapter fitting**, not including downloads and test runs) and no more than three declared training variants. Use validation for tuning; keep the 40 held-out cases sealed until final candidate choice. Preserve all trial heads and failed results.

**P3 / dream atlas:** Generate raw paired variants on the sealed bank plus the 12 exploratory memories; score without editorial intervention; make a compact, browseable atlas from raw B results and optional LFM-polished ideas. Reveal donor sources and labels *after* blind evaluations; annotate surprising partial motifs rather than silently selecting only lucky samples.

**P4 / wandering notebook:** Reserve some deliberate slack for informal probes *after* the first controlled result: change temporal pooling, vary 4/8/16 virtual slots, compare value-only to hidden/K+V, swap the receiver query, inspect where the first motif disappears. Limit to one changed variable per recorded probe when practicable, and separate exploratory observations from preregistered held-out metrics. Do not repeat peeks at the sealed set as if each were a fresh validation.

**P5 / retrospective:** Publish a short observed-outcome memo: what worked, what was aesthetically useful, what failed, costs, evidence links, and whether to pursue deep KV translation or a portable latent format. The programme may end with a documented no-go and still count as a worthwhile exploratory investigation.

**Compute/storage safeguards:** Keep training isolated from the existing host workflows, with a maximum wall/compute budget and stop/cancel support; initial CPU-first feasibility rather than assumptions about available TITAN hardware. Record actual CPU model, RAM, peak RSS, downloaded model pin/hash, adapter bytes and timings. No weights, cache tensors containing confidential text or temporary huge archives committed as Git blobs. If large retained assets are needed, use a licensed external model/artifact store and a hash manifest.

## 8. Self-review of the original sketch (V0 → V1)

The **V0 sketch** was to transplant a full donor KV cache into a different 500M-class model, then run a three-model dream relay and judge whether the results were exciting. That would have been appealing but poorly designed. Before submitting this plan, I challenged it as follows:

| Severity | V0 weakness | V1 correction |
| --- | --- | --- |
| **P1** | Full K/V translation could fail on layers/heads/RoPE/token alignment before the art generator functions | Start with V-derived target-native soft prefixes; make direct KV graft optional and explicitly label method |
| **P1** | Recipient may see text or post-editor may manufacture missing motifs, yielding a false transplant claim | Prohibit D from recipient inference; score raw outputs *before* editor; keep exact B input records |
| **P1** | Interesting words alone may be random effects, not semantic transfer | Include matched zero/static, swapped-donor, random and equal-budget text controls plus held-out trait scoring |
| **P1** | One model may be unable to solve the task even when shown D | Measure full-text B oracle; distinguish receiver incapacity from bad adapter |
| **P2** | Training two near-1B models through autograd could overrun hosted CPU limits | Frozen donor and 135M recipient; train tiny adapter only; 5-step budget preflight |
| **P2** | Native Qwen/Smol model mismatch could be hidden behind llama.cpp GGUF runtime | Isolated pinned HF/PyTorch experiment and explicit inputs_embeds continuation test |
| **P2** | Generative output without source lineage would not be useful research evidence | Indexed atlas with donor IDs, raw generated text, model/adapter IDs, control identity and editorial separation |
| **P2** | Repeatedly viewing held-out results would corrupt an apparent win | Freeze splits; treat exploratory probes as exploratory; run final held-out check once |
| **P3** | “Compression ratio” inferred from 8 slots would exaggerate memory economy | Account for feature pooling, prefix layer/width/bytes, adapter training, compile latency, original text storage |
| **P3** | Complex multi-agent cloning would obscure single-pair transfer | One donor, one recipient, one optional text-only editor; multi-model branching is future work |

**Residual weaknesses after revision (for independent reviewer):**
- The V-only pooled fingerprint is a **feature compressor**, not the whole cognitive state. Success may be closer to transferable classification than general conversation memory.
- A supervised motif triple is intentionally low entropy. True unconstrained memories, source-token recovery and long contexts are outside this first pilot.
- Small B may produce poor creative material. A separate editor improves utility but must never inflate transfer metrics.
- Two model weights plus training-backprop memory may still be too expensive on a 4-vCPU runner. Actual feasibility remains unknown until P0.
- Fixed soft prefixes are **not** demonstrated direct KV-cache migration; direct cache translation is a separate hypothesis.
- An adaptive exploratory workflow is attractive but especially prone to cherry-picking. The sealed split, raw archive and honest status labels are mandatory.

## 9. Independent-review questions and proposed decision

**Request reviewer to assess the idea, not to implement it yet:**

1. Is V-only donor-cache → 8 target-native soft embeddings a meaningful *bounded* memory-transplant proxy, or should the first mechanism instead be deep-KV reconstruction?
2. Are the proposed Qwen 0.5B donor / SmolLM2 135M recipient, PyTorch hooks and compute envelope plausible on a standard neurounner CPU runner? What is the smallest fallback pair if not?
3. Does the synthetic three-trait assay actually distinguish donor-derived semantic influence from cheap label decoding and accidental prompt correlations? Which ablation is missing?
4. Are the strong textual and shuffled/zero controls sufficient to prevent a misleading positive result, and is the proposed blind creative gallery honest about what it can demonstrate?
5. Should the corpus support richer open-ended narratives immediately, or is a small controlled memory grammar the right first step?
6. Where should raw evidence, adapter checkpoints and the atlas index live under neurounner's archival policy to avoid PR/repo bloat?

**Proposed disposition:** **CONDITIONAL RESEARCH APPROVAL, IMPLEMENTATION HOLD.** If an independent reviewer agrees the controls and model injection semantics are adequate, authorize only P0 and a narrow P1 calibration increment on a separate implementation branch. No autonomous schedule, large training sweep, merge or MEMPLANT framework claim is authorized by this proposal.

## 10. Grounding and references

- [KV-Lingo](references/02-kv-lingo.md) — learned cross-model KV translation and behavior distillation.
- [Cache-to-Cache](references/05-cache-to-cache.md) and [XKV](references/06-xkv.md) — donor/recipient communication and gated latent grafts.
- [ICAE](references/08-icae.md) and [CMC](references/09-cmc.md) — compact memory embeddings, including query-aware conditioning.
- [Prefix-Tuning](references/14-prefix-tuning.md) and [Soft Prompt Tuning](references/15-soft-prompt-tuning.md) — virtual vectors in frozen recipients.
- [AKV](references/12-akv.md) — positional and history semantics, relevant if later branching actual caches.
- [neurounner CPU training evidence](../docs/TRAIN_001_CPU_LORA.md), [model registry](../config/candidates.json), [model journal](../docs/MODEL_JOURNAL.md) and [conversation topologies](../docs/CONVERSATION_TOPOLOGIES.md).

This proposal is entirely an **unexecuted design**. No cited literature or previous neurounner run is evidence that the proposed memory transplant already works.
