# Model longlist

This document is the durable longlist for models that look potentially useful for neurounner but are not all ready for immediate qualification.

The longlist is intentionally broad. Inclusion means **interesting enough to remember**, not endorsed, benchmarked, llama.cpp-compatible, appropriately licensed, or suitable for unattended use. Before moving any entry into `config/candidates.json`, verify the exact revision, runtime support, quantization, license, file hashes, and practical hosted-runner footprint.

## Recently evaluated / promoted — 24 October 2026

These entries have now graduated from speculative longlist status into reproducible local candidates. Detailed metrics and CPU IDs live in `docs/MODEL_JOURNAL.md`.

| Model | Current status | Initial conclusion |
|---|---|---|
| Liquid d1-3B Q4_K_M | active, multimodal verified | interesting decision specialization, but weak as raw visual geometry interpreter; pair with a stronger extractor |
| Qwen3-VL-2B-Instruct Q4_K_M | active, multimodal verified | best cheap visual sidecar in the current simple fixture: 4/6 actions, 72.2% field accuracy, strict JSON compliance |
| Qwen3-VL-4B-Instruct Q4_K_M | active, multimodal verified | 4/6 actions and 66.7% field accuracy; did not beat 2B on the simple fixture |
| LFM2.5-VL-450M Q4_K_M | active, multimodal verified | extremely fast but poor spatial/action grounding; keep as cheap filter/extractor candidate |
| Qwen2.5-Omni-7B Q4_K_M | active, vision+audio verified | correct simple vision, speech transcription and mixed image/audio reasoning; preprocessing is slow on CPU |
| Nemotron 3.5 Bolt E4B Q4_K_M | active, reasoning modes verified | thinking-off is compact; thinking-on reaches a final answer with larger budget and shows orderly self-checking |
| K2-Horizon IQ1_M | active only as degradation reference | severe template/reasoning corruption; reject for serious reasoning at this quant unless later evidence reverses it |
| K2-Horizon Q6_K | active | coherent reasoning; materially better and faster than IQ1_M on the same EPYC 7763 runner |
| QwenStoryteller Q4_K_M | active specialist | coherent/persona-rich but ignores strict embodied/mechanical constraints; use before a mechanical normalizer, not after it |

## Active or already tracked

### Phi-4-mini-instruct

- Source: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Status: **already represented in the active candidate registry** through a pinned GGUF build.
- Interest: compact ~4B-class general model with a distinct training lineage from Qwen/LFM; useful as an independent critic, repairer or long-running incubator.
- Next use: fixed-population panel and adversarial pairings.

## Novel architecture, training method or decision-focused models

### Maria-1B

- Source: https://huggingface.co/Zahid2005/Maria-1B
- Interest: Dynamic Query Adaptation / HyperGQE is unusual enough to justify architectural investigation rather than judging it only as a generic 1B language model.
- Potential neurounner role: adaptive query-conditioned mutator, selector or routing experiment.
- Before testing: determine exact inference stack, whether the HyperGQE behaviour survives available export/quantization paths, and whether a practical CPU runtime exists.

### Crowd-v1

- Source: https://huggingface.co/Vxtzq/Crowd-v1
- Interest: designed to be trained in a distributed fashion, which is directly relevant to neurounner's broader distributed-runner experimentation.
- Potential neurounner role: reference architecture for distributed learning, possible participant model if inference quality is useful.
- Before testing: inspect architecture, training assumptions, model size/runtime support and whether its distributed-training design creates anything useful for our inference ecology.

### d1-3B

- Source: https://huggingface.co/LiquidAI/d1-3B
- Status: **tested and promoted to the active registry** using the official Q4_K_M GGUF plus Q8 projector.
- Interest: 3B decision model rather than merely another instruction-tuned chat model.
- Observed: multimodal inference works, but the cache-isolated six-scene geometry test produced only 2/6 correct immediate actions and poor strict-schema adherence.
- Potential neurounner role: decision-stage selector or policy critic **after** scene state has been extracted into a cleaner structured representation.
- Current priority: high for structured decision tests; lower for raw-image interpretation.

### Zenyx-V3-Base

- Source: https://huggingface.co/Arko007/Zenyx-V3-Base
- Interest: potentially very high token throughput.
- Potential neurounner role: bulk mutation, filtering or long-duration generation where useful work per runner-hour matters more than per-response sophistication.
- Before testing: verify architecture/runtime support and benchmark real CPU decode rather than assuming speed from model claims.

### Zenyx-MCEN-Delta-Mamba

- Source: https://huggingface.co/Arko007/Zenyx-MCEN-Delta-Mamba
- Interest: partially trained ~0.5B model and unusual Mamba/delta-style lineage.
- Potential neurounner role: intentionally incomplete/strange mutation source, architectural-diversity control, or comparison against Sprocket's undertrained failure modes.
- Caveat: partial training may make it useless as a judge while still making it valuable as a source of odd behavioural fragments.

### K2-Horizon-3.7B-Claude-Opus-Reasoning-Distilled mashup

- Source: https://huggingface.co/mradermacher/K2-Horizon-3.7B-Claude-Opus-Reasoning-Distilled-i1-GGUF
- Status: **tested at IQ1_M and Q6_K on the same EPYC 7763 runner; both pinned in the active registry.**
- Interest: deliberately weird model mashup / reasoning-distillation lineage.
- Result: IQ1_M (~1.75 bpw) collapsed into malformed reasoning/template fragments and was also slower in the controlled same-host benchmark. Q6_K produced coherent stepwise reasoning.
- Decision: retain Q6_K as an unusual deepener candidate; retain IQ1_M only as a quantization-degradation reference unless later tests rehabilitate it.

## Character and persona generators

These models may be more useful for generating actor personalities, social quirks, survival instincts or unusual local goals than for producing canonical KernelSpecs.

### gpt-4chan

- Source: https://huggingface.co/wildarrows/gpt-4chan
- Interest: potentially distinctive, chaotic conversational/personality prior.
- Potential neurounner role: character/persona mutation, hostile or antisocial behavioural priors, dialogue/personality experiments.
- Caveat: outputs may contain offensive or low-quality material; any use should be sandboxed behind filtering and never treated as a trusted judge.

### sage-survival-llm

- Source: https://huggingface.co/abdulrehman6617u/sage-survival-llm
- Interest: survival-oriented knowledge/persona could produce distinctive scarcity, threat, shelter, foraging or risk-avoidance instincts.
- Potential neurounner role: survival-character generator or source of environmental heuristics.
- Before testing: inspect actual model base, scale, license and degree of specialization.

### Phi-4-mini-instruct

- Source: https://huggingface.co/microsoft/Phi-4-mini-instruct
- Status: tracked actively; listed here because it may also prove useful as a character model despite its primary role as a general critic.

## RP, theorycrafting and worldbuilding

### QwenStoryteller

- Source: https://huggingface.co/mradermacher/QwenStoryteller-GGUF
- Status: **tested and promoted to the active registry** at Q4_K_M.
- Interest: story-specialized model may retain consequences, motives and emergent narrative better than general instruction models.
- Observed: substantially more coherent and persona-rich than Sprocket, but repeatedly introduced backstory, first-person narration and non-local motives despite explicit constraints.
- Potential neurounner role: upstream persona/worldbuilding source whose output is later stripped down by LFM/another mechanical compiler; poor choice for authoritative KernelSpec generation.

### prism-roleplay-1.5-small-fast

- Source: https://huggingface.co/VertexAGI/prism-roleplay-1.5-small-fast
- Interest: explicitly optimized for roleplay and speed.
- Potential neurounner role: cheap actor personality layer, multi-agent social interaction, long-running conversational ecology.

### prism-creative-1-5-mini

- Source: https://huggingface.co/VertexAGI/prism-creative-1-5-mini
- Interest: creative specialization in a compact model.
- Potential neurounner role: divergent worldbuilding, behaviour mutation, scenario generation and novelty injections.

## Small-model fine-tuning references

These are interesting even if they are not ultimately strong panel members. Their training recipes, data choices and resulting specialization may inform future neurounner fine-tunes.

### amethyst-1-mini

- Source: https://huggingface.co/VertexAGI/amethyst-1-mini
- Interest: reference point for what can be extracted from a deliberately fine-tuned small model.
- Potential neurounner role: architecture/training reference; possibly a lightweight specialist if its behaviour is useful.

### amethyst-1.5-mini

- Source: https://huggingface.co/VertexAGI/amethyst-1.5-mini
- Interest: later/sibling reference for compact specialization.
- Potential neurounner role: compare against amethyst-1-mini to understand what changed with recipe/model iteration.

## Vision and multimodal interpretation

These are potentially important later because neurounner could use visual snapshots/renders as another observation surface rather than relying only on textual state.

They are also good candidates for quantization studies: compare whether smaller quants preserve image interpretation and spatial/mechanical reasoning sufficiently for NPC/world analysis.

### Qwen3-VL-8B-Instruct

- Source: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Interest: higher-capacity vision-language model with potentially useful scene, geometry and interaction interpretation.
- Potential neurounner role: periodic expensive visual reviewer; screenshot/world-state interpretation; visual failure analysis.

### Qwen3-VL-4B-Instruct

- Source: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Status: **tested and promoted to the active registry** with pinned Q4_K_M model + Q8 projector.
- Result: cache-isolated six-scene fixture gave 66.7% field accuracy, 4/6 correct actions and 100% exact schema compliance.
- Comparison: did not outperform Qwen3-VL-2B on this simple task despite higher memory cost.
- Potential neurounner role: visual critic where later, harder tests demonstrate a quality advantage over 2B; otherwise 2B is the better routine sidecar.

### Qwen3-VL-2B-Instruct

- Source: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Status: **tested and promoted to the active registry** with pinned Q4_K_M model + Q8 projector.
- Result: current visual leader on the simple fixture: 72.2% field accuracy, 4/6 correct actions, 100% exact schema compliance.
- Caveat: it showed a strong RIGHT-direction bias and failed the leftward cases, so it is not yet trustworthy as sole action authority.
- Potential neurounner role: cheap visual pre-filter, observation extractor and routine visual sidecar.


## NVIDIA Build free-endpoint and specialist candidates

These candidates are interesting specifically because NVIDIA Build can add a **remote GPU inference layer** to neurounner without requiring the GitHub runner itself to host the model. A free endpoint should be treated as an opportunistic development/research resource rather than guaranteed compute: requests may be rate-limited or throttled, hosted model availability can change, and the NVIDIA trial service may log inputs/outputs.

The preferred architecture is therefore:

```text
local fast ecology
  -> deterministic/schema gate
  -> local LFM/Qwen selection
  -> NVIDIA free endpoint only for:
       * disagreement
       * elite review
       * visual/physical interpretation
       * specialist prediction/reranking
       * periodic frontier-scale critique
  -> record response + provider/model metadata
  -> continue locally
```

Do not send secrets, private repository material, personal data, or other confidential state to trial endpoints. Remote results should be marked as weaker-reproducibility evidence than pinned local GGUF results because the provider may update or retire the hosted backend.

For ordinary chat/reasoning NIMs, NVIDIA exposes an OpenAI-compatible endpoint at `https://integrate.api.nvidia.com/v1`. Specialist models such as rerankers, Kumo predictive models and cuOpt use task-specific APIs and should be wrapped behind separate neurounner provider adapters rather than pretending they are chat models.

### MiniMax-M3.1-Flash — remote frontier/agentic oracle

- Source: https://build.nvidia.com/minimaxai/minimax-m3.1-flash
- NVIDIA availability: free hosted endpoint; no downloadable NVIDIA package on the current Build page.
- Architecture: approximately 428B total / 23B active MoE, native multimodal, long context, configurable reasoning effort.
- Interest: extremely capable remote critic for difficult survivors, long-horizon agentic analysis, multimodal world-state review, and occasional high-quality recombination.
- Potential neurounner role: **rare remote deepener/adjudicator**, especially when local 3–8B judges disagree.
- Caveat: NVIDIA lists this Build-hosted version for research/development use; do not make the ecology depend on continuous availability.

### Nemotron 3.5 Super VL 120B-A12B — remote visual/deep reviewer

- Source: https://build.nvidia.com/nvidia/nemotron-3.5-super-vl-120b-a12b
- NVIDIA availability: free hosted endpoint; current Build page does not offer download.
- Architecture: roughly 120B total / 12B active, multimodal text/image/video, configurable reasoning, very long context.
- Interest: unusually strong fit for reviewing visual Cybersand snapshots, short videos, interaction traces, diagrams and long experiment packets.
- Potential neurounner role: **elite multimodal reviewer**, visual failure analyst, disagreement breaker, and periodic population-level critic.
- Priority: **very high for cloud integration** because it gives us a capability class that four-core CPU runners cannot realistically reproduce.

### Nemotron 3 Ultra 550B-A55B — remote frontier adjudicator

- Source: https://build.nvidia.com/nvidia/nemotron-3-ultra-550b-a55b
- NVIDIA availability: free hosted endpoint; model is also downloadable in forms intended for substantial NVIDIA GPU infrastructure.
- Architecture: approximately 550B total / 55B active hybrid MoE/Mamba reasoning model with up to 1M context.
- Interest: frontier-scale reasoning is useful precisely where local models disagree or repeatedly sanitize away a strange but promising mechanism.
- Potential neurounner role: **very sparse arbitration**, deep audit of elite kernels, long-context review of an entire experiment epoch, or generation of new research directions.
- Do not use for routine traffic; its value is as a qualitatively different judge rather than another high-volume participant.

### DeepSeek V4.1 Flash — remote multimodal heavyweight

- Source: https://build.nvidia.com/deepseek-ai/deepseek-v4.1-flash
- NVIDIA availability: downloadable and free hosted endpoint.
- Architecture: approximately 552B total / 8B active MoE with text+image input and long context.
- Interest: enormous resident model with a surprisingly small active parameter count; useful comparison against Nemotron Super/Ultra for whether different frontier families preserve different behavioural novelty.
- Potential neurounner role: remote multimodal judge, alternate frontier critic, cross-family tie-breaker and long-context population reviewer.
- Priority: **high** for independent-panel experiments against Nemotron rather than serial conversation.

### Cosmos3-Nano-Reasoner — physical-world visual reasoner

- Source: https://build.nvidia.com/nvidia/cosmos3-nano-reasoner
- NVIDIA availability: downloadable and free hosted endpoint.
- Interest: explicitly built for physical AI/robotics and structured reasoning over images/video, including space, time and physical-world behaviour.
- Potential neurounner role: **visual physics critic** for Cybersand frames/video; turn observed scenes into local mechanical hypotheses; review whether an NPC behaviour appears physically plausible or produces unexpected emergent motion.
- Priority: **very high**. This is more directly aligned with the project than a generic VLM.
- Local/fine-tune interest: strong, subject to practical hardware/runtime requirements and model size.

### Nemotron 3.5 Bolt E4B — local/cloud compact reasoner

- Source: https://build.nvidia.com/nvidia/nemotron-3.5-bolt-e4b
- Status: **local Q4_K_M tested and promoted to the active registry**; hosted NVIDIA endpoint comparison still pending.
- NVIDIA availability: free hosted endpoint; full-precision weights are available for customization/post-training.
- Result: thinking-off produced a compact final immediately; thinking-on used a longer explicit self-check chain and successfully reached a final answer when given sufficient budget.
- Interest: strong constraint tracking, distinct from Qwen/LFM families.
- Potential neurounner role: high-priority reasoning critic, long-running incubator, and eventual hosted-vs-local parity experiment.

### Nemotron 3.5 Rerank 8B — multimodal relevance gate

- Source: https://build.nvidia.com/nvidia/nemotron-3_5-rerank-8b
- NVIDIA availability: free hosted endpoint.
- Output: relevance logits rather than generative prose.
- Interest: text+image reranking can cheaply decide **which image/frame/document fragment is worth sending to an expensive reasoning model**.
- Potential neurounner role: frame/snapshot relevance gate, experiment-evidence selector, candidate-memory retrieval reranker, and visual RAG component.
- Strong proposed topology:
  `many screenshots/frames -> rerank against current behavioural question -> top few -> Cosmos/Super-VL reasoning`.
- Priority: **high specialist** because it can reduce expensive multimodal reasoning calls.

### Kumo Tabular — structured world-state predictor

- Source: https://build.nvidia.com/nvidia/kumo-tabular
- NVIDIA availability: downloadable and free hosted endpoint.
- Model family: tabular foundation model for in-context classification/regression, with small/medium/large variants around 28M/62M/215M parameters.
- Interest: world, actor, material or strategic state could be projected into rows/columns and used for direct predictions without converting everything into prose.
- Potential neurounner roles:
  - predict likely behaviour success/failure from structured episode summaries;
  - classify promising vs unpromising kernels from measured features;
  - forecast actor outcomes or experiment metrics;
  - serve as a non-language control for whether structured prediction beats LLM intuition.
- Priority: **high experimental specialist**, especially once neurounner accumulates structured evaluation data.

### Kumo Relational — graph/multi-table world predictor

- Source: https://build.nvidia.com/nvidia/kumo-relational
- NVIDIA availability: downloadable and free hosted endpoint.
- Interest: relational prediction over multi-table/graph-like structured data may map even better than flat tables to actors, contacts, materials, regions, interactions and historical episodes.
- Potential neurounner role: predict outcomes from actor↔environment↔material relations, identify risky/interesting interaction structures, or provide a structured strategic layer above individual behaviour kernels.
- Priority: **high once we have relational episode data**; less useful before the project records rich structured relationships.

### NVIDIA cuOpt — deterministic route/decision-optimization sidecar

- Source: https://build.nvidia.com/nvidia/nvidia-cuopt
- Classification: **not an LLM/model participant**; it is an open-source GPU-accelerated optimization engine.
- Current Build role: downloadable/self-hosted optimization rather than a conversational free-endpoint oracle.
- Interest: pathfinding, constrained routing, assignment and rerouting are useful as a deterministic specialist against which learned/generated navigation ideas can be compared.
- Potential neurounner role:
  - generate route/assignment ground truth or strong baselines;
  - solve strategic multi-agent routing problems;
  - provide an oracle for evaluating whether an NPC's locally emergent route is efficient;
  - generate difficult route-planning scenarios for behavioural actors.
- Keep on the longlist, but implement behind an **optimizer adapter**, not the model/chat provider abstraction.

### Proposed NVIDIA provider integration

A future neurounner provider layer should separate these interface classes:

1. **`nvidia_chat`** — OpenAI-compatible chat/reasoning/multimodal models using `integrate.api.nvidia.com/v1`.
2. **`nvidia_rerank`** — query/document or query/image relevance scoring.
3. **`nvidia_predictive`** — Kumo tabular/relational prediction APIs.
4. **`nvidia_optimizer`** — cuOpt or other deterministic solver interfaces.

For every remote call record at minimum:

- provider and exact model identifier;
- request hash and experiment/candidate IDs;
- request parameters including reasoning mode/effort;
- UTC timestamp;
- response text or structured output;
- token/usage metadata when exposed;
- retry/429/timeout events;
- whether the call was free-trial hosted or self-hosted;
- a `remote_unpinned_backend=true` provenance flag.

### Suggested cloud/local ecology

```text
GitHub CPU runner
  |
  +-- local Sprocket / LFM / Qwen / Gemma etc.
  |     generate + normalize + cheap critique
  |
  +-- deterministic validators
  |
  +-- if visual evidence:
  |       Nemotron rerank -> top frames -> Cosmos3 Nano or Nemotron Super VL
  |
  +-- if structured prediction:
  |       Kumo Tabular / Relational
  |
  +-- if route optimization:
  |       cuOpt baseline/oracle
  |
  +-- if elite/disputed kernel:
          MiniMax M3.1 Flash
          DeepSeek V4.1 Flash
          Nemotron Super VL
          Nemotron Ultra
          -> independent remote judgments
          -> local deterministic aggregation
```

The remote models should normally **not talk directly to one another**. Give them the same immutable packet independently, then aggregate locally. This preserves model-family disagreement and prevents one frontier model from anchoring the others.

## Ultra-small training candidates — 24 October 2026

These models are being tracked specifically for **fast CPU fine-tuning experiments on standard GitHub-hosted four-vCPU runners**. The priority here is not raw general-model quality; it is how quickly we can observe a measurable behavioural change, checkpoint it, resume it on a fresh runner, and learn what kinds of specialization survive at very small scale.

### SmolLM2-135M-Instruct — TRAIN-001 passed

- Source: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- Size: ~135M parameters.
- License: Apache-2.0.
- Architecture/runtime: standard decoder-only Transformers model; normal Hugging Face CPU path and chat template.
- Status: **TRAIN-001 completed successfully across two separate GitHub-hosted CPU jobs.**
- LoRA result: 2.44M trainable parameters; ~117 training tokens/s; ~1.67 GiB peak RSS; ~29.7 MB resumable adapter+optimizer checkpoint.
- Behavioural result: held-out exact five-key KernelSpec schema improved from 0/6 before training to 5/6 after 24 steps and 6/6 after a fresh-runner resume to step 40.
- Caveat: semantic/mechanical quality remained mediocre despite excellent format learning; the next dataset must train causal mechanics and failure modes rather than merely JSON shape.
- Current role: proven ultra-cheap specialization substrate and checkpoint/resume reference.
- Detailed evidence: `docs/TRAIN_001_CPU_LORA.md`.

### void.1 — symbolic 90M control

- Source: https://huggingface.co/dotlabs/void.1
- Size: ~90.15M parameters.
- License: Apache-2.0.
- Architecture/runtime: standard Transformers text-generation model with Qwen3 lineage/tokenization metadata; pretrained from scratch with a symbolic-reasoning emphasis.
- Training fit: **very good**, and smaller than SmolLM2-135M.
- Interest: unusually relevant as a tiny symbolic/mechanical reasoning substrate; could eventually be compared against SmolLM2 on the same compiler dataset.
- Caveat: base model rather than instruction-tuned, so the first specialization may require more examples or a completion-style objective.
- Priority: **high; likely TRAIN-002** if TRAIN-001 works.

### StorySprint-30M — minimal training-speed control

- Source: https://huggingface.co/AbdullahKhan10/storysprint-30m
- Size: ~30M parameters.
- License: MIT.
- Architecture: GPT-2-style decoder, 6 layers, 6 heads, hidden size 384, 256-token context.
- Provenance: trained from scratch on TinyStories; the published baseline reached its target validation loss in under an hour on one NVIDIA T4.
- Training fit: **excellent as a minimal CPU/checkpoint/resume control**, but its extremely narrow TinyStories prior makes it a weak candidate for the real behaviour-kernel compiler.
- Potential neurounner role: prove the training orchestration itself, or deliberately generate tiny/simple character motifs.
- Priority: fallback speed control if 135M CPU throughput is unexpectedly poor.

### AnveshAI-CoderLM-0.1B

- Source: https://huggingface.co/AnveshAI/AnveshAI-CoderLM-0.1B
- Current published model card/search result: approximately 100.2M parameters, Apache-2.0, compact code model with a custom hybrid ParallelMinGRU + Differential-GQA architecture and ~512-token context.
- Interest: the recurrent/attention hybrid is architecturally distinct and small enough for CPU experiments.
- Training fit: **moderate**, not first choice. It relies on custom model code and is specialized for code/FIM rather than instruction following, so PEFT/Trainer compatibility needs a dedicated spike.
- Potential neurounner role: compact structured-code/kernel emitter if the custom architecture fine-tunes cleanly.

### Zenyx-100M-Base

- Source: https://huggingface.co/Arko007/Zenyx-100M-Base
- Status: **longlist only pending exact artifact verification**. The exact supplied repository did not resolve cleanly during the 24 October review, while nearby Zenyx nano models use custom JAX/Flax architectures.
- Interest: potentially useful ultra-small base-model/control if the exact checkpoint and runtime are recoverable.
- Training fit: unknown until architecture, parameter count, files and license are verified.
- Do not substitute a similarly named Zenyx model silently; preserve this exact longlist entry until the target repository can be inspected.

### SmolVLM-256M-Instruct

- Source: https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- Size: approximately 0.3B total; based on Idefics3 with a 93M vision encoder and SmolLM2 language component.
- License: Apache-2.0.
- Architecture/runtime: standard Transformers multimodal model; Hugging Face publishes an explicit fine-tuning path.
- Training fit: **high for a later multimodal specialization experiment**, but not appropriate as the first CPU-training proof because vision preprocessing/training adds extra memory and compute variables.
- Potential neurounner role: tiny screenshot/frame classifier or specialized local physical-state extractor fine-tuned on synthetic Cybersand scenes.
- Priority: train after the text-only checkpoint/resume pipeline is proven.

### Immediate training order

1. **SmolLM2-135M-Instruct** — real narrow specialist; standard tooling.
2. **void.1** — next preferred training comparison: smaller symbolic base model on the same compiler dataset.
3. **StorySprint-30M** — minimal orchestration/speed control if needed.
4. **SmolVLM-256M-Instruct** — multimodal specialization after text training is stable.
5. **AnveshAI-CoderLM-0.1B** — custom-architecture experiment.
6. **Zenyx-100M-Base** — pending exact artifact verification.

## Suggested future qualification dimensions

When longlist entries graduate into actual tests, avoid reducing evaluation to chat quality. Depending on model type, useful measurements include:

- CPU prompt/decode throughput on the standard four-vCPU runner;
- resident RSS and startup time;
- quantization sensitivity;
- structured KernelSpec validity;
- novelty retained after normalization;
- interesting failures generated per 10k tokens;
- consistency as a judge;
- willingness to destroy weak concepts rather than endlessly elaborate them;
- character distinctiveness and persistence;
- ability to translate world/vision input into local mechanical observations;
- performance during a multi-hour persistent incubator job;
- whether unusual architectural features survive GGUF/export/quantization at all.

## Promotion rule

An entry moves from this longlist into `config/candidates.json` only after we identify a concrete runnable artifact and pin:

1. exact model/revision;
2. exact file or conversion recipe;
3. SHA-256;
4. runtime requirements;
5. intended experiment role.

That keeps the longlist exploratory while the active registry remains reproducible.
