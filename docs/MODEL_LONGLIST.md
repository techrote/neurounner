# Model longlist

This document is the durable longlist for models that look potentially useful for neurounner but are not all ready for immediate qualification.

The longlist is intentionally broad. Inclusion means **interesting enough to remember**, not endorsed, benchmarked, llama.cpp-compatible, appropriately licensed, or suitable for unattended use. Before moving any entry into `config/candidates.json`, verify the exact revision, runtime support, quantization, license, file hashes, and practical hosted-runner footprint.

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
- Interest: 3B decision model rather than merely another instruction-tuned chat model.
- Potential neurounner role: decision-stage selector, behavioural action chooser, policy critic or bridge between language-generated kernels and explicit decisions.
- Priority: **high**, because its stated purpose maps unusually well onto behaviour research.

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
- Interest: deliberately weird model mashup / reasoning-distillation lineage.
- Requested test: try both approximately **1-bit-class** and **6-bit-class** quantizations if practical, specifically to see whether the strange character/reasoning properties survive aggressive quantization.
- Potential neurounner role: deepener, adversarial critic, unusual recombination model, quantization-sensitivity study.
- Caveat: choose exact quant files only after inspecting the repository and their real sizes/runtime support.

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
- Interest: story-specialized model may retain consequences, motives and emergent narrative better than general instruction models.
- Potential neurounner role: turn mechanical kernels into plausible character/world consequences; generate scenario tests and interaction chains.

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
- Interest: likely a better initial balance between multimodal capability and four-core runner practicality.
- Potential neurounner role: routine visual critic if CPU/quantized throughput is tolerable.
- Priority: **high among vision candidates**.

### Qwen3-VL-2B-Instruct

- Source: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Interest: smallest of this vision trio and therefore potentially suitable as a cheap image/world interpreter.
- Potential neurounner role: visual pre-filter, object/state extraction, cheap scene classification, or high-volume visual observation conversion.
- Priority: **high for first multimodal qualification**.


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
- NVIDIA availability: free hosted endpoint; full-precision weights are available for customization/post-training.
- Interest: compact reasoning/tool model derived from Gemma-family work, with published results that make it directly comparable to Qwen3.5-4B.
- Potential neurounner role: **high-priority local fine-tune/quantization candidate**, independent 4B-class critic, long-running incubator, and cloud-vs-local parity experiment.
- Particularly interesting experiment: quantize to GGUF and compare the same fixed population through hosted BF16 versus local CPU quantizations.

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
