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
