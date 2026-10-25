# neurounner

CPU-only micro-model inference experiments on standard GitHub-hosted Linux runners.

The immediate goal is to characterize small local models as generators, critics, and conversational participants for later behaviour-research work. This repository is deliberately separate from Cybersand and from formal NPC training/evaluation pipelines.

> Start with [RAG.md](RAG.md) for the retrieval/authority map. Generated corpora must follow the [dataset filing system](docs/DATASET_FILING_SYSTEM.md); Actions artifacts are transient staging only.

## Current pinned stack

- Runner: `ubuntu-24.04`
- llama.cpp: `b11852`, commit `7894e8b92481a8bb8ffe94f73b656ffe0dd01e10`
- Ubuntu x64 CPU archive SHA-256: `6ec84fd5ddf5bee393d27384b9b6acb19a35cbac71b0d22b4d73fc727363fcca`
- Sprocket-500M chat Q4_K_M revision: `eb157d4ddddc4619e661d4a2426bb2dd000fc668`
- Sprocket GGUF SHA-256: `edc2ac323a2513fa6f8e3e54adf120243de0f7a6b8cc11b349b3e6ad61494a09`
- LFM2.5-1.2B-Instruct QAD-Q4_0 revision: `8ed288026e23958ad9dfa92d53ed773a8eee7125`
- LFM GGUF SHA-256: `bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1`

## Experiments

The first suite contains four bounded experiments:

1. **sprocket3** — three independent Sprocket processes share one four-vCPU runner.
2. **lfm1** — one LFM2.5-1.2B process uses the runner.
3. **pair** — Sprocket and LFM are both resident on one runner and infer concurrently.
4. **ring** — Sprocket A → Sprocket B → Sprocket C → LFM, repeated for a short circular conversation.

Runtime and model files are verified before execution and cached by immutable content identifiers. No model or runtime binary is committed to Git.

Use **Actions → Prime inference cache** once after changing pins, then **Actions → Model experiments** to run the suite.

## First hosted result

The complete initial suite passed on GitHub-hosted four-vCPU Linux runners at source head `0e0163856493334421b6eda748f63cf728df72b6`.

| Experiment | Result |
|---|---:|
| 3 × concurrent Sprocket | 464 completion tokens in 6.542 s — 70.93 tok/s aggregate |
| LFM2.5-1.2B alone | 246 tokens in 5.232 s — 47.02 tok/s |
| Sprocket + LFM concurrent | 278 tokens in 4.540 s — 61.24 tok/s aggregate |
| 3 × Sprocket + LFM ring | 649 tokens in 21.622 s — 30.02 tok/s across eight turns |

The four-process ring kept all models resident simultaneously with large memory headroom. See [the initial hosted experiment record](docs/INITIAL_EXPERIMENTS.md) for outputs, memory observations, artifact hashes, and next measurements.

## Larger-model qualification

The second hosted suite passed at source `f7d6d08b555498fb9fc1b24ba0bba8ea0d92d7e8`. It added a three-Sprocket → LFM structured foundry experiment and qualified six CPU GGUF candidates.

| Model | llama-bench tg64 | Server RSS | First assessment |
|---|---:|---:|---|
| LFM2.5-VL-450M | 129.54 t/s | 0.447 GiB | extremely cheap text worker; weak first kernel |
| LFM2.5-1.2B-Instruct | 63.98 t/s | 1.299 GiB | current routine selector/compiler |
| LFM2.5-1.2B-Thinking | 29.43 t/s | 1.260 GiB | viable but reasoning-budget limited |
| LFM2.5-8B-A1B | 31.52 t/s | 8.611 GiB | strong candidate for periodic deeper review |
| Qwen3.5-2B | 23.95 t/s | 2.013 GiB | viable alternate critic |
| Qwen3.5-4B | 6.59 t/s | 5.533 GiB | much slower; potentially useful for diversity/quality probes |

See [the second hosted experiment record](docs/SECOND_EXPERIMENTS.md) for exact outputs, caveats, model pins and artifact hashes.

## Durable datasets

- [Dataset index](datasets/README.md)
- [BABBLE-001 durable archive](datasets/babble-001/README.md) — 12 thematic guides, 48 exact Sprocket conversations, all retained MiniMax compile/repair provider history, and 12 final personality/training packets.
- [Dataset filing system](docs/DATASET_FILING_SYSTEM.md) — mandatory preservation/provenance contract for future generated corpora.

## Design notes

- [RAG / retrieval authority map](RAG.md)
- [Dataset filing system](docs/DATASET_FILING_SYSTEM.md)
- [Conversation and orchestration topologies](docs/CONVERSATION_TOPOLOGIES.md)
- [Model storage policy](docs/MODEL_STORAGE.md)
- [Future model longlist](docs/MODEL_LONGLIST.md)
- [Model performance and behaviour journal](docs/MODEL_JOURNAL.md)
- [Initial hosted experiment record](docs/INITIAL_EXPERIMENTS.md)
- [Second hosted experiment record](docs/SECOND_EXPERIMENTS.md)
- [Eight-runner startup smoke](docs/STARTUP8_2026-10-24.md)
- [TRAIN-001 CPU LoRA checkpoint/resume proof](docs/TRAIN_001_CPU_LORA.md)
- [VILLAGE-001 cybernetic goblin foundry](docs/VILLAGE_001_CYBERGOBLIN_FOUNDRY.md) — 36-member Sprocket-derived SLM population, 72 planned CPU training lanes, NVIDIA compiler/critic panel and hierarchical blind qualification.

The current ~4B comparison shortlist is pinned in [config/candidates.json](config/candidates.json): Phi-4-mini-instruct, Gemma 3 4B IT, Ministral 3 3B Instruct, SmolLM3 3B and Qwen3-4B, alongside Qwen3.5-4B.

## CPU fine-tuning proof

TRAIN-001 proved that a 135M instruction model can be specialized with LoRA on standard four-vCPU GitHub-hosted runners and resumed on a fresh runner.

Observed on SmolLM2-135M-Instruct:

- ~117 training tokens/s;
- ~1.67 GiB peak RSS;
- 2.44M trainable LoRA parameters;
- ~29.7 MB complete resumable checkpoint;
- held-out exact KernelSpec schema improved from 0/6 to 5/6 after 24 steps and 6/6 after a fresh-runner resume to step 40.

The format adaptation was rapid, but semantic/mechanical quality is not yet authoritative. See [TRAIN-001](docs/TRAIN_001_CPU_LORA.md) and the [model journal](docs/MODEL_JOURNAL.md).
