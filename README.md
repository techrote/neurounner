# neurounner

CPU-only micro-model inference experiments on standard GitHub-hosted Linux runners.

The immediate goal is to characterize small local models as generators, critics, and conversational participants for later behaviour-research work. This repository is deliberately separate from Cybersand and from formal NPC training/evaluation pipelines.

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
3. **pair** — Sprocket and LFM are both resident on one runner and take alternating turns.
4. **ring** — Sprocket A → Sprocket B → Sprocket C → LFM, repeated for a short circular conversation.

Runtime and model files are verified before execution and cached by immutable content identifiers. No model or runtime binary is committed to Git.

Use **Actions → Prime inference cache** once after changing pins, then **Actions → Model experiments** to run the suite.
