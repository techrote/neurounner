# MZ-003: RWKV-7 0.1B genuine pretrained recurrent game-action probe

**Research-only / Draft-HOLD**, stacked on MZ-001/002 PR #642. This is a new model family with generalized-delta-rule recurrence, distinct from Mamba-130M and existing d1/LFM/Sprocket/Smol.

Checkpoint: https://huggingface.co/Hakureirm/rwkv7-0.1b-hf at immutable revision 16136bb51c0cf16356f410956d38b48f249c73e1. The model ships Python remote modeling code. The runner downloads three Python source files from the exact pinned revision, records SHA-256 and performs limited AST import/dynamic-exec screening. **This is not a full malicious-code security audit.** The audited revision executes only inside an isolated GitHub CPU research runner; no production game loading.

## Actual experiment gate

- Disjoint TRAIN/DEV/TEST seeds 400+/900+/1200+, source-equivalent external synthetic MZ-002 11x11 game worlds and unchanged 114-feature cheap comparator.
- Real pinned pretrained model forwards captured through a hook on the output projection's input; no fake feature vectors. Frozen backbone.
- Same synthetic samples and 140 real optimizer steps for an RWKV-hidden-state seven-action head and a 114-feature structured baseline; independent majority comparator.
- Held-out goal-swap tests where only target changes, raw invalid action rate before host masking, CPU latency p50/p95, peak RSS and four matched novel worlds each for RWKV, structured, random non-STOP and heuristic teacher.
- Original PNG/JSON traces, TRAIN/DEV/TEST corpus fingerprint, model/source SHA and head checkpoint in a 14-day Actions artifact.
- Fail closed on missing code/weights/optimizer/evidence. Source tests alone do not count as pretrained inference or real training.

This frozen-backbone probe does not test fine-tuning RWKV. A negative result rules out promotion of this specific head-only approach. Mamba MZ-002 used different training sizes; do not rank architectures directly without a matched corpus.

**Hard boundary:** never modify CyberSand game code or content, Godot, Workbench, native simulation, materials, maps, assets or production defaults. No merge or Ready. No native PAINT receipts claimed.
