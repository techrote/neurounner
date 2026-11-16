# MZ-004 — Pythia-70M pretrained transformer for CyberSand-style decisions

Research-only, **Draft/HOLD**. Stacked on MZ-003 (RWKV-7 PR #645), which itself stacks on MZ-001/002 (#642). Source-only gameplay experiments: game code/assets/content are READ-ONLY, never modified.

**Pretrained model:** [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped), immutable revision `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`, Apache-2.0. Standard GPT-NeoX architecture via Transformers; **no custom remote code**.

## Real experiments and fairness

Use precisely the same 494 TRAIN / 140 DEV / 137 TEST construction as MZ-003, with disjoint world seeds 400+/900+/1200+ and same synthetic teacher and 114-feature structured comparator.

- Baseline A: pretrained **frozen** GPT-NeoX hidden features, trained 7-action head, 140 AdamW full-batch steps.
- Baseline B: trained 114-feature head, same 140 steps.
- Baseline C: pretrained frozen features + **48 extra minibatch updates** of head only.
- Experimental D: same initial trained head, same 48 extra minibatches, **real backpropagation through the pretrained final transformer block**. Only final block attention output projection and post-attention LayerNorm train; all other pretrained weights stay frozen. Tune and frozen-extra start with identical head weights and see identical minibatches.
- Baseline E: constant-majority. Four unseen world rollouts compare tuned/frozen/structured/random-non-STOP/scripted teacher.

Measure per-action recall, balanced accuracy, macro-F1, paired goal-change **both correct**, predicted switch rate, raw illegal actions before host masking, CPU p50/p95, peak RSS and original JSON/PNG. Record source and model commits, exact trained parameter names, checkpoint SHA-256, loss curves and trace evidence in 14-day GitHub Actions artifacts.

This experiment isolates whether small **actual backbone updates** can improve matched game-goal discrimination over an equally trained frozen head; it does NOT test full SFT, claim broad generalization, or compare architectural quality independently of model sizes and training budgets. The experiment is NOT native Godot PAINT.

## Boundaries

Own only `experiments/driv3r/model_zoo/pythia70/**` and its dedicated workflow. No edits to CyberSand native/Godot, Workbench, terrain/materials/maps/game content, simulations, production providers or existing MZ-001/003 files. Never merge/promote Ready. No paid API/GPU.
