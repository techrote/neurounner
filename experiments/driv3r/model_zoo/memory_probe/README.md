# MZ-008 — Delayed material cue / hazard inversion game-memory probe

**Research only — Draft/HOLD.** No edits to CyberSand native code, Godot, Workbench, game scenes/scripts, content, physics, maps, materials, assets or production provider settings.

## Why this is necessary

MZ-007b's 18.78M GRU reached 76.28% held-out synthetic teacher agreement, yet did **slightly better** with history erased. That cannot justify a memory-capable game controller. This experiment changes the training/evaluation *task*, not the game.

Each nine-step synthetic encounter has a temporary **material beacon** (ticks 0–1), a temporary **hazard inversion rule** (ticks 4–5), and a current **navigation versus authoring mode** (tick 8). The beacon and inversion cannot be read from the final observation. The host-legal target action (LEFT, RIGHT, BRANCH, or RETRACE) depends on **the XOR of both historical signals plus current mode**. STOP remains host-legal. All eight counterfactuals of each same-background world are grouped in one TRAIN, DEV or TEST partition. The test suite asserts that the final observation is identical across beacon/inversion counterfactuals for the same current mode.

The **177,799-parameter, two-layer GRU** is intentionally below the zoo's typical 20M–1B range. It is an inexpensive mechanism probe that should determine whether scaling recurrent memory training deserves further CPU budgets. It is **not** an actual CyberSand navigation or painting policy. A **5,191-parameter current-state MLP** gets exactly the same batches and optimizer updates, but not comparable FLOPs or capacity.

## Essential causal controls

After 550 real optimizer updates, measure independent held-out TEST scores for the trained full-history GRU, current-only control, and the *same trained* GRU with all previous observations erased, every presentation of the beacon removed, or every presentation of the inversion removed. A fourth intervention flips both earlier beacon appearances without touching the current observation; a competent policy must flip its chosen action. Report paired both-correct and action-switch metrics, raw illegal proposals, true unbatched CPU latency, peak process RSS, loss history, exact source/checkpoint hashes, and independently reload/re-score saved weights.

**Pilot correction:** an early unqualified ablation erased only the first of two historical cue appearances, causing false positive ablation scores. Both repeated cue appearances are now scrubbed and explicitly covered by a regression test.

## Execution

From repository root:

```bash
python -m unittest -v experiments.driv3r.model_zoo.memory_probe.test_memory
python -m experiments.driv3r.model_zoo.memory_probe.run --out mz008-evidence \
  --train-worlds 96 --eval-worlds 24 --updates 550 --batch 64
```

The dedicated workflow uses CPU-only pinned Torch, checks checkpoint provenance, independently reproduces held-out predictions and retains trained artifacts for 14 days. Do not merge/mark Ready or register this model in production. A synthetic short-memory test, even if passed, is insufficient to establish long-horizon navigation, artistic usefulness or transfer to native CyberSand; that requires richer partially observable game-world validation.
