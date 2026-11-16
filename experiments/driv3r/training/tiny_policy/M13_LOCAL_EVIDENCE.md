# M13: host-goal-conditioned tiny game skills — local executed evidence (16 November 2026)

**Disposition: REJECT for promotion.** This is synthetic model-side training only; no CyberSand native/Workbench/materials/game/provider source was changed. PR #634 stays Draft/HOLD, unmerged.

## Hypothesis and explicit authority boundary

Test whether 5 additional *host-supplied* waypoint/progress scalars (167 float32 inputs total, with a 12-state causal history) enable low-latency learned legal decisions and deliberate geometric textures. The waypoints/shape sequences are a deterministic privileged **synthetic host design**, not independently invented by any model. Families: Greebles (teeth/retraces), Pipework (orthogonal elbows), Architecture (room/rectangle loop). World is 96x96, 96-action budget; action set UP/DOWN/LEFT/RIGHT/STOP with blocked/protected-cell masking. Train/Dev/Eval world seeds are disjoint, but EVAL has been directly inspected and is not sealed native FINAL.

Three distinct, real scratch architectures trained with masked cross-entropy/AdamW for four CPU epochs, **196 optimizer updates per model / 588 total**, on 26,502 TRAIN and 6,634 DEV state-action examples. Teacher labels use privileged A* shortest-goal routing; inference only sees the local planes and the current host-provided waypoint vector, never the global planning path. MLP 22,661 params, GRU 31,493, TCN 29,189. No pretrained 20M–300M backbone, LoRA or model-teacher distillation in this experiment.

## Actual 12 unseen-seed / three-family / 96-budget results

| Policy | Mean unique cells | Mean branch nodes | Mean completed host waypoints | Early STOP / 36 |
|---|---:|---:|---:|---:|
| Privileged A* teacher | 74.667 | 2.333 | **9.611** | 18 |
| Deterministic local goal-greedy | 31.528 | 1.000 | **4.194** | 3 |
| Novel-cell greedy (ignores goals) | **97.000** | 0.000 | 0.861 | 0 |
| Random-legal | 46.333 | 15.667 | 0.083 | 0 |
| Trained MLP | 14.389 | 0.278 | 0.944 | 0 |
| Trained GRU | 61.917 | 2.250 | 1.222 | 0 |
| Trained TCN | 43.250 | 3.639 | **1.917** | 0 |
| Untrained MLP | 1.000 | 0.000 | 0.000 | 36 |

The TCN's 86.102% DEV teacher-action agreement produces only 1.917/approximately 10 completed goals, substantially below the simple goal-greedy baseline. MLP DEV agreement 80.615%, GRU 77.404%. **Open-loop imitation did not translate to closed-loop waypoint recovery**. TCN minus local goal-greedy: -2.278 goals, seed-cluster bootstrap 95% CI [-3.556, -1.111]. No winner or game-artist promotion.

Coverage can be deceptive: novelty achieves 97 unique cells, 0 traversed-edge branch nodes, and fewer than one intended target per run. Random's high branch count likewise is not deliberate drawing.

## Evidence and source limitations

- 288 original PNG/JSON receipt pairs; **23,900 original host actions** independently recomputed from loaded checkpoints and replayed.
- All 583 non-manifest SHA-256 entries matched, all 288 PNG pixels matched reconstructed worlds, 4 action/legal/observation/goal-tamper tests rejected mutations, and 5 model-host/causality unit tests passed.
- Repeated full execution: **582/584 files byte-identical**. Only report.json and manifest.json changed because unpinned, non-deterministic wall-clock latency telemetry changed; checkpoints, receipts and rendered pixels matched exactly.
- Median local CPU batch-one, 12-step *forward only*: MLP 0.0392ms, GRU 0.2159ms, TCN 0.2816ms. Excludes host/serialization/raster; not an end-to-end SLA.
- Original executed Python sources (sha256): m13_goal_skill.py `584e15e223258df051769844ccef36b5907e40e830cf68217520dec81a19de2f`; audit_m13_goal_skill.py `b59bab50ab2b9c1b4957b30d163e140e926bf0ef8dd10082f7786569ba880c21`; test_m13_goal_skill.py `05c53cbfe62489c76efb456c9efc4fa9096f9c723a5d720258cd2ad1824628ba`.
- Original evidence ZIP SHA256 `0ff00c1d356e5dda75ef25f01d82f40011eaaf125422a4bf777ed3c6d3c9f201`. Kept outside Git (model/checkpoint binary files must never enter Git history).
- **Not source-qualified in GitHub Actions:** this run used an archived local `core.py`, and the executed M13 scripts/checkpoints have not been transferred to the PR branch or uploaded as immutable Actions artifacts. Do not equate this local run with exact-HEAD qualification; remote M11 is separately exact-HEAD qualified.

## Next discriminating step

Transfer exact tested source under this owned directory, run the model training/auditor as one bounded [M13] job in the existing isolated Actions workflow, retaining failure outputs and checkpoints in an immutable artifact. Then test *goal-progress constrained action decoding* on genuinely new topologies (not the already observed 63000–63011 group) and compare with goal-greedy and A* controls. Branch/style statistics need human inspection and should not be treated as native authoring acceptance.
