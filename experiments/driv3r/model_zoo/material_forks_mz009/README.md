# MZ-009 — Three-fork partially observed material game-policy experiment

**Research only / Draft/HOLD.** Absolutely no changes to CyberSand game code, Workbench, Godot, native engine, materials, maps, assets or production settings. The procedural host is external and synthetic; outcomes must not be described as native gameplay.

## Motivation and falsifiable prediction

MZ-008 proved short delayed-cue memory in an artificial XOR classification task. MZ-007b learned current-state actions but did **not** use history. MZ-009 puts the history requirement into **closed-loop trajectories** with real action consequences: a temporary material beacon at ticks 0–1, a temporary hazard inversion at ticks 2–3, eight initial corridor moves, and three later material forks separated by six corridor moves. The player must choose the lane with the permitted material; the hidden rule is absent at the forks. A wrong choice collides with protected material and requires RETRACE before trying again. STOP is host-legal but ends a route early if the three paint targets are incomplete. Agent observations include local fork materials, position/stage, legal actions and last action, but never expose hidden safe material at decision time.

This is a **procedural environment with live action-driven state transitions**, not simply independent action classification. The source evaluates teacher-forced labels *and* genuinely executed, host-masked trajectories; collisions and zero-collision completions must be reported separately. Merely eventually reaching three forks by repeatedly crashing is not equivalent to learning the remembered rule.

## Trained models and qualification

From-scratch two-layer GRU (177,605 parameters) versus same-minibatch current-only MLP (4,933 parameters). The GRU is intentionally below the zoo's ~20M–1B priority range, serving as a mechanism probe before allocating budgets to the ~19M recurrent model. These are **not** FLOP- or capacity-matched. Three train trajectories per counterfactual world include a clean path and induced wrong turns at either of two forks, so recovery is observable and trained. Each source world's four beacon/inversion combinations and three injected trajectories remain in the **same** TRAIN/DEV/TEST split. 160 train seeds, 24 dev/test seeds each; 800 real updates; 48 sampled episodes per minibatch.

The test suite asserts no historical clue leakage into the first-fork current observation, exact legal-action mask behaviour, wrong-material RETRACE, seed disjointness, complete trajectories and real recurrent weight gradients. In CI, independently reload the source-matched checkpoint, score it on held-out TEST, compare to stored scores, and check actual trained trajectories and prior-history interventions.

Independent ablations include current-only MLP, GRU with only the most recent 1, 4 or 8 observations, and a forced wrong first-fork action. Cue-reversal at the first fork is also measured with the rest of the seed/layout held fixed. Report raw illegal proposals **before** host legal-action masking, clean vs merely eventual completion, collisions, painted fork count, inference milliseconds, RSS, source SHA, checkpoint SHA-256, loss curves and an original decision-trace SVG gallery. The cue/recovery environment is deliberately constrained and synthetic; do **not** infer general environmental understanding or real CyberSand performance.

## Commands

```bash
python -m unittest -v experiments.driv3r.model_zoo.material_forks_mz009.test_world
python -m experiments.driv3r.model_zoo.material_forks_mz009.run \
  --out mz009-evidence --train-seeds 160 --eval-seeds 24 \
  --rollout-seeds 8 --updates 800 --batch 48
```

Large checkpoints and raw metrics stay in immutable GitHub Actions artifacts, never Git history. Do not merge, mark Ready, register in the game, use paid API/compute or edit outside this research folder and its dedicated workflow. Next milestone: scale this controlled environment to MZ-007b's ~19M GRU and introduce longer partially observed spatial navigation and painting, with stronger held-out world variation.