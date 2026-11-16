# DG014 — trajectory-level neural donor inheritance (research-only)

**Status: Draft/HOLD.** Stacked on DG013 (#246); not a native CyberSand integration. No game code/assets/configurations are changed.

## Question

DG013 demonstrated a severe negative result: donor-conditioned single-step imitators did not reach most goals within 72 actions, and their authored one-step teacher was similarly poor. Can **newly trained neural specialist donors**, a **trajectory-trained receiver**, and an explicit but limited map-distance observation produce effective sustained game behaviour? Does swapping an inherited law change the receiver's actual goal, rather than simply its logged action label?

## Experiment

A new, independent external 9×9 world harness reproduces CyberSand-like *categories* (not exact native physics): obstacles, wet terrain, protected cells, painting, retrace legality, visitation, and primary/alternate spatial goals.

Three authored source-law families: (1) wet-motion traversal cost; (2) choice between two spatial goals; (3) painting at the selected goal versus painting at the start before navigation. Each law is learned separately by neural specialist donors. Each CPU seed trains **12 independently initialized neural donors** (six narrow, six unseen-architecture gated) and **two policies** (donor-conditioned and equal-task law-blind). Donors have privately permuted and reflected input axes. The receiver sees only 8 canonical probes' actual donor outputs per specialist, not donor weights, labels, input permutations or private reflections.

The receiver is supervised on **multi-step Dijkstra-oracle traces**, across 600 source worlds and seven out of eight possible law triples; the **(+,+,+) triple is excluded from all receiver training**. Paired laws are demonstrated on the same training worlds to reduce source-world/law confounding. Gated donor architecture and 100 sealed world seeds are entirely absent from receiver training. A seed-set intersection assertion enforces no source/sealed world duplication.

Seal: 100 distinct playable worlds per seed, 50 withheld (+,+,+), 50 mixed. At most **64 actions per world**; terminate at the first successful goal/paint completion. Five matched arms: inherited correct gated donors, law-blind receiver, wrong-goal donor, wrong-wet donor, and independent Dijkstra oracle. Report success, held-out success, path length, unique cells, repeats, paint count, wet traversals, true-law weighted path cost, and legality.

**Research limits:** Dijkstra is a scripted teacher and the observation **includes precomputed obstacle-aware shortest-path distances to *both* candidate goals**. This is not end-to-end planning discovery. Donor functions are supervised authored preferences; the probe signatures are output-mediated, not transplanted KV caches or donor weights. The path cost is not reward-optimized directly by the student. The RETRACE action is legal but is not a separate trained recovery curriculum. No unknown third-order physics is discovered.

## Independently completed local CPU experiment (pre-CI)

Source SHA-256: `23918487bfc03acf5bb683857bcbed0b6e279d82fb26727720f3f37ae4495974`. Local torch 2.10.0+cpu. Each seed trained 12 neural donors at 140 optimiser steps and two receiving policies at 750 steps; original weights, private axis maps, per-file SHA, JSON traces and SVG trajectory gallery are preserved.

| 100-world sealed goal-and-paint completion | Seed 26011624 | Seed 26011625 |
|---|---:|---:|
| Correct inherited neural donors | **99/100** | **99/100** |
| Correct inherited, withheld +++ only | 49/50 | 49/50 |
| Law-blind policy | 38/100 | 41/100 |
| Wrong spatial-goal donor | 17/100 | 13/100 |
| Wrong wet-cost donor | 100/100 | 99/100 |
| Scripted Dijkstra oracle | 100/100 | 100/100 |

All policy arms emitted **zero illegal actions**. The wrong-goal donor control strongly changes trajectory outcomes on the same worlds. However, **wrong-wet donors do not reduce success**: the agent reaches goals despite the material-cost swap. The inherited policy's true-law mean weighted traversal cost (19.87 / 21.47) remains materially worse than the Dijkstra oracle (11.15 / 12.21), so a high completion rate is not proof of optimal motion-rule inheritance.

This is a significant improvement on DG013's low goal completion in a **changed, easier, more structured task**, not a controlled improvement on an identical environment. The teacher, goal specification, global obstacle-distance features, training distribution and world size all changed. Do **not** compare 99% directly against DG013 7/36 as a model-only advance.

## Reproduction

```bash
python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python3 -m pip install numpy==1.26.4
python3 -m experiments.memplant.trajectory_transfer_014.experiment --selftest
python3 -m experiments.memplant.trajectory_transfer_014.experiment --seed 26011624 --train-worlds 600 --steps 750 --test-worlds 100 --horizon 64 --donor-steps 140 --out /tmp/dg014-24
```

The exact-head GitHub Actions matrix repeats both seeds on pinned CPU dependencies and retains 14 trained model checkpoints per seed, calibration maps, metrics, original SVG paths, SHA manifest and source hash in immutable Actions artifacts. **CI results must not be claimed before the actual run succeeds.**

## Next falsification

Remove both globally precomputed shortest-path potentials and require local/map-image path integration; introduce obstacles that change after painting and require RETRACE; train recovery policies on interventions rather than oracle-only expert rollouts. Evaluate action changes and true-law *path cost* for material swaps, not just completion. Maintain original source-legal action masks, same-world matched counterfactuals, per-family holdouts and negative-result disclosure.

**No paid external model/GPU/API, no CyberSand edits, no merge/Ready.**
