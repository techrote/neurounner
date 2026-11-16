# TRAP-PREVENT-001 — bounded local escape-sensor training (16 Nov 2026)

**DRAFT/HOLD; DEVELOPMENT-only synthetic negative result.** No claims of native gameplay success, real SLM adaptation, artistic quality, held-out generalization, paid compute or any production change. This is a *new model-side experiment*, not an A* routing supervisor.

## Hypothesis and design

Previous STOP-margin original-receipt audit found every terminal STOP was host-forced. The next test asks whether a *learned* controller can benefit from 1/2-turn local exit information **before** geometric trapping. Two 33-slot linear-softmax action rankers share exact initialization, 36→72→128 TRAIN worlds, imperfect locally risk-aware teacher labels, off-policy bad-but-legal interventions (22%), minibatch order and optimizer schedule (70 genuine Adam updates/stage = 210/arm/seed). The baseline's final nine feature slots are masked; the experimental ranker receives local successor degree, two-turn distinct successors, dead-end/one-exit flags, corridor and adjacent-material/ink signals, recovery budget, and relative frontier. These are candidate-wise bounded **local** observations, not destination/path or global reachability oracles. Host menu and state transitions are identical.

Training: 4 structural TRAIN families only, 3 seeds/family and six source-faithful trap-corridor variants; deterministic split IDs. DEVELOPMENT: 4 structural families, 2 seeds/family plus four corridor variants, 12 scenarios × intact/forced × 4 horizons (36,72,128,160) × 2 arms = 192 original episodes per seed. Forced legal deviations at decisions 12, 35 and 71 where alternative movement exists. Both arms share the forced-deviation schedule, but each receives a different legal action only if its own selected action differs; hence this is a **paired scenario** comparison, not identical intervention-action replay. No native 64-move ceiling is changed. Sealed structurally distinct held-out worlds are never instantiated.

## Original local CPU results

| Seed | 128 forced arm | Mean unique ink / 128 | Early STOP / 12 | Mean branches | Avoidable low-exit choices |
|---|---|---:|---:|---:|---:|
| 586 | Baseline | 128.000 | 0 | 9.833 | 2 |
| 586 | Local sensor | **113.083** | **5** | 23.250 | 2 |
| 587 | Baseline | 127.667 | 0 | 7.667 | 3 |
| 587 | Local sensor | 127.500 | 0 | 7.167 | 0 |

| Seed | 160 forced arm | Mean unique ink / 160 | Early STOP / 12 | Mean branches |
|---|---|---:|---:|---:|
| 586 | Baseline | 159.917 | 0 | 17.000 |
| 586 | Local sensor | **131.750** | **5** | 23.250 |
| 587 | Baseline | 159.667 | 0 | 8.167 |
| 587 | Local sensor | 156.917 | **1** | 7.333 |

Each arm at each seed produced full 128 and 160 decision lengths in all **intact** 12/12 DEVELOPMENT worlds, usually tracing banal long L-shaped paths. In forced 128–160 trials, the local-risk representation reduces counted avoidable low-exit actions for seed 587 but *fails to prevent* occasional geometric trap/recovery exhaustion and can induce more branches and shorter survival. This is a **negative / mixed** result; lower one-step risk does not imply better long-horizon control or useful line character. The seed-586 failures are especially material and must not be omitted. Branch counts are neither reward nor native art acceptance.

Each seed yielded **two genuine 33-parameter checkpoints**, 420 matched total Adam updates, plus complete original per-decision action menus, probabilities, commanded and executed actions, intervention flags, occupancy-risk features, material hashes, synthetic PNG trajectories and SHA-256 manifests. Seed 586: 192 episodes, 18,620 decisions, 29.486 seconds, peak RSS 361,756 KiB. Seed 587: 192 episodes, 18,984 decisions, 30.824 seconds, peak RSS 364,372 KiB. Aggregate 384 original episodes, 37,604 source-replayed decisions, 528 predeclared legal forced-move interventions. CPU times/RSS are *local processes*, not model serving or real GPU costs.

**Local checkpoint SHA-256** (the binaries belong in artifacts, never Git history):

- Seed 586 baseline `c28a9726879eacf92278c859869a9aa1486e8819a8597a99e8317f714a80d64c`; sensor `9ca7f4f6b5d6bc3889097eb088cc4ebc7004d1799cf5c8323d8483c416902974`.
- Seed 587 baseline `b423a4904e823ffb382fc8cf7849a4a3fa305a1c74b6a80fed0c6ef7fbeb1664`; sensor `59940c33e50dd7791595bf96560f1c8e7ed3e206af9ef757418e5ad08f98f002`.

## Qualifications and preserved failure

A first-pass `test_negative_candidate_guard` failed because the new sensor silently accepted an invalid action string as a non-movement action. Repaired by checking `w.menu(MAX_RECOVERIES)` before all sensor calculations, reran the test and the campaigns with the final source. The old 7-test subset and current full *available local* 32-test historical source inventory passed. Those 32 tests are **not** a claim about current GitHub HEAD's complete (58+) suite.

Independent exact-material replay recovered **384/384 original episodes and 37,604/37,604 sequential decisions** from restored checkpoint choices, exact probability vectors, forced-deviation selection, host legality, synthetic material edits and risk metrics. Verified **774/774 original experiment artifact files** against source-generated manifest SHA-256, plus four checkpoint hashes; deliberate corruption is rejected by manifest and state/logit checks. All terminal STOPs remain literal host-legal actions. No fake native receipts.

The original gallery depicts visibly different yet often less reliable paths, especially the seed-586 sensor under forced deviation; don't treat greater branching as improved character.

## Next test

The next bounded ablation should differentiate **recoverable detours** from **unrecoverable option collapse** and measure motif identity conditional on survival. Local exit count alone is insufficient; more training or stronger steering weights should not be promoted as a fix without matched blind-seed counterexamples. Owner-gated native work remains separate.
