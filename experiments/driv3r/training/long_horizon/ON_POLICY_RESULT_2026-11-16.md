# D-RIV3R Long-Horizon — Paired on-policy teacher correction (2026-11-16)

**Status:** EXECUTED OFFLINE / NEGATIVE DEVELOPMENT RESULT / Draft/HOLD. **This is not native paint, not held-out qualification and not d1/LFM/Sprocket weight training.** Owner accepted D-COMMON M1 `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6` with schema `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`; none of that contract or the production game was changed.

## Experimental question and matched control

Does an extra **100 Adam updates on on-policy states** with imperfect local-teacher correction yield better continuation and recovery than **100 more updates on the original offline teacher corpus**? This is a bounded DAgger-like test, not reinforcement learning. The two arms share a genuinely trained 36 → 72 → 128 curriculum precursor (300 updates), identical live weights **and Adam moments** at the fork, the same total **400 updates** and seed-specific optimizer schedule. Both are 625-parameter NumPy MLP rankers.

The on-policy continuation collection uses only TRAIN 1000-series worlds, no hindsight or reward oracle, with predetermined local legal mistakes at turns 12, 35, 71 where alternatives exist. Evaluation uses matched DEVELOPMENT 3000-series worlds and both intact and forced trajectories at 36, 72 and 128 decisions. Structural HELDOUT 9000-series families remained sealed. A 128-decision synthetic episode is **not** authorised native execution; 64 logical native moves remains the ceiling.

## Original 128-decision DEVELOPMENT evidence

Eight matched worlds per seed/condition (including recovery traps); four checkpoint files total. Here **Offline** means additional conventional teacher imitation; **Correction** means additional on-policy-state teacher imitation. Cells are unique synthetic painted cells, not native receipts or artistic scores.

| Seed | Perturbation | Arm | Mean ink | Mean branch cells | Mean turns | Mean backtracks | Early STOP |
|---|---|---|---:|---:|---:|---:|---:|
| 573 | Intact | Offline | 126.125 | 25.750 | 37.000 | 1.875 | 0/8 |
| 573 | Intact | Correction | 125.375 | 23.750 | 35.625 | 2.625 | 0/8 |
| 573 | Forced | Offline | 124.750 | 24.125 | 38.125 | 3.250 | 0/8 |
| 573 | Forced | Correction | 124.750 | 21.125 | 37.250 | 3.250 | 0/8 |
| 574 | Intact | Offline | 125.750 | 23.375 | 36.125 | 2.250 | 0/8 |
| 574 | Intact | Correction | 126.125 | 21.875 | 36.875 | 1.875 | 0/8 |
| 574 | Forced | Offline | 126.125 | 23.125 | 38.125 | 1.875 | 0/8 |
| 574 | Forced | Correction | 125.000 | 25.000 | 37.750 | 3.000 | 0/8 |

**Result:** no robust positive effect. Correction sometimes increases branches (forced seed 574) but reduces them under all other 128 settings. It does not improve forced-run ink in either seed, and no STOP/failure difference is observed. This is a meaningful **negative finding**, not proof on-policy aggregation is useless for richer state/action representations. Branch count or high ink yield alone does not demonstrate an intentional, steerable worm.

At 36 and 72 decisions there were no early STOPs or meaningful ink differences between arms. Detailed counts and action distributions, all negative cases, exact decision-probability vectors and full event transcripts are preserved in each immutable run directory, not cherry-picked into this report.

## Reproducibility, runtime and exact checkpoint IDs

Both original runs used Python/NumPy, `--train-worlds-per-family 4 --dev-worlds-per-family 1 --updates-per-stage 100`. Source pins are stored in each `report.json`. Four actual `.npz` weight checkpoints were restored and independently used in original rollout verification:

| Seed | Offline checkpoint SHA-256 | Correction checkpoint SHA-256 | Runtime including full evaluation | Peak RSS (KiB) |
|---|---|---|---:|---:|
| 573 | `52ffc4a54e3d7f4605703b58870e278f6533fa4698a2a775b28bea6e3d162027` | `1e4670e90228ef64fe5e9869f130b134b6376e39a3773aa8454bd5b5692756b1` | 17.501 s | 115,220 |
| 574 | `7b2a26753aca76c70144024cd72ca4d8f20cdb7b8d849eb0b714b299c4c4240f` | `17bc42124294f7e5b3ec9c6860be4bbee343eeda2265487aa81fd5a4c346f5ae` | 18.014 s | 115,328 |

128-decision mean per-policy choice latency ranged from approximately **0.089–0.103 ms**. All 192 original DEVELOPMENT trajectories were independently replayed from logged legal candidate menus, original material edits, recorded neural probabilities and exact deterministic World state hashes. Both runs passed manifest verification (195 original files per seed, four checkpoints total). The first local 30-test suite had one failed mutation test because it assigned a no-op STOP to an already-STOP event; the test was corrected to mutate to an invalid action. The **next complete 30-test local suite passed**. This failure is retained in the local evidence, not silently hidden.

**Local provenance:** source SHA-256 for `on_policy_adaptation.py` is `eba517fa1c2b3d70485366ce803dbbc629f53f9089f002c69cd3f710a2c11fc4`; verifier `f2ebc15a210313c10ee794c312fb982defdc4874aaa224726a83ab9bf7b6d804`; test `df69e5808aec5e23dbc459c9b1c1e5a824f77072a1e6503d44c3dd4503611f1d`. GitHub blob identities were checked against original tested files. The local source suite was executed before the branch's independent intent-lab additions; **a passing local subset is not a full exact-HEAD branch qualification**. The updated dedicated workflow now runs full discovered lane tests and the two independently verified paired experiments, uploading checkpoints and transcripts as Actions artifacts; an Actions success must be separately observed before qualification is claimed.

## Boundaries and next research inference

No held-out model outcome, actual large-SLM weights, native receipts, paid API, external GPU, game files, production provider defaults or merges. All synthetic 128-decision evaluations exceeded the historical 64-logical-move native limit; those observations are not a request to raise that limit. The more promising follow-up is **explicit style/history conditioning and geometric-target control**, not assuming that another round of one-step teacher imitation yields purposeful line character. Keep this negative result, source and all trajectories in the experiment ledger.
