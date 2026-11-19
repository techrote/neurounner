# DG020 H44 — hard output-head feasibility FAIL (19 November 2026)

**Classification: EXPOSED-ONLY TRAINING/MECHANISM FAIL.** This is **not** a new prospective cohort. H40's restricted preregistered prospective **30/30 PASS** remains unchanged; H41's non-discriminative placement-family result, H42's synthetic donor-channel interpolation FAIL and H43a/H43b's endpoint regressions remain negative evidence.

**Preregistration:** [PR #253 comment 7003228250](https://github.com/techrote/neurounner/pull/253#issuecomment-7003228250), posted **before execution**. No constraint was weakened after observing the result. All work remains on the same Draft/HOLD research lane; no merge or Ready.

## Intervention and causal controls

H44 asked if changing **only the eight output-layer weights** of the original genuinely trained 209-parameter receivers, with their hidden layers and output bias frozen, can preserve **every actual trained-donor law endpoint** while correcting **every action** on the 65-point synthetic interpolation grid. All three H40 receiver checkpoints, eight genuine trained donor checkpoints, original 4,800 exposed H40 worlds, original laws 0.35/2.30, world geometry and legal masks were preserved. Original 4,800-row observation/decision SHA-256: `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`; original checkpoint identities checked during CPU execution.

Within each paired world, observation, goals, topology, legal mask and receiver weights are identical; only the donor-derived wet-material cost is changed. No family ID, law label, donor ID, oracle path cost or oracle action is supplied to inference. Labels are supervision/LP constraints *only*. Synthetic interpolated features between **genuine trained donor endpoints** are **not independently trained intermediate-law donors**.

The frozen network's route-score difference is linear in its 8 output-head weights. H44 extracted 416,000 signed feature constraints per receiver (4 donor profiles × 65 alpha probes × 1,600 exposed worlds) from the **actual trained network's two ReLU layers**. A HiGHS cutting-plane linear program minimized L∞ weight displacement subject to a hard, correctly signed score margin ≥1e-5 on every example; output weights had **no movement budget cap**. The full constraint matrix was rechecked after every active-set pass.

## Results

| Receiver | Original incorrect grid decisions /416,000 | Full hard-constraint LP | Infeasibility witness |
|---|---:|---|---|
| 26011639 | 19,200 | **INFEASIBLE** | Eight signed hidden vectors; **exact positive rational null certificate** |
| 26011640 | 20,075 | **INFEASIBLE** | Seven signed hidden vectors; numerical convex residual ≤4.96e-10; **not** claimed as exact certificate |
| 26011641 | 19,892 | **INFEASIBLE** | Six signed hidden vectors; **exact positive rational null certificate** |
| **Total** | **59,167/1,248,000** | **H44 FAIL** | Independent witness replay PASS |

**No new candidate receiver checkpoint was accepted or produced.** The original genuine-law endpoints remain **0/38,400 incorrect** in frozen baseline replay. The independent auditor checked all original SHA-256s, reconstructed every selected witness vector directly from original PyTorch receiver/donor checkpoint inference, verified bitwise agreement with the LP support vectors, reproduced the 59,167 original grid errors, and used exact rational arithmetic to verify two strictly positive dependencies. Those two certificates show that **no final-layer output weights alone** can strictly separate the selected constraints while hidden activations remain unchanged, even with unlimited weight movement. This is not an impossibility theorem for retraining the hidden layers or using different genuine donors.

### Post-hoc subsystem attribution (not a new acceptance rule)

| Constraint subset | 26011639 | 26011640 | 26011641 |
|---|---|---|---|
| Invariant worlds, all 65 alphas | Feasible | Feasible | Feasible |
| Divergent worlds, all 65 alphas | **Infeasible** | **Infeasible** | **Infeasible** |
| All invariant probes + divergent genuine endpoints | Feasible | Feasible | **Infeasible** |
| All divergent probes + invariant genuine endpoints | **Infeasible** | **Infeasible** | **Infeasible** |

This locates a representation/threshold problem, **not** merely a weak optimizer or a justification for relaxing invariant-action constraints. With fixed hidden activations, even the divergent-only full synthetic-grid requirement cannot be met.

### Calibration warning

Genuine donor estimate errors at the original trained endpoints were approximately -0.00132 to +0.00160. A separate **proxy diagnostic only** (not a neural acceptance result) applied the exact additive physical cost rule to each *donor-estimated* interpolated cost rather than the nominal true-alpha law. It disagreed with the nominal oracle on **500/1,248,000** probes (157/168/175 by receiver), **all in wet-poor/wet-rich nominal-law-invariant buckets**. This highlights near-boundary mismatch between imperfect donor estimates and ideal-law supervision; it does **not** authorize labeling the proxy a success, excluding those cases post hoc, supplying true law/answers at inference or weakening the hard gate.

## Evidence and next mechanism

**Owner evidence archive:** `DG020-H44-exposed-hard-head-LP-INFEASIBLE-20261119.zip`; ZIP SHA-256 `9966ce731240944ebede302d79d9922304400579770335556d0192ab1b0330b5`. Verified **41 payload files + SHA-256 manifest**; contains all executable source, full frozen-hidden feature arrays (3 NPZ files), 19 selected signed witness examples and individual observation hashes, raw original H40 receipts and all 11 neural checkpoints, H43 failed candidate checkpoints and raw H43 decisions, source/solver logs, independent neural/witness auditor and detailed post-hoc subsystem diagnostics. The binary archive is preserved in the owner handoff, **not yet durable GitHub binary custody**. Source and analysis custody are separate from the original model binaries.

**H44 gate: FAIL (training-only), no prospective PASS claimed.** Next *smallest* mechanism: a targeted **hidden activation-knee adjustment** in the original 209-parameter receiver, constrained from the outset by **hard genuine-donor endpoint no-regression and invariant-world no-transition requirements**, and checked against these infeasibility witnesses. Exposed-world success alone cannot pass the prospective scientific gate. Retain H40 restricted 30/30 PASS with its methodological limitations; do not enlarge models/families, invoke paid GPU/API or modify CyberSand/DRIV3R.
