# DG020 H57 — retrospective CPU action and reduction-order robustness (19 November 2026)

**Disposition: 16/16 local numerical replay modes PASS; distinct-host cross-CPU action-level qualification NOT EXECUTED / NOT PASSED; no new scientific gate.** The original H56 preregistered restricted 45/45 prospective PASS stands unchanged. Draft/HOLD, no merge or Ready.

## Exact source / authentic neural custody

H57 reuses the sealed H56 2,400 worlds (7,200 world/receiver rows and 57,600 genuine neural endpoint decisions), original eight independently trained H34 donor checkpoints and all three original 209-parameter H45/H52 receiver checkpoints. Neither receiver nor donor is trained, selected or modified. All registered world hashes are regenerated from the locked H56 source, donor models are independently re-probed under the shared cheap-law dry-anchor convention, and original checkpoint file hashes and JSONL receipt SHA are verified before inference.

- H56 registered evaluator SHA-256: `f8c059eec92215af7cbbe4752a15733667ca75dbadf2147b6ed6a8810572b871`.
- H55 frozen dependency SHA-256: `2909ee6ca0e604f866721a6081dec55fd0007f0e8c82a4cd65e2d2c57cbe2815`.
- Original H56 raw neural receipts SHA-256: `6d3b55b4273c1a278a528f6d2044c303bf30cb8841bfe4bf5619c0f2607f6a86`.
- H56 world-fingerprint SHA-256: `187af8cbea7ddca426075d5c9155211adf4c15d1bc989ea6b22d39d3d0ddd5f8`.
- H57 exact executed source SHA-256: `f873cb3220446b6abedbb1e7e482e7476d84c616b594b6643dd79019a562e62c`.
- H57 complete result JSON SHA-256: `0652ccf477a91fba9224636728b4edea90ba837f6feebae64fd44c1299461f7e`.
- Local environment: Linux x86_64, Python 3.13.5, CPU PyTorch 2.10.0, NumPy 2.3.5. **ONE CPU host**, not 16 hardware platforms.

## Executed numerical tests (all 57,600 frozen genuine endpoint actions per mode)

| Float32 / float64 mode | Tested modes | Actions changing | Incorrect actions | Max abs logit-margin difference |
|---|---:|---:|---:|---:|
| PyTorch float32, 1/2/4/8 inference threads | 4 | 0 each | 0 each | 0; bit-identical |
| Full PyTorch float64 forward | 1 | 0 | 0 | 0.0000397398341 |
| Float32 tile-sum permutations | 5 | 0 each | 0 each | up to 0.0000381469727 |
| Float32 explicitly serial tile-sum orders | 6 | 0 each | 0 each | up to 0.0000381469727 |
| **Total** | **16** | **0/921,600** | **0/921,600** | — |

This is a stronger same-CPU arithmetic perturbation test than the earlier H56 single float64 replay, but it **does not** qualify other CPU instruction sets, kernels or environments.

## Concentrated near-tie weakness (not observed incorrect actions)

| Correct signed neural margin ≤ | H56 actions inside window |
|---|---:|
| 1e−5 | 9 |
| 2.5e−5 | 173 |
| 3.9739834e−5 (observed H56 float64 drift) | 698 |
| 7.6293945e−5 (historically observed H48 cross-CPU margin drift) | **2,521** |
| 1e−4 | 3,274 |

The entire set of 2,521 actions within the H48 observed drift window belongs to receiver **26011641** (933 divergent and **1,588 invariant**); seeds 26011639 and the H52-repaired 26011640 have none. Minimum correct signed margins per receiver: 26011639 **+0.0001373291**, 26011640 **+0.0002746582**, 26011641 **+0.0000038147**. Lowest-margin case: 26011641 / modular_spokes / cheap_divergent. **A margin inside an empirical drift window is not an actual changed action or a certified failure.** The H48 observed maximum drift is *not* a guaranteed numerical-error bound. No tested H57 mode changed an action.

## Scientific interpretation / next smallest mechanism step

**H57 local numerical qualification PASS; actual heterogeneous-host qualification remains outstanding; prospective causal-selectivity gate NOT RETESTED.** H56's original restricted additive 2-law causal-selectivity PASS remains unchanged. H53b NOT EXECUTED; H54 INVALID; H46 synthetic-law non-identifiability and H51 earlier continuous-invariance negatives remain preserved.

Next: complete binary GitHub custody of the genuine checkpoints and full H56 raw receipts; run an exact-HEAD pinned action-level CPU replay on **different** CI hosts, failing closed on any action or oracle mismatch. Prioritize receiver 26011641. If actual platform-dependent action flips appear, perform a **training/exposed-only** signed-margin mechanism repair with invariant constraints, not fitting to H56 revealed answers. Do not broaden receiver, laws, maze families or game claims, and do not modify CyberSand/DRIV3R.

**Owner evidence**: the H57 archive contains the exact executed script, raw H56 original evidence, machine-readable H57 results, SHA-256 manifest and all original negative records. The scripted H57 source and JSON outputs are also retained as gzip/base64 textual custody alongside this report. Reconstruct those text files by base64-decoding then gunzipping; verify the exact SHA-256s above before replay. Git textual custody does not substitute for still-outstanding full binary H56 custody.
