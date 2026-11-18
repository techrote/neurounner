# DG020 H25 — post-hoc near-margin falsification (2026-11-18)

**Disposition: original H25 prospective 30/30-cell PASS unchanged; new, out-of-support near-margin selectivity diagnostic FAIL.** This test operates on previously exposed H25 observations and cannot count as independent prospective validation. No new receiver was trained or selected.

## Frozen provenance and controlled interventions

- Original full H25 evidence ZIP: `79602f8e86acb91fdd551517997c63b5ba9f021d9a95141ba5df97a220c51d3a` (SHA-256).
- H25 checkpoint SHA-256: seed 26011639 `49a74ad64fbfd79a4f5771bfa71158186b5025da038e621c87edc0331c234df3`; seed 26011640 `21df8ec5e5a1a9d16aa6fefd42054eb987a3092719521b7335e32699e7d789c3`; seed 26011641 `093750d8e0f9b35affe5c831b0c01024a002638e37672fd367db57b3d6093018`.
- Frozen trained H21R donor checkpoints and H25 receivers (209 parameters) loaded on local CPU. The two material laws remain wet costs 0.35 and 2.30. In each matched pair, geometry, fees, goals, legal actions and receiver weights remain identical; the donor-derived law is the **only** arm-specific intervention.
- Diagnostic source: `h25_posthoc_invariance_stress.py`, SHA-256 `0959774818f7b4e5aa0e28be1fb8c89f92e44a68d1a3e44cf47207330823e23d`. Component ablation `h25_posthoc_oracle_donor_ablation.py`, SHA-256 `d826385c7bf25e51b97f8dd9764e59ad1e62871ca50b4de5df39e181e29b8fb2`.
- Four deterministic fee permutations within each route preserve route totals; four wet-position permutations preserve each route's wet count. Both retain the exact additive oracle outcome.
- Near-margin intervention: for each invariant world, find the law with smallest absolute oracle route margin `m`, target `sign(m)*epsilon`, and adjust every one of 12 route-0 fees by `(target-m)/24` while subtracting that amount from each route-1 fee. Both law arms receive the **same** fee adjustments; the oracle action under both laws is unchanged; all resulting fees remain in [0, 0.30]. Test targets epsilon=0.01, 0.001 and 0.0001. These targets are *outside* the H25 validation generator's roughly 0.06–0.19 minimum-margin support.

## Results: genuine donors and original frozen receivers

| Perturbation | Matched pairs | Both-law correct | Inappropriate invariant law flips | Baseline action changes |
|---|---:|---:|---:|---:|
| Fee permutations, four copies | 19,200 | 19,200 | 0 | 0 |
| Wet-placement permutations, four copies | 19,200 | 19,200 | 0 | 0 |
| Invariant margin 0.01 | 2,880 | **2,730** | **150** | 150 |
| Invariant margin 0.001 | 2,880 | **2,716** | **164** | 164 |
| Invariant margin 0.0001 | 2,880 | **1,644** | **1,236** | 1,236 |

The 150 failures at 0.01 all occur in seed 26011641, `wet_rich_invariant` (150/320 seed/category). At 0.001, 160 fail in that category and four in `equal_wet_invariant`, again only seed 26011641. At 0.0001, failures affect all three receiver seeds: `wet_rich_invariant` 800, `wet_poor_invariant` 320, `equal_wet_invariant` 116. **All analytical oracle choices remain invariant** under each perturbation.

## Frozen receiver with exact analytic donor-cost substitution (diagnostic only)

| Minimum oracle margin | Genuine-donor wrong pairs | Perfect-feature wrong pairs | Disagreement between feature sources |
|---|---:|---:|---:|
| 0.01 | 150/2,880 | **144/2,880** | 6 |
| 0.001 | 164/2,880 | **172/2,880** | 8 |
| 0.0001 | 1,236/2,880 | **280/2,880** | 972 |

The 0.01 failures predominantly survive perfect scalar wet-cost input: **receiver aggregation/absolute-margin calibration** remains defective near boundaries. At 0.0001, trained donor forecast error additionally matters. Perfect-value substitution is never presented as genuine inheritance success.

## Reproducibility and scientific boundary

The owner handoff `DG020-H25-posthoc-invariance-stress-20261118.zip` contains both scripts, full 47,040 raw per-intervention decision/logit/oracle receipts, complete per-cell summaries/failure indices, the ideal-feature component ablation, logs and SHA-256 manifest. Its ZIP SHA-256 is `2dba4656a2a3dbc4484686461f42b1c8bf7169aff759cc1dae40a15aab8e25f8`. Raw decisions SHA-256: `72f96e5946cc414da9c1bca020fc0b8af6a0e6e4b544d7154c34b543830c8f38`. The original H25 full evidence and this new ZIP are **not yet in durable GitHub custody**.

**Verdicts:** H25 originally registered synthetic gate PASS. Post-hoc edge-case selectivity FAIL. Broader physically meaningful game-state inheritance NOT ESTABLISHED. These are different claims, not a retrospective rewrite of H25's registered thresholds. All these new transformed observations are exposed; exclude them from future prospective acceptance.

**Smallest justified next mechanism change:** same-size, training-only, near-tie *per-law absolute-margin calibration* on matched invariant and divergent pairs, with explicit donor-error sensitivity; do not repeat H23's delta-only loss regression. Preregister genuinely unexposed near-margin worlds and meaningful controls before training or evaluating another receiver. Keep H22/H23/H24 negatives and all original checkpoint/receipt bytes immutable. H25-specific pinned GitHub Actions checkpoint CI and external methods review remain outstanding.

Ownership: only `techrote/neurounner` existing PR #253; Draft/HOLD, never Ready/merge. No new PRs/branches/issues, paid compute or CyberSand/DRIV3R writes.
