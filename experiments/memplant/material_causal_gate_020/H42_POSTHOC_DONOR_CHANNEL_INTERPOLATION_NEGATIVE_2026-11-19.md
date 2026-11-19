# DG020 H42 — post-hoc donor-channel interpolation reveals hidden invariant failures

**Research disposition:** H42 continuous-donor-feature selectivity **FAIL** as an exploratory *post-hoc mechanism test*. **H40 retains its preregistered restricted prospective 30/30 PASS**, without reinterpretation. There is no new prospective cohort, intermediate-law-trained donor, or independent unseen-world-family validation.

## Frozen provenance and scope

H42 re-evaluates the original 4,800 H40 world pairs, frozen 209-parameter receivers (seeds `26011639`, `26011640`, `26011641`), and four pairs of **original genuinely trained donor neural networks**. The H40 raw receipt SHA-256 must be `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`, and original donor and receiver checkpoint hashes are checked before evaluation. No new weights, world families, law families, or data splits.

For each unchanged world and genuine donor profile, probe the frozen receiver at 65 evenly spaced interpolations from the trained cheap-law donor's predicted wet cost to the trained expensive-law donor's predicted wet cost. The receiver receives the interpolated feature and unmodified observed fees/wetness; it never receives the law identifier, world family, oracle decision, or true route cost. For audit *only*, the analytic routing oracle is evaluated under the linearly interpolated physical wet cost.

**Critical limitation:** These 63 interior feature values are *synthetic interpolations*, NOT the outputs of independently trained intermediate-law donors. They cannot be counted as a genuine-neural scientific gate test. The intervention is a stress test of the existing receiver's conditional decision surface between the two already-trained law endpoints.

## Results: frozen genuine endpoints still correct; interior response unstable

| Check | H42 result |
| --- | ---: |
| Genuine trained donor endpoint actions recovered | **38,400/38,400 correct** |
| Action disagreements against archived H40 decisions | **0** |
| Max absolute CPU logit drift from original H40 receipt | **9.5367 × 10⁻⁶** |
| Interpolated feature-grid decisions versus linear physical rule | **1,188,833 / 1,248,000 correct** |
| Wrong intermediate-grid actions | **59,167** |
| Distinct invariant worlds with an incorrect interior action | **118/2,880** |
| Invariant-world donor-profile action trajectories with unintended action change | **470** |
| Equal-wet donor-profile trajectories exhibiting unintended action change | **178** |
| Donor-profile trajectories with more than one action threshold | **292** |
| Wrong-action score margins ≥ 0.001 | **52,231/59,167** |
| Maximum wrong-action route-score margin | **0.00762844** |

Error breakdown (world pairs x four genuine donor profiles x 65 feature positions):

| Bucket | Original worlds | Wrong interpolated actions | Worlds affected |
| --- | ---: | ---: | ---: |
| Cheap-edge divergent | 960 | 19,200 | 960 |
| Expensive-edge divergent | 960 | 38,360 | 960 |
| Wet-rich invariant | 960 | 0 | 0 |
| Wet-poor invariant | 960 | 732 | 73 |
| Equal-wet invariant | 960 | 875 | 45 |

The worst errors are substantive, not float tie noise: 52,231 mistakes have a wrong-direction neural score margin at least 0.001, about two orders of magnitude larger than CPU endpoint replay drift.

**Concrete invariant counterexample:** receiver `26011640`, family `interleaved_braid`, `equal_wet_invariant` world index 5, observation SHA `a7e2d53bcb843f740f88744866d4518c67d70fae12b000b647b0d8388b215f3d`. Both routes have exactly two wet tiles. The oracle always chooses route 0, with fee-sum difference about −0.0029324 independent of wet material cost. The frozen receiver chooses the correct route at both genuine donor endpoints but changes incorrectly to route 1 at interpolated alpha around 0.55 (score margin about −0.000435), then back to route 0 by alpha 0.60. All four trained donor profiles exhibit two transitions on this world. This is not a tile-order effect or a genuine-donor endpoint failure; it exposes non-monotonic donor-feature conditioning.

## Negative-control correction

H40's so-called `zeros` control used **`1 + fee`**, not a literal zero tensor. H42 separately tests the *actual* all-zero donor feature and the dry/neutral `1 + fee` feature. On all 2,880 original invariant pairs each happens to return the correct action, but that agreement is **not independently discriminative** on these worlds. On divergent pairs each law-constant sham is necessarily wrong on one arm, so a zero both-correct count must not be treated as strong causal evidence.

The separate diagnostic using exact analytic material cost injected as a scalar scores 4,800/4,800 original endpoints. This control **explicitly leaks the physical law** to the feature and is reported only as an *oracle-fed proxy*, excluded from scientific PASS claims. It illustrates that H40's engineered donor probe reduces inheritance to a scalar wet-cost channel.

## Verification and scientific boundaries

The original receiver/donor checkpoint hashes and original H40 neural decision receipts were checked. Original action agreement is exact (0 disagreements); numeric logits on this CPU/PyTorch runtime differ by at most 9.54e-6, so **do not assert cross-platform zero logit drift**. The 4,800-row audit receipts and summary, including every world observation SHA, are preserved in the H42 evidence archive. Independent compact replay reproduced **59,167** errors and the **178/292/470** threshold counts using a separately implemented scanner. The full executable H42 analysis was clean-directory replayed with SHA-256 manifest verification and identical output files.

The original H36 prospective 29/30 FAIL and H38 invariant-regression FAIL are preserved. The original H40 restricted 30/30 PASS is not invalidated by these posthoc synthetic-input tests, but H42 demonstrates that endpoint correctness does **not** yet warrant the stronger claim that the frozen receiver has learned a smooth, monotone or globally selective material-law rule.

## Smallest next intervention

Within the same 209-parameter architecture, add hard per-world no-regression constraints **along the donor-feature segment defined by the two existing trained donor laws**, using only exposed worlds. Invariant cases, especially equal-wet, should never flip inside the segment; divergent cases should have one correctly oriented threshold with calibrated position. Preserve genuine donor-source stability, both original law endpoints, abstain from oracle-fed inputs at inference, and retain zero/neutral/shuffled sham results. Re-freeze a candidate before registering another genuinely unseen-family cohort using genuinely trained donors. Do not add model capacity, maze complexity, new material laws, or claims about general memory transplant.

## Custody

- Reproducer committed in this PR: `experiments/memplant/material_causal_gate_020/h42_independent_compact_replay.py`.
- Full H42 evidence: `DG020-H42-posthoc-donor-channel-negative-20261119.zip` in owner handoff. Archive contains original H40 receipts and all 11 frozen checkpoint files, exact executed H42 source, H41 helper source, 4,800-row per-world H42 diagnostics, independent audit, logs and hash manifest. H40 original 67 MB historical package remains separately preserved.
- To replay: extract the H42 archive and run `python h42_posthoc_donor_channel_stress.py . --out replay`, or run the committed compact independent scanner against that extracted evidence root.
- Still **outstanding**: H40/H42 binary and full receipt GitHub custody, exact-HEAD H40-specific pinned CI using the original assets, and genuinely external methodological review.

**Governance:** existing `techrote/neurounner` PR #253 only; remain Draft/HOLD, never Ready/merge. No CyberSand/DRIV3R writes, no new branch/PR/issue, no paid GPU/API compute.
