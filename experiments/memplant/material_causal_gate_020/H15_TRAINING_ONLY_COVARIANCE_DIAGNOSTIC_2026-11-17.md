# DG020 H15 — training-only donor covariance / fee-nuisance diagnostic (17 November 2026)

**Exploratory exposed-training analysis only. No H16 acceptance, and H15 remains SCIENTIFIC FAIL 19/30. No models were retrained.**

## Custody
H15 experiment source SHA-256 `499db557a2c11f980071b65bcbbd7515a29b2310f236288c6889a452809e4c75`. Independently regenerated **12,000 archived H15 training worlds** and exactly matched the 12,000 saved training-world hashes. Original three ridge calibrators replayed at zero coefficient difference. Genuine frozen learned donors (two training same-law source groups per seed) and frozen trained H15 receivers were used. **No H15 acceptance receipts nor the third donor group were examined.** These 48,000 law × source × training-world evaluations do not constitute 48,000 independent worlds.

## Finding 1 — neural donor route-advantage errors are conditional, correlated and asymmetric

| H15 calibrated donor-advantage metric, training only | Cheap law | Expensive law |
| --- | ---: | ---: |
| RMSE | 0.009933 | **0.026584** |
| Wrong donor route sign | 55 / 24,000 | **1,142 / 24,000** |
| Wrong sign in wet-rich invariant category | 0 / 4,800 | **895 / 4,800** |
| Wrong sign in expensive-edge divergent category | 0 / 4,800 | **247 / 4,800** |

Across the three receiver seeds, the two independently trained donor-source groups' expensive-law residuals are substantially correlated (Pearson **0.587, 0.645, 0.915**, 4,000 matched training observations per seed). Thus naive fusion cannot remove the common nuisance. An exploratory law/source-ID-blind ridge residual correction using visible wet counts, wet-fee statistics and donor forecasts failed to improve cross-*training*-family RMSE: balanced→asymmetric **0.0200212→0.0200223**; reverse **0.0201128→0.0201174**. These two families were already H15 training data, **not new generalization holdouts**.

## Finding 2 — donor-only fee standardization is promising, but not sufficient alone

Feed the **genuine trained donor**, rather than the receiver, a constant reference fee **0.15** per tile when extracting the learned material component, then restore each actual observable tile fee before the unchanged H15 corrector/receiver. Keep observed wet mask, texture, geometry, goals and masks intact; no law IDs, family IDs, oracle costs or answers enter inference.

| Donor-only intervention, H15 training observations | Expensive-law wrong route signs | Wet-rich wrong route signs | Expensive-law RMSE |
| --- | ---: | ---: | ---: |
| Unmodified per-tile fee input | **1,142 / 24,000** | **895 / 4,800** | 0.026584 |
| Fixed fee 0.15 | **295 / 24,000** | **39 / 4,800** | 0.015396 |
| Mean fee per material | 295 / 24,000 | 162 / 4,800 | 0.016368 |
| Zero texture | **2,676 / 24,000** | **2,676 / 4,800** | 0.037764 |

However, simply switching **the frozen trained receiver** to fixed-fee donor input (primary training donor group, 2,400 cases per bucket) improves wet-rich invariant both-correct **2,129→2,397**, but worsens expensive-edge divergent both-correct **2,228→2,102**. Wet-rich training-group sham action changes fall **340→3**, but are exposed training results. **No gate PASS**: the unchanged receiver is not calibrated to the altered input distribution; both-direction correctness still fails.

## Smallest next change (H16 candidate, NOT executed)

Retrain **only the existing 209-parameter receiver** on **donor-only fixed-reference-fee neural forecasts with actual observed fees restored**, using both matched divergent and invariant pairs and same-law donor consistency. No enlargement of laws, model, worlds or topology. Keep original-fee receiver conditioning as a control. Preregister scripts, thresholds, seeds, fresh untouched holdouts, source hashes and independent auditor **on PR #253 before new inference**. Gate must enforce both-direction divergence, invariant selectivity, same-law withheld-group sham stability, genuine pre-inference shuffled/zero/source-blind controls, and unseen-family generalization, without pooled metrics masking failures. Existing H15/H14/H13 cohorts stay exposed negative/control evidence.

## Archive and limitations

The associated conversation archive `DG020-H15-training-only-covariance-nuisance-audit-20261117.zip` (SHA-256 `f9a128c54109de17cf4d4dbfa864ea3b870ece5b3c62c15340e31dd0e8fe7821`) includes all reproducibility scripts, authentic frozen donor and receiver checkpoints, generator source, 12,000 archived training hashes, 48,000 route-error table records, ablation outputs, independent receipt-accounting report and **46 verified SHA-256 manifest entries**. Accounting verified 48,000 unique evaluation keys, 20 route-error category recalculations, 24 ablation summary rows, and 120 frozen-receiver rows. Local CPU PyTorch 2.10.0 only; **not exact-head pinned CI**.

**Scientific gate still FAIL; existing PR remains Draft/HOLD.** No new branch, PR, issue, paid compute or CyberSand/DRIV3R changes.
