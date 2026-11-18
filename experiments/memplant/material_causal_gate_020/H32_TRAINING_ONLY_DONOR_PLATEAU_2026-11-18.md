# DREAMGRAFT DG020 H32 — source-blind donor plateau: training-only mechanistic result

Date: 2026-11-18. Ownership: `techrote/neurounner`, existing Draft/HOLD PR #253. Never merge/Ready; no new branch, issue, PR, paid API/GPU, or CyberSand/DRIV3R edits.

## Scientific disposition

**Latest stringent preregistered prospective verdict remains H27 FAIL (28/30 cells). H32 has no prospective scientific gate verdict.** This experiment addresses near-tie donor-source variability entirely on existing H27 TRAINING examples, not unseen families. H26's restricted prospective PASS and H25/H26 near-tie post-hoc negatives remain intact.

## Bounded hypothesis and executed interventions

Genuine trained donor replicates vary within each inherited material law. H31's ReLU receiver remains locally affine across that variability, causing 124 source-induced action changes on 7,680 exposed near-tie training matched pairs.

**H32** uses genuine frozen H21 donor outputs and unsupervised one-dimensional largest-gap clustering to identify three unlabeled donor material-output regimes. It configures the *same* 209-parameter PyTorch `Receiver.tile` into a low-gain, piecewise-flat response, with four ReLU hinges between the donor-derived clusters. Its four-effective-coefficient output head is fitted by least squares against original H27 training-world tile supervision. No fixed material-law values, family IDs, oracle scores, answers or donor-source IDs enter inference. **This is a strongly engineered three-regime decoder, with an output head trained on genuine donor-conditioned observations: it is a mechanism/upper-bound control, not evidence of emergent end-to-end neural inheritance.**

**H32b** starts from H32, applies deterministic seeded damage, and trains *all 209 weights* for 1,000 local CPU AdamW updates with existing H27 training worlds, individual genuine donors, feature jitter and a donor-defined plateau-teacher regularizer. **H32c** recalibrates H32b's last neural layer against the original training tiles. No unseen holdout was generated or evaluated.

Within matched pairs, routes, fees, masks, goals and receiver weights remain identical; only the inherited neural donor feature changes between laws.

## Independent replay: full two-family 7,680 matched TRAINING pairs

Original H31 evidence manifest was verified before replay. H27's 15,360 original training observation hashes were reproduced across the three receiver seeds. Each original checkpoint was loaded as the genuine PyTorch 209-parameter network; four genuine independent donor source/replicate pairs supplied all features. Independent replay emitted 7,680 per-world source-action/logit receipts.

| Measurement | H31 | H32 engineered plateau | H32b full-weight retrain | H32c recalibrated |
|---|---:|---:|---:|---:|
| All four genuine donors both laws correct / 7,680 | 7,556 | **7,680** | 7,663 | 7,675 |
| Within-law donor-source action changes | 124 | **0** | **0** | 1 |
| Inappropriate invariant law flips | 19 | **0** | 7 | 0 |
| Divergent both correct / 3,072 | 3,008 | **3,072** | 3,062 | 3,067 |
| Invariant both correct / 4,608 | 4,548 | **4,608** | 4,601 | 4,608 |
| Swapped-law donor divergent both correct | 0 | 0 | 0 | 0 |
| Zero-donor divergent both correct | 0 | 0 | 0 | 0 |

An independently seeded per-arm **wrong-law shuffled donor** control achieved H32 769/3,072 divergent both-correct (approximately chance) versus 3,072/3,072 with correctly associated genuine trained donors. The same-law four-replicate control produced zero H32 action changes.

**H32b fails training-only strict robustness:** 17 incorrect pairs (10 expensive-divergent, 7 wet-rich-invariant), including seven inappropriate invariant flips, despite eliminating replicate-source disagreements. **H32c still fails training-only strict robustness:** five expensive-divergent incorrect pairs and one renewed individual donor-source disagreement.

## Limits, custody and next smallest change

H32's perfect results apply to previously exposed H27 TRAINING worlds, with a hand-initialized three-regime plateau. They cannot be reported as a new prospective PASS or as broad, open-ended material-law transplantation. H32b/c genuine all-weight training has *not* yet jointly achieved zero donor-source changes, zero invariant flips and correct actions at ultra-tight margins.

Owner handoff: `DG020-H32-trainingonly-donor-regime-repair-20261118.zip`, SHA-256 `daa0629328f3a805c4796369cac8ac3355b1df23d354078224501adc3f9e14d3`, 10,822,805 bytes, 22 independently SHA-256-verified files. Includes original H31 archive and original donor checkpoints, all exact executed H32/H32b/H32c scripts, all original trained checkpoint bytes, seeded controls, 7,680 independent action/logit receipts, independent replay source, negatives and full manifest. **Complete H32 original bytes are in owner handoff, not yet in durable GitHub custody or exact-HEAD H32 CI.**

**Next smallest mechanism change:** retain the same original 209-parameter receiver and genuine donors. Use training-only full-network plateau initialization plus explicit final absolute route-margin calibration, demanding zero individual same-law source action changes, zero invariant law flips and all-donor both-correct on the original training perturbations; do not open a fresh preregistered unseen-family cohort until this joint criterion is met. Do not treat the hand-designed plateau as emergent learned donor inheritance.

**Disposition:** H32 training-only upper-bound mechanism positive, H32b/c training-only repairs INSUFFICIENT; stringent scientific gate remains **FAIL**. PR remains **open Draft/HOLD, unmerged**.
