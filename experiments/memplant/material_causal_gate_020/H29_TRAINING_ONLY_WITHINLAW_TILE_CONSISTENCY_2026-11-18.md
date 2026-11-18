# DG020 H29 — training-only within-law donor consistency (18 November 2026)

**Disposition: training-only mechanism trial INSUFFICIENT; scientific causal-selectivity gate remains unpassed.** The most recent prospectively registered stringent test is H27: **FAIL 28/30 cells**. H28 and H29 are engineering pilots. **No H28/H29 prospective cohort was generated, scored or inspected, and this report does not claim a scientific PASS.**

## Exact experimental intervention

Started from the three **original SHA-256-verified H27 genuine neural receiver checkpoints** (209 parameters, seeds 26011639/26011640/26011641), as H28 had done. Held constant the identical 5,120 H27 **training-only** matched worlds per seed (2,560 original training worlds plus 2,560 near-margin variants), minibatch schedule, four individual genuine trained donor pairs, original two law values 0.35/2.30, route topology, learning rate 0.00004, 1,600 CPU AdamW steps, and all H28 calibration/trust/hinge losses.

The **only H29 mechanism change** was an additional training loss enforcing same-law per-tile receiver output consistency across four independently trained source/replicate donor pairs:

`loss_H29 = loss_H28 + 80 * mean((tile_output[source_rep,law,route,tile] - mean_source_rep(tile_output[law,route,tile]))**2)`

Absolute per-law tile cost and route-margin supervision and paired law-delta calibration remained active to avoid rewarding a globally donor-blind policy. Oracle margins are training targets, never receiver inputs. No family identifiers, law labels, oracle distances, future actions or legal-mask changes were introduced at inference.

The executed H29 source SHA-256 was **`df422e8f780cff894654e419fe3af75d6f45791f03b113271fbe4f6049bea0c1`** (`h29_withinlaw_tile_consistency_trainonly.py`). Local CPU PyTorch 2.10.0. That source has **no H29 prospective evaluator or holdout generator**.

## Independent training-only re-evaluation

All entries below use the *same 7,680 previously seen near-margin H27 training observations* (2,560 per seed) and all four authentic donor pairs; absolutely no prospective samples. The independent audit re-created every training-world hash and matched the original H27, H28 and H29 receiver-checkpoint digests before inference.

| Training-only outcome | H27 | H28 | H29 |
| --- | ---: | ---: | ---: |
| Any individual same-law donor-replicate action changes | 136 | 122 | **120** |
| Wet-rich invariant donor changes, BOTH training families (1,536 cases) | 65 | 48 | **49** |
| Expensive-edge divergent donor changes, BOTH families (1,536 cases) | 30 | 45 | **42** |
| Primary neural action errors | 0 | 0 | **0** |
| Any genuine donor-replicate action error | 136 | 122 | **120** |

Seed-level action changes (H27 → H28 → H29): **26011639: 62 → 56 → 55; 26011640: 34 → 32 → 33; 26011641: 40 → 34 → 32**.

This corrects a reporting granularity limitation in the earlier H28 brief: its wet-rich 28→24 comparison covered just one of the two training families (256 observations per seed); the full two-family wet-rich counts are 65→48→49. No prior prospective verdict is changed.

**FAIL to justify a prospective experiment.** H29 suppresses only **two** additional donor-induced training action changes compared with H28 and slightly *increases* wet-rich invariant failures (48→49). No meaningful training-only causal-selectivity repair has been shown. Avoid selecting or relaxing acceptance criteria based on H27's exposed holdout.

## Original byte custody and independent replication

H29 original checkpoint SHA-256:
- 26011639: `bd7480c6d34a75ce1cc1c8f8869936aae501a86c6914c73f398981d5c06b7455`
- 26011640: `1eabd720b94431520d683a10c397c3698cd4b35b0fbebe4bf7ed8ac353c783ca`
- 26011641: `4424e95c72f7a248649fb86ab5c223f33e489aa20a23c277189552baf5f3aa75`

The owner handoff bundle `DG020-H29-training-only-withinlaw-tile-consistency-INSUFFICIENT-20261118.zip`, **14,971,748 bytes**, SHA-256 **`42712e02825ff4d8782b954f7ca42edad774b95e3f25a00996f9eb7e65737890`**, includes the executed H29 source, three new checkpoint bytes, three H27/H28 comparison checkpoints per seed, genuine donor checkpoint capsule, all 7,680 raw near-tie training observations and H27/H28/H29 four-donor neural actions and logits, exact training hashes, independent audit, original negative outcomes, and an 80-entry SHA-256 manifest. **80/80 manifest files verified; exact archive ZIP integrity PASS; clean-directory independent re-evaluation PASS.**

The full H29 handoff ZIP is **not yet in durable GitHub custody**, so these text hashes alone are not a substitute for Git-backed original bytes. Likewise, H29-specific exact-HEAD GitHub Actions neural replay has **not** been performed. Do not claim those qualifications.

## Smallest next mechanism change

Within-law tile-output variance is insufficient as a mere extra penalty. Next use **training-only conditional donor-feature canonicalization/denoising** to map the narrow same-law source/replicate estimates toward one robust estimate, while preserving the large genuine *between-law* delta. Explicitly constrain worst-replicate action margins for divergent and invariant training pairs, and ensure both-direction law correctness before any new prospective registration. Keep the 209-parameter receiver, two laws, fixed geometry and non-pooled seed/family/bucket gates. A fresh unseen-family holdout requires precommitted source/checkpoints/thresholds first.

Governance: only existing Neurounner Draft/HOLD PR #253; no Ready, merge, new PR/issue/branch, paid GPU/API usage, or CyberSand/DRIV3R writes.
