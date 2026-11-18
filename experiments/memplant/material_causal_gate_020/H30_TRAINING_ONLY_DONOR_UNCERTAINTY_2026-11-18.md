# DG020 H30 — genuine donor uncertainty interval diagnostic (18 November 2026)

**Classification: TRAINING-ONLY NEGATIVE DIAGNOSTIC — not a prospective scientific PASS/FAIL cohort.** The latest valid stricter preregistered gate remains **H27 FAIL 28/30**; H28/H29 are insufficient training-only pilots. No new holdouts were created or evaluated.

## Experiment

Reconstructed the original 7,680 *previously exposed H27 near-tie training examples* (2,560 per receiver seed), verified each against the original H27 training-observation SHA-256 list, and evaluated the original four genuine trained donor source/replicate pairs. The analysis calculates route-advantage intervals by directly aggregating genuine donor predicted per-tile costs as a **diagnostic only**, and separately replays the original frozen 209-parameter H27/H29 receivers on exactly those inputs. It does not replace a receiver with an analytical proxy for any success claim. Both material-law arms have identical geometry, fees, textures, legal action masks and receiver weights; law change affects only inherited donor wet-cost information.

| Original training-only measurement (7,680 examples) | Result |
|---|---:|
| Direct additive genuine-donor interval crosses action boundary | **375** |
| Direct additive donor action disagreements | **375** |
| Of those, invariant pairs | **375** |
| Of those, divergent pairs | **0** |
| Frozen H27 receiver, individual same-law donor changes | **136** |
| Frozen H29 receiver, individual same-law donor changes | **120** |
| H29 changes with oracle margin < 0.001 | **120** |
| H29 changes despite stable direct-additive donor-cost sign | **55** |
| H29 changes on divergent training pairs | **55** |
| H29 changes on invariant training pairs | **65** |
| Independent cross-audit of original H29 raw neural receipts | **7,680/7,680 PASS** |

The 375 sign-straddling cases arise solely in the wet-rich and wet-poor invariant categories (208 and 167). The H29 residual errors decompose into 49 wet-rich, 16 wet-poor, 13 cheap-edge divergent and 42 expensive-edge divergent. This exposes **two mechanisms**, not one: uncertainty in genuine donor wet-cost forecasts near invariant action boundaries, *and* receiver margin-calibration distortions in divergent cases where the direct additive donor-cost sign is stable.

**Interpretation:** The 375 analytical disagreements are not an impossibility proof for a trainable nonlinear receiver. H29's 120 remaining donor-source changes do not constitute a prospective cohort. No general material-law inheritance or game-policy functionality is established.

## Reproducibility

Original signed H29 evidence archive was reverified unmodified: SHA-256 `42712e02825ff4d8782b954f7ca42edad774b95e3f25a00996f9eb7e65737890`. H30 original audit source SHA-256 `103c00287cbd911a43bda28e213971eef7220c861c359bd15da39445a84ad2e9`. An independent second script compared every new H30 receipt against the previously retained H29 full observation, oracle and neural source-replicate decision streams. Exact 7,680/7,680 agreement; 0 identity/order/action mismatches.

The owner handoff archive `DG020-H30-trainingonly-donor-uncertainty-NEGATIVE-20261118.zip` contains the **unchanged original H29 archive**, executed H30 source, 7,680 donor uncertainty receipts, report, independent cross-auditor and manifest. Archive SHA-256 `82ddc04b644247ddc75c3099449da1485040831546d90dfe024bc06534a351d4`, 15,904,396 bytes. The original full ZIP remains in the owner handoff, **not yet durable Git custody**. An exploratory nonpreregistered multi-fee probe ablation is included, explicitly not a qualified candidate.

## Mechanism decision and next gate

**Do not open H30 prospective evaluation.** On *training-only* matched pairs first, investigate single-genuine-donor source-blind cost denoising (no source ID, law label or hard-coded cost-class lookup) **and** independently calibrate the existing receiver's route-advantage slope/offset. Compare per-seed divergent/invariant near-ties against all genuine source-replicate variations, zero, sham, and shuffled controls. Only an unequivocal improvement permits a fresh preregistered unseen-family confirmation. Preserve negative results.

**Governance:** Existing Neurounner PR #253 only; Draft/HOLD, no merge or Ready, no additional branch/issue/PR, no paid GPU/API spend; CyberSand and DRIV3R read-only.
