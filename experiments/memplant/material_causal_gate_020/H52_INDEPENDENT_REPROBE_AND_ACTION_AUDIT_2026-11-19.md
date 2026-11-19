# DG020 H52 — independent trained-neural donor reprobe and continuous-invariance audit (2026-11-19)

**Disposition:** EXPOSED-ONLY mechanistic PASS, prospective H53 scientific gate NOT EXECUTED. No checkpoint promotion.

## Strict source identity

- Reconstructed exact H52 checkpoint from original authentic H45 seed 26011640: `b1662812437ae4a83e16f0f345e2c8291acf48e8748776340d98f4be9a9fc9fc` (byte-identical to documented candidate).
- Locked original H40 receipt SHA-256: `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`.
- Independently verified SHA-256 of all 8 genuine, unchanged H34 trained donor checkpoints; maximum original validation MSE `0.0009282943` (below locked 0.02).
- Re-probed donor networks with original 5x5 Legendre quadrature, actual Tanh activations, dry-cost anchoring. Maximum absolute difference versus archived H47 endpoints `1.19209289551e-07`.
- Receiver is the genuine 209-parameter H45 checkpoint with only eight output-head float32 coefficients replaced per committed `h52_rebuild_head_patch.py`. No model retraining, changes to geometry or true-law membership, labels/family IDs/oracle answers at inference.

## Independently recomputed actions

| Check | Measured result |
| --- | ---: |
| Genuine donor-trained-law endpoint actions (seed 26011640, 1600 worlds x 4 profiles x 2 laws) | 12800 |
| Incorrect genuine endpoint actions | **0** |
| Profile-wise continuous invariant interval ReLU knots, including profile endpoints | 405768 |
| Wrong analytical float64 invariant knot decisions | **0** |
| Wrong independent full-PyTorch float32 invariant knot decisions | **0** |
| float32 / float64 action disagreements | **0** |
| Previously published H51 negative witnesses checked using observation-SHA disambiguation | 20 |
| H51 witnesses still wrong | **0** |
| Lowest signed full-PyTorch float32 knot/endpoint margin | `8.392333984375e-05` |

The report H52 stated **101,442** per-world unique knot evaluations; this independent audit enumerates **405,768** world × genuine-source-profile knot evaluations, including each profile's endpoints. These are different accounting units, not contradictory counts. All original H40 worlds are **exposed**, so this establishes no unseen-family generalization. Float32 guard bands are empirical on this CPU; not a formal cross-runtime proof.

## Scientific gate integrity and H53 provenance blocker

H53a previously failed before any new-cohort neural evaluation because it used the wrong donor activation. H53b's repaired script is separately preregistered as SHA-256 `93111827873a8b11236859adaf2801566635c4210d062d78e68095945c75740a`. **At this audit, that exact source file is not present in the current PR tree, and it is not present in the owner evidence archives available to this run.** Therefore this audit did *not* reconstruct a purportedly equivalent generator, alter any preregistered threshold, or evaluate the H53 new cohort. The 45-cell all-perfect prospective gate remains **NOT EVALUATED**. Publishing and hashing the exact preregistered source and fixed H52 binary should precede genuine H53b execution. The H53a negative and H51 prior negatives are retained.

**Governance:** existing `techrote/neurounner` Draft/HOLD PR #253 only. No changes to CyberSand or DRIV3R, no paid compute, additional branches, PRs or issues. No merge/Ready.
