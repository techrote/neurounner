# DREAMGRAFT DG020 — H48 retrospective controls and cross-runtime neural audit

**Date:** 2026-11-19. **Classification:** exposed-only methods/replay review. **No new prospective gate or checkpoint training.** All work remains on the existing Draft/HOLD PR #253.

## Disposition

- **H48 authentic control accounting: PASS.** Verified all 35/35 original H47 SHA-256 manifest entries, original 4,800 H40 paired observations, 8 genuine trained donor checkpoints, and six original/candidate 209-parameter receiver checkpoints. Reexecuting H47's actual neural negative-control script reproduced **all 108 stored decision arrays exactly** (zero differing entries, byte-identical NPZ SHA-256 `64e218a105cc5720547ae64cbfff18cff0ea1b38163322e01c27680bff666097`).
- **H48 bit-exact cross-runtime full-network replay: FAIL.** Original independent H47 script exits because regenerated donor probes differ from the archived float32 endpoint values by up to `1.49e-7` on this CPU. An explicitly labeled conditional replay (authenticate original trained checkpoints; verify regenerated donor probes within `1e-6`; then replay the **archived donor scalar inputs**) evaluated **7,526,400** neural route margins, with **3,406,554 different float32 bit patterns**, maximum absolute logit-margin drift **0.0000762939453125**, and **one changed action**. Therefore do **not** claim a portable bit-exact full-network replay PASS. Original H47 zero-drift results refer to its original CPU environment.
- **Original H47 mechanistic repair: FAIL.** Calibrated donor-source alignment still leaves **5,060** erroneous synthetic off-grid decisions; original H47 findings are retained.
- **Prospective scientific gate: NOT RETESTED.** H40's restricted historical **30/30 PASS** remains as originally reported, subject to H41 placement-insensitivity and H46 synthetic-law non-identifiability. No broader material-law inheritance, planning or game integration is established.

## Control-informativeness finding

These are **retrospective** counts on the 4,800 exposed H40 worlds: 640 divergent and 960 invariant worlds per receiver. Donor-dependent rows cover six receiver checkpoints × four authentic donor-source profiles, while law-invariant zero/neutral rows cover six receiver checkpoints.

| Genuine neural input condition | Divergent cheap arm correct | Divergent expensive arm correct | Divergent both correct | Invariant both correct |
|---|---:|---:|---:|---:|
| Correct trained donor-law checkpoint | 15,360/15,360 | 15,360/15,360 | 15,360/15,360 | 23,040/23,040 |
| Opposite-law trained donor checkpoint (pre-inference swap) | 0/15,360 | 0/15,360 | 0/15,360 | 23,040/23,040 |
| Shuffled trained donor-law checkpoint (pre-inference, frozen RNG) | 7,698/15,360 | 7,789/15,360 | 3,874/15,360 | 23,040/23,040 |
| Literal all-zero donor **cost tensor** | 1,920/3,840 | 1,920/3,840 | 0/3,840 | 5,760/5,760 |
| Neutral `1+fee` cost tensor | 1,920/3,840 | 1,920/3,840 | 0/3,840 | 5,760/5,760 |

The zero and neutral inputs resulted in **identical actions for all 9,600 tested receiver/world instances**, although the tensors are different. This is not proof of shortcut learning; it shows the two controls provide no independent **action-level** discriminatory evidence in this cohort.

The **zero both-correct** count for a law-invariant control on a divergent counterfactual pair is **mathematically guaranteed** because the correct two actions are different and the control receives identical inputs for both arms. Likewise its exactly 50% aggregate per-arm divergent accuracy is forced by the matched-pair design. Such zero-donor or neutral results alone are **not positive evidence** that genuine donor-conditioned behavior is causal. The genuine vs opposite-law endpoint contrast is informative within the existing engineered two-law task; shuffled 3,874/15,360 (25.22%) is chance-like.

All donor-dependent invariant endpoints remained correct, with zero wrong law-induced invariant flips on these exposed worlds.

## New numerical reproducibility finding

A distinct CPU runtime (Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, Linux x86-64) failed original H47 **exact-equality** donor-probe reconstruction. In the conditional replay described above, one **synthetic off-grid** decision changed: seed `26011641`, H45 candidate, source `29002179-rep0`, unaligned cost, world index **255**, alpha **255/256** (off-grid index 191), saved margin **+4.76837158203125e-7** versus replay margin **0.0**. **All genuine two-law endpoint decisions were retained.** This very near tie is consistent with previously disclosed H45/H47 one-decision drift, but the exact low-level cause has not been proven. Host-specific bitwise failures must not be silently relabeled PASS.

## Evidence and custody

Original H47 archive SHA-256: `88ef6fcbac73bc8ebe4f0f326b74beebab92633e46a1cd949437e8aa353aa6b0`.

H40 receipt SHA-256: `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`.

The original H47 ZIP has a reproducibility packaging defect: `h47_donor_source_alignment.py` imports `h45_independent_neural_replay.py`, but this dependency is missing from that ZIP. The H48 **self-contained supplementary evidence ZIP** includes the unchanged H45 dependency, both independently executed H48 scripts, full original H47 payload, raw H48 audit outputs and logs, and a new SHA-256 manifest. H48 evidence ZIP SHA-256: `945cd0a327a4c6455860c735fcfcbda113c5a1f3133fe7ff226eb52f2cf5c402` (39,623,449 bytes). The original H47 manifest still verifies its original 35 members; no negative evidence overwritten.

Executed H48 source SHA-256:
- `h48_control_informativeness_audit.py`: `a7199d7025d0149ac14e3f5dda86e2c336b05e8a4fe8f97bd4614d35e3d86825`
- `h48_cross_runtime_replay.py`: `fc11348c814936da541680256eda25896848f092ca6a1197f110b33ce8a646a0`
- Original missing H45 import added unchanged: `1ff05b4e80a50e23edce26ca48164345ee90a28a1469c207c6a078e88f77deee`.

The complete binary H48 evidence remains in the owner-facing downloadable ZIP; **it is not yet fully committed in GitHub**, and external methodological review remains outstanding. A report commit does not resolve those custody blockers.

## Smallest justified continuation

1. Preserve H40/H47 genuine checkpoints, original receipts, all negative decisions and full H48 sources under durable GitHub/CI custody without modifying the scientific record.
2. Require pinned CPU CI to report exact file/checkpoint hashes, both-direction genuine endpoints, invariant-world no-regression, sham controls and explicit numerical tolerances; record cross-host bit drift as a separate diagnostic.
3. Obtain independent methodological review of the engineered 5×5 material probe and placement-insensitive 2×12 tile sum before any new prospective claim. Stop optimizing H46's non-identifiable intermediate-alpha objective.

**Governance:** PR #253 remains **Draft/HOLD, not Ready, not merged**. No new issues, branches, PRs, laws, receivers, topology, paid GPU/API, or CyberSand/DRIV3R changes.