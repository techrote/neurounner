# DG020 H50 — retrospective genuine-neural receiver precision audit (19 November 2026)

**Disposition: H50 exposed-only numerical replay PASS. Prospective causal-selectivity gate NOT RETESTED.** This does not alter H40's historical restricted 30/30 PASS, H41's placement/negative-control limitations, H46's synthetic-input non-identifiability, H47's off-grid FAIL or H48's cross-runtime float32 disagreement. PR #253 remains Draft/HOLD, unmerged.

## Design

Independent CPU float64 re-evaluation of **six authenticated trained 209-parameter receiver checkpoints**: the original three frozen H40 receivers (H36 seeds 26011639/26011640 and H39 seed 26011641) and the three H45 negative repaired candidates. The test used all 4,800 original H40 worlds, eight authentic trained-donor checkpoint identities and their four archived genuine trained-donor source/replica scalar profiles, both physical laws, and separately H47's engineered donor-source alignment.

**This particular audit does not rerun donor inference.** It freezes the H47-preserved genuine trained donor scalar values, rounds the material scalar to float32 as at the original receiver boundary, then compares the original archived float32 route-score signs with a fresh CPU float32 and direct float64 evaluation of the *same* receiver network and world. No geometry, fees, goal, legal mask, texture, receiver parameters or donor scalar input changes within an arithmetic comparison.

H40 raw receipts SHA-256: 5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d.
H47 archived margins SHA-256: 77e96494144c2015dab45ad4823dfbc17056b1c064cdd2e5a793e1c324da7bee.
Executed H50 script SHA-256: 1561542377a8ed8f04e88d68a853624d595d65f7a7f03325c98ee4258569fe50.
Executed H50 full result JSON SHA-256: c486d2ca7420a6cbb4fbf6ff7bace8e50aa27c6c1536571497c18b88a0d518d8.

## Observations

| Evaluation condition | Genuine endpoint decisions | Archived-f32 to new-f64 action changes | New-f64 wrong actions | Minimum absolute new-f64 margin |
| --- | ---: | ---: | ---: | ---: |
| H40 original trained donor scalars | 38,400 | **0** | **0** | 0.000133004615 |
| H40 engineered source-aligned scalars | 38,400 | **0** | **0** | 0.000141387245 |
| H45 candidate, original donor scalars | 38,400 | **0** | **0** | 0.000007354694 |
| H45 candidate, source-aligned scalars | 38,400 | **0** | **0** | 0.000010011604 |

Fresh float32 versus archived float32 **0 action changes** in all four conditions. Float64 violations of divergent physical-law correctness **0**, invariant correctness **0**. All 3 seeds, both world families, all five bucket types and four trained-donor source/replica profiles were included. This audit establishes arithmetic sensitivity of these **exposed endpoints** only, not a new prospective physical-law test.

In raw units, the worst H45 candidate minimum float64 margin is about eighteen times smaller than H40's minimum. H45's previously documented thousands of **synthetic off-grid failures remain uncorrected**. In particular, float64 endpoint parity neither rescues H46's ill-posed cross-profile synthetic target nor makes H47's calibration mechanism a genuine single-donor law transplant.

## Reproduction and custody

Executed on CPU PyTorch 2.10.0+cpu / NumPy 2.3.5; a clean archive extraction regenerated the complete H50 result JSON **exactly** and verified all 30 original upstream/evidence payload SHA-256 entries. The complete 21,937,579-byte H50 owner handoff archive contains the executed Python source, output JSON, all original H40 raw rows, H47 raw margin arrays and control receipts, six receiver checkpoint files, eight original genuine donor checkpoints, retained H46–H48 negatives, a manifest and clean-room instructions. Archive SHA-256: a428c25d146cf7a6b182c9088d83713aed4abb307646aa1c3d1b05848a32c056.

**Limitation:** These complete binary assets are in owner handoff, not yet durably hosted on GitHub. An exact-HEAD GitHub Actions replay of H50 cannot be claimed until the original inputs are in Git custody. The empirical H48 cross-machine maximum float32 margin drift (0.0000762939453125) is not a mathematical or universal cross-CPU bound. H50 changes arithmetic on one CPU; it does not establish cross-vendor bit-exactness.

**Smallest next action:** retain complete binary source/receipts and original negative evidence in Git-backed custody, enable an exact-HEAD action-level replay for authentic checkpoints, then obtain independent methodological review of shortcut/leakage claims. No new holdout, receiver expansion, law family, branch, issue, PR, paid GPU/API compute, CyberSand or DRIV3R write is justified by H50.
