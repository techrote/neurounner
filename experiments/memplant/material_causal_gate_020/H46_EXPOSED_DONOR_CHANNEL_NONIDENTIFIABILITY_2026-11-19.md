# DREAMGRAFT DG020 — H46 exposed donor-channel identifiability result (19 Nov 2026)

**Preregistered before testing:** https://github.com/techrote/neurounner/pull/253#issuecomment-7005695915

**Classification:** H46 bit-identical contradictory-input witness **PASS**; universal nominal-law-label correctness for all synthetic interpolated donor channels **mathematically infeasible**. This is an **exposed-only methodological falsification**, *not* a new prospective causal-selectivity result. H40's restricted, originally preregistered 30/30 PASS remains historical and unchanged; H41's spatial/generalization and negative-control limitations remain.

## Precise obstruction

The original H40 test supplies 4,800 matched exposed worlds, four independently trained genuine donor-source profiles, and three genuine 209-parameter receivers. H42–H45 interpolate each source's measured trained endpoint wet-cost signals as c_s(alpha) = (1-alpha) c_s0 + alpha c_s1. Their target labels instead use the *nominal* physical cost c*(alpha) = 0.35 + 1.95 alpha. For a divergent world with wet-count difference dw and fee-sum difference df, the nominal oracle's action changes at alpha* = (1 - 0.35 - df/dw)/1.95.

Two donor sources generally have different c_s(alpha*) because genuine donor estimates are not exactly equal. Selecting the same representable float32 donor cost strictly between those two source estimates yields two different interpolation alphas straddling alpha*. One physical label is action 0 and the other action 1, while the actual complete receiver input tensor is **bit-identical**. No deterministic receiver—regardless of output-head or hidden-knee optimization—can produce both answers without an additional feature, which would violate the restricted inference contract. The calculation was independently checked on the *real complete PyTorch models*; this is not merely a proxy feature result.

## Locked audit results

| Metric | H46 |
|---|---:|
| Original exposed worlds | 4,800 |
| Divergent worlds with an interior target-law transition | 1,920 (640 per receiver) |
| Divergent worlds with at least one complete bit-identical, opposing-oracle-input witness | **1,920/1,920** |
| Six cross-profile candidate pairs per divergent world | 11,520 |
| **Verified exact contradictory pairs** | **11,286** |
| Float32 inverse round-trip attempts that did not produce a shared cost | 234 (retained negative/inconclusive attempts) |
| Independently replayed contradictory pairs | **11,286/11,286** |
| Checkpoint/source/observation discrepancies in independent CPU audit | 0 |
| Receiver/donor training or weight changes in H46 | 0 |

First examples (world-local index 0, family interleaved_braid): receiver seed 26011639 has nominal threshold alpha*=0.001049772287, with two alphas 0.000904666309 and 0.001194780145, oracle actions 0 and 1, but identical entire input and logits [-44.30488586425781, -44.31460189819336]. The corresponding physical cost margins have opposite signs: -0.000282956657 and +0.000282765323. Seeds 26011640 and 26011641 also have fully verified examples, with opposite-action labels 1 and 0.

These are **contradictory target labels**, not a claim that the network made 11,286 distinct action errors. Every witness makes exact universal compliance logically impossible. The 234 unsuccessful construction attempts do not prove impossibility and were not counted.

## Evidence and reproduction

Original H40 raw receipt SHA-256: 5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d.

Owner evidence: DG020-H46-synthetic-channel-identifiability-20261119.zip, SHA-256 aa6ef16e42dc7abb4d8570fec40ac3b15ca053a48ff841f107fd9112c8d55728. All eight genuine donor checkpoints, three original receiver states, three frozen H45 receiver states, H40's 4,800 receipts, both executed scripts, all 11,520 per-pair attempt receipts, all 11,286 exact witness receipts, independent full-neural replay, logs and SHA-256 manifest. Clean archive ZIP integrity and per-file manifest verified.

Extract the owner archive and run the two included full scripts:

- python h46_donor_channel_identifiability.py . --out h46-output
- python h46_independent_witness_replay.py . --out h46-output
- sha256sum -c H46_MANIFEST.sha256 (against the pristine extracted archive, before scripts rewrite result files)

The accompanying repository-side verifier, h46_verify_exact_input_collision.py, rechecks every witness against genuine trained donor/receiver networks and retained raw observations. **The complete binary custody is in the owner archive, not in GitHub**; exact-HEAD H46 CI has not been claimed.

## Consequence, next smallest change, and claim boundary

**Stop fitting the mathematically contradictory universal synthetic-alpha oracle target.** Its 65/257-grid failures do not support enlarging the receiver or a claim of general memory transplantation. Keep H45's finite-grid PASS and off-grid FAIL unchanged.

Continue using the *actual scientific gate*: two genuinely trained material-law endpoint donors, receiver inference with no oracle features or law/family/source identifiers, matched geometry/goals/legal masks and weights, both-direction divergent correctness, invariant no-change, independent genuine donor-source tests, nontrivial negative controls, and genuine unseen non-permutational fee/wet-count distributions. Any donor-side calibration must be supervised only on its own material behavior, not passed as a receiver answer or physical-law label; its effectiveness must be independently checked before a new preregistered cohort. Do not change physical endpoint gate thresholds after seeing this result.

Synthetic intermediate donor features **are not independently trained intermediate-law donors**. H46 does not invalidate H40's restricted prospective endpoint PASS, nor does it establish spatial generalization, physical-law universality or CyberSand game functionality.

**Repository ownership:** existing neurounner PR #253 only; Draft/HOLD, unmerged, never Ready. No new branch/PR/issue, no paid GPU/API, no changes to CyberSand or DRIV3R.
