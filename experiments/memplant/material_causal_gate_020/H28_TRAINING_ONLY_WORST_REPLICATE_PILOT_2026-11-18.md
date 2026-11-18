# DG020 H28 — simultaneous genuine-donor worst-replicate calibration, training-only pilot (18 November 2026)

**Status: NOT A PROSPECTIVE EXPERIMENT; NO CAUSAL-GATE PASS CLAIM.** The latest preregistered scientific gate is H27 **FAIL 28/30**. This pilot was deliberately not taken to a new acceptance cohort: the mechanism remained unstable on *previously designated training worlds*. No H27 acceptance-world observations, labels or receipt outcomes were used for pilot training or model selection.

## Original evidence validation

The original self-contained H27 owner handoff archive was reverified byte-for-byte: SHA-256 `0f8edf6d4a5b8ed69ef09656f52e4920ab6b5887baf24e8aab606990d32810a6`; **115/115 manifest entries**, zero mismatches. A fresh local CPU run of the retained **independent genuine-donor / frozen-receiver audit** reproduced the **28/30 H27 FAIL** and both individual-replicate failure cells (six changes at `26011639/outer_in/wet_rich_invariant`, five plus one improper law flip at `26011641/stutter_jump/wet_rich_invariant`). The 34 MB original archive remains separately preserved; GitHub custody of its exact H27 bytes and H27-specific CI are still outstanding.

## New training-only mechanism diagnostic

All 15,360 **existing H27 training samples** (5,120 per receiver seed, including 2,560 near-tie training transformations per seed) were reconstructed. Recomputed observation hashes matched their original H27 training-hash lists exactly for every seed. Four individually trained authentic H21R donor source/replicate pairs were loaded. There was **no** use of H27 prospective holdouts in this diagnostic.

Their material forecasts differ measurably even within the same inherited law:

| Material forecast | Minimum across four genuine donor source/replicates | Maximum |
|---|---:|---:|
| Cheap wet material | 0.3493163 | 0.3509640 |
| Expensive wet material | 2.3012445 | 2.3034949 |
| Dry material | 1.0010209 | 1.0014228 |

The resulting **p99 route-advantage span between authentic donor-replicate choices** on the existing near-tie training worlds was **0.00169–0.00197** across the three frozen H27 receivers. This is comparable to the lower edge of the H27 oracle-margin range (0.001–0.006); the resulting robustness problem is mechanistically credible. These are training-only measurements, not fresh generalization evidence.

## Bounded H28 pilot

An experimental *same 209-parameter receiver* was warm-started from each original H27 frozen checkpoint and retrained using the **exact same H27 training observations** and the **same genuine H21R donor variants**. No new mechanics, law IDs, route geometry, world-family expansion, API/GPU charges or game changes.

The single conceptual intervention was replacing H27's *alternating one donor replicate per optimizer step* with *simultaneous four-replicate per-sample worst-case absolute route-margin calibration*. Paired-law consistency, tile-cost supervision, and equal-wet regularization were retained. 1,600 local-CPU AdamW steps per seed; no acceptance cohort generated or examined.

| Receiver seed | H27 training-only near-tie donor-variant action changes /2,560 | H28 pilot /2,560 | H27→H28 wet-rich invariant changes /256 |
|---|---:|---:|---:|
| 26011639 | 62 | 56 | 8→4 |
| 26011640 | 34 | 32 | 5→5 |
| 26011641 | 40 | 34 | 15→15 |
| **Total** | **136** | **122** | **28→24** |

All three averages showed **zero incorrect primary actions on these reused training near-ties**, but this is *not* a robust pass: the **122** individual-replicate changes persist. Some category-specific behavior worsened, including expensive-edge divergent donor changes of **6→10** (seed 26011639) and **4→7** (seed 26011640). The empirical tradeoff fails to justify promoting this candidate to prospective testing. The H27 scientific FAIL remains binding; H28's scientific gate is **NOT TESTED**, not PASS or a preregistered FAIL.

### Reproducibility

Local training-only pilot source SHA-256: `d49a12a7c24fc5ca0d9e91ed825b13f7c0ec106d90868722c7904825a6abfbda`.

Original **pilot** (not accepted scientific) receiver checkpoint SHA-256:
- 26011639: `b04737782f3612c12f48873fba29ca851d5ec6df3d8dc4e72c8a46d1e019faed`
- 26011640: `068d3f481083aaa4af31f7e21fa275eefff08f87a708235ba994a3bf5a776617`
- 26011641: `2928c2b0b1dfa073147e57572e5cbe6b31fd8aa90fdba14974166cc5459ea98c`

The self-contained **training-only** owner-handoff package `DG020-H28-training-only-worst-replicate-PILOT-20261118.zip` is **22,363 bytes**, SHA-256 `d471fbf0e92e748534b43350615fd23a5e17a63460636e002c75367f6d7f6034`. It contains exact script, checkpoint bytes, raw training-only comparison and donor-noise reports, independent H27 replay log and a hash manifest. It **does not** contain an H28 prospective test and must not be treated as a locked future acceptance source. H28 pilot files are not yet in durable Git custody.

## Smallest next change

On **training-only** data, add an explicit **within-law, within-world donor-replicate tile-output consistency loss** to penalize changes from authentic donor source/replicate substitutions while retaining the between-law response. This is more direct than worst-replicate absolute-margin calibration alone. Measure its effect against *identical* training-only near-tie pairs across all four donors, including both divergent categories and wet-rich invariant. Only after the training-only control demonstrates clear improvement should the exact source, trained checkpoint hashes, unseen family grammar, hard per-cell sham/invariant/divergent thresholds and zero/shuffled controls be preregistered **before** any new prospective holdout. Do not tune to H27 exposed acceptance worlds.

**Governance:** Only the existing `techrote/neurounner` Draft/HOLD PR #253 branch changed. Never merge or Ready. No new PRs/issues/branches, paid API/GPU or CyberSand/DRIV3R writes.