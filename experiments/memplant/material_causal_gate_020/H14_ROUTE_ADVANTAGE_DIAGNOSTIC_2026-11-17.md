# DG020 H14 — independent exposed-cohort route-advantage diagnosis (17 Nov 2026)

**Scientific disposition remains H14 FAIL: 20/30 preregistered acceptance cells.** This audit does **not** introduce a new preregistration, acceptance cohort, trained model, or claim of scientific improvement. It reprocesses the 4,800 archived H14 JSONL matched-pair receipts. All original negative results remain authoritative.

## Scope and provenance

- Original H14 frozen source SHA-256: `3285f95a91e1e94d0020622d3fcea7c5362cf47b4a615887a0ea19ffdc44de40`.
- Original H14 source was preregistered in [PR #253](https://github.com/techrote/neurounner/pull/253#issuecomment-6951995262) before evaluating the fresh cohorts.
- Source-frozen local H14 report and checkpoint replay: [H14 findings](./H14_FINDINGS_2026-11-17.md).
- This diagnostic uses those **exposed** worlds, frozen predicted logits, real trained donor forecasts, and per-law oracle route-cost receipts. It does not use the holdouts for training.
- H14 remains **local CPU PyTorch 2.10 only**; pinned GitHub Actions qualification has not been established.

## Per-law failure decomposition

The signed route advantage is `route1_cost - route0_cost`; positive values choose route 0. For neural receiver logits, use `route0_logit - route1_logit`. Recomputed errors are prediction minus oracle signed advantage, evaluated on all 960 examples per bucket.

| Bucket (960 pairs each) | Wrong law 0 | Wrong law 1 | Law 1 receiver advantage error SD | Law 1 donor-cost advantage error SD |
|---|---:|---:|---:|---:|
| Cheap-edge divergent | 9 | 0 | 0.02003 | 0.02503 |
| Expensive-edge divergent | 0 | **76** | **0.02952** | **0.03744** |
| Wet-rich invariant | 0 | **141** | **0.02932** | **0.03812** |
| Wet-poor invariant | 0 | 0 | 0.02065 | 0.02640 |
| Equal-wet invariant | 0 | 0 | 0.00342 | effectively zero |

These are failure counts **per law**, not paired acceptances. The original H14 full-gate denominator and verdict remain 4,800 matched pairs and 20/30 cells PASS.

**Wet-rich invariant:** all 141 erroneous expensive-law neural receiver actions coincide with wrong-sign expensive-law route advantage in the summed, genuinely learned, primary fused donor forecasts. This particular failure is upstream in the donor-law estimate, not merely the receiver's final ranking.

**Expensive-edge divergent:** 76 expensive-law neural receiver action errors: 41 also have wrong-sign donor-cost route advantage; the other **35 occur despite a correct-sign donor-cost comparator**. Both donor estimation and learned receiver ranking contribute here.

**Same-law source identity:** original H14 retained 264/4,800 changed action pairs on substitution with independently trained third-seed same-law donors; the expensive-law wet-rich bucket alone recorded 163 changes.

## Exploratory two-source logit average — *not* acceptance

Averaging archived primary-source and foreign-source logits **after inference** improves both-correct totals from original H14's 4,574/4,800 to **4,606/4,800**. Basic correctness/required-flip tests would pass 24/30 cells versus original H14's 20/30, but this exploratory result does **not** evaluate all preregistered sham, source independence, control and donor-error gates. It is post-hoc on the exposed H14 worlds and must **never** be reported as an H15 PASS or formal scientific improvement.

## Mechanistic conclusion and smallest next experiment

The main residual failure is expensive-law close-margin donor-advantage noise (receiver error SD ≈0.029 against oracle margins 0.015–0.070) plus a smaller independent receiver-ranking component. Training the same-size receiver alone cannot fix wrong-signed donor-cost signals unless it gains a better continuous correction of donor-law estimates.

Next candidate: use training-only cross-source continuous calibration of the scalar material advantage (no law labels, donor identity, family IDs, oracle route costs, or correct actions at inference), and the same genuine neural donors / 209-parameter receiver / two laws / 2×12 topology. Evaluate new unseen fee families with fresh seeds and hashes only after pre-registering both-direction, invariant, cross-source sham, pre-inference shuffled/zero/source-blind, donor MSE and per-cell thresholds. Compare against the deterministic **neural-donor** cost comparator. No broadened model, world complexity or game modification.

**Research custody:** this is additional negative-mechanism explanation on the existing Draft/HOLD PR #253, not a new PR, branch or issue. The original H13 invalidation and H5–H7/H11/H12/H14 FAILs remain available. No CyberSand/DRIV3R writes or paid compute.
