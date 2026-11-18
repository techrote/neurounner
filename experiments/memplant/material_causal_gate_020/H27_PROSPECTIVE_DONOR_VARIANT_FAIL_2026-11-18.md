# DREAMGRAFT DG020 — H27 prospective donor-replicate near-tie experiment (18 Nov 2026)

**SCIENTIFIC VERDICT: FAIL — 28/30 independently preregistered cells. Do not reinterpret as PASS.**

This is a new prospective negative, not the reused H26 post-hoc cohort. [Preregistration before H27 acceptance-world generation and neural inference](https://github.com/techrote/neurounner/pull/253#issuecomment-6984028877). Source SHA-256 `01bc2a76de1de9485f5d5847c2770ee380bae808d432560f76e7122e49161279` and original trained receiver checkpoint hashes were registered before inference. Source was retained in owner handoff and is **not yet Git-custodied**; no H27 exact-head Actions replay is claimed.

## Fixed experiment

Three original genuinely trained 209-parameter receivers, warmed from H26 exact original checkpoints; genuine H21R donors, unchanged two material laws (wet cost 0.35/2.30), fixed 2×12 route topology and two legal actions. Training-only near-boundary examples with margins 0.0005–0.012, 2,000 CPU AdamW updates per seed, with *four individually trained donor source/replicate pairs* sampled during training to increase robustness and per-law absolute-route-margin loss. Within a test pair, observation, route geometry, wet pattern, fees, textures, goals, legal mask and receiver weights are **identical**; only the inherited wet-cost donor coordinate changes, dry donor coordinate fixed. No source IDs, world-family IDs, law labels, oracle costs or answer labels at receiver inference.

Fresh `outer_in` / `stutter_jump` wet-placement and beta-fee grammars, closest oracle route margin **0.001–0.006**, each seed × family × five bucket 160 worlds = **30 nonpooled cells, 4,800 matched counterfactual pairs**. Zero collision with 85,439 previously exposed observation hashes (including historical post-hoc stresses) and 15,360 current training observations. These new families are permutation/fee-profile families **not physically different mechanics**.

## Preregistered results

| Check | Result |
|---|---:|
| Independently gated cells | **28/30 PASS — overall FAIL** |
| Divergent both-direction correct with required flip | 1,920/1,920 |
| Invariant both-law correct | 2,879/2,880 |
| Inappropriate invariant action flips | 1/2,880 |
| Averaged foreign same-law authentic donor changes | 0/4,800 |
| Individual authentic donor source/replicate changes | **11/4,800** |
| Shuffled-donor divergent both-correct | 460/1,920 (low-power sanity check) |
| Worst equal-wet p99 law-induced advantage drift | 0.000006676 (limit 0.003) |
| Genuine averaged donor maximum squared error | 0.000006737 (limit 0.020) |
| Illegal actions | 0 |
| Frozen genuine-checkpoint independent action/logit disagreement | **0.0** |

**Failure 1 — seed 26011639 × outer_in × wet_rich_invariant:** 160/160 primary pairs correct and no incorrect primary flips, but **6/160 individual same-law donor-replicate action changes**, violating preregistered **≤1/160**. All six caused by authentic `29002117-rep1` features. Averaged-source sham was insensitive.

**Failure 2 — seed 26011641 × stutter_jump × wet_rich_invariant:** 159/160 primary both-correct, **one inappropriate invariant flip**, and **5/160 individual genuine donor-replicate action changes** (limit ≤1). Four mismatches from `29002117-rep1`, one from `29002117-rep0`, one from `29002119-rep0` with overlap in worlds. No pooled rescue.

Independent local code recreated the 5×5 quadrature using the eight original trained donor checkpoints, replayed all three original frozen receivers across 4,800 raw observations and verified every oracle, action, primary/foreign/individual donor-variant logit, zero/shuffle control, cell gate and SHA, **max logit drift 0**. This is an independent implementation replay, not new acceptance.

## Exact custody and limits

Original receiver checkpoint SHA-256:
- seed 26011639: `072a7495ae3064fbf74595acbbde0c3568526da92902e0795ae6c93edfbdcf74`
- seed 26011640: `ecb77d1192361b73344bbc331cc57f3db9d9c9b4feac4d41eeb0f7c977a9514b`
- seed 26011641: `7a23c381ba27384c025fb150a7c08a7bb788e7731730cc0aa80526b82418951a`

Raw receipt SHA-256: seed 39 `854c9229a5eb79b0bb3e65b06b7e865d28b4c60257079c7bc88bb753e512a917`; seed 40 `193e501e6e3f2813470487416c4d0f73cc5561915487c92a0d10168cadfc91c3`; seed 41 `e76c6272963be2fde68a597cd973c1348bc59c89d018bbe49173440d3d24e34a`.

A 34,051,276-byte **self-contained owner-handoff archive**, with exact source, 3 original checkpoints, donor checkpoints, 4,800 raw receipts, full independent audit, training hashes and logs, older H22–H26 evidence and negative outcomes, has SHA-256 `0f8edf6d4a5b8ed69ef09656f52e4920ab6b5887baf24e8aab606990d32810a6`: `DG020-H27-prospective-individual-donor-neartie-FAIL-20261118.zip`. The archive was tested with ZIP CRC and includes a per-file manifest. **Full archive and H27-specific CI are not yet committed to Git / exact-head GitHub Actions**, so independent cross-platform qualification and durable evidence custody remain open.

## Interpretation and smallest next mechanism

The stronger authentic individual-replicate sham exposes law-invariant action sensitivity hidden by earlier nearly identical averaged donor sources. H27 preserved 100% required divergent flips but **failed near-boundary invariant/donor-replicate selectivity**. This does not prove general physical-law inheritance, memory transplant, or CyberSand game competence.

Next, first measure the authentic donor-replicate material-cost spread against the 0.001–0.006 oracle gaps on **training-only** examples; then consider a **single same-sized simultaneous worst-replicate per-law absolute-margin calibration** rather than alternating one donor replicate per training step. Freeze and preregister genuinely untouched worlds before acceptance. Never tune against exposed H27 cases. Keep sham-control discriminative power central and preserve all negative prior H23/H24/H25/H26 findings.

**Governance:** existing `techrote/neurounner` PR #253 only; open Draft/HOLD, never merge or Ready. No new branches/issues/PRs or paid GPU/API; CyberSand and DRIV3R strictly read-only.
