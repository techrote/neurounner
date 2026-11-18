# DG020 H34 — genuine withheld-donor-source screen, exposed training worlds (18 November 2026)

**Disposition: H34 training-only donor-source screen PASS 30/30 nonpooled cells; scientific causal-selectivity gate NOT PASSED.** The latest demanding prospective unseen-world result remains **H27 FAIL 28/30**. PR #253 remains **Draft/HOLD**, not Ready and never merge without owner acceptance.

## Locked source, preregistration and scope

[H34 preregistration, before donor training or inference](https://github.com/techrote/neurounner/pull/253#issuecomment-6994065981), source SHA-256 **`afbfb713777a864ec030663e65aeeebffa0d117ad28fc6ee90e387c8d256e030`** (`h34_withheld_donor_probe.py`, 12,734 bytes, compiled but not executed before registration).

Scientific question: do the completely frozen, genuinely trained H33b 209-parameter receivers maintain their learned two-law rule when provided with **independently trained genuine donor source/replicate checkpoints never used during H32/H33/H33b fitting**? Test entirely on H27's **original exposed training-world placements** (2 original families, 5 original buckets, 256 pairs/cell, 3 receivers = 7,680 matched pairs). This is deliberately not a new prospective-world cohort and not an unseen-family test.

Two new donor sources `29002131` and `29002137`, two independently initialized replicas per source/law, eight original 4→24→24→1 donor networks, 8,192 independent synthetic tile-cost supervision samples each, 1,000 CPU AdamW updates (batch 256, LR .002), 2,048 separate validation tiles. Wet costs `0.35/2.30`, dry cost `1.0`. All 8 donor checkpoints were genuinely optimized and independently retrained from their locked seeds, with **8/8 byte-identical original PyTorch checkpoint SHA-256 values** in the clean-directory replay. Largest new held-out donor MSE **0.0007572045**, well below the locked `0.02` bound. New donor-source weights were not selected using world outcomes, and no receiver weight was retrained.

H33b receiver SHA-256 (seeds `26011639`, `26011640`, `26011641`): `dfe5af62e09924dd6556d80dd92c3554f011e6462acb1438b4533d5b01bbc610`, `001114244f2b90adf310c11193df2a9a04e19228db9dcd6108da050088d7e801`, `08e134b68072bac8ce0b551601313a24e3807f88cdfca7cf3c848fac32e7dd63`.

All **15,360 exact original H27 training-world hashes** were verified. Geometry, observations, goals, fees, nuisance textures, legal masks and receiver parameters are identical between law arms. Only the *genuine law-conditioned donor neural material forecast* is swapped. The frozen receiver is never given law/family/source IDs, oracle labels, route costs or answer features. Original four H21R trained donor variants are kept as fitted-source controls.

## Results — all 30 locked training-only cells pass

| Criterion | Authentic withheld-source result |
| --- | ---: |
| Independent nonpooled cells | **30/30 PASS** |
| Both-law correct across **all four** new genuine donor profiles | **7,679/7,680** |
| Divergent both-correct required flips, all new sources | **3,071/3,072** |
| Invariant pairs both correct | **4,608/4,608** |
| Incorrect invariant-law action flips | **0** |
| Individual new same-law genuine donor source/replicate action changes | **1/7,680** |
| Previously-fitted → new same-law donor action changes | **1/7,680** |
| Shuffled wrong-law divergent both-correct | **776/3,072** |
| Swapped-law divergent both-correct | **0/3,072** |
| Zero-donor divergent both-correct | **0/3,072** |
| Illegal actions | **0** |
| Independent neural checkpoint replay | **PASS, 7,680/7,680 raw decisions and 30/30 cells** |

Thresholds were committed to the PR preregistration comment: per divergent cell both-correct ≥95%, required flips ≥98%; per invariant cell both-correct ≥98%, wrong flips ≤1%; per-law ≥98%; individual same-law sham and old-to-new same-law changes ≤1%; shuffled wrong-law divergent both-correct ≤60%; donor validation MSE ≤.02; no illegal actions. All cells independently passed; no post-hoc threshold adjustment.

## Preserved negative: one new genuine donor fails near a decision boundary

**Receiver seed `26011641`**, placement `cyclic_run`, bucket `expensive_edge_divergent`, compact-world index **1783**. Exact world SHA-256:

`dcaeda0280d5882b07cd4253883e6ad1245fa98407355a2392f976c1b037b2ea`

Oracle actions `[1,0]`; the new `29002131-rep1` genuine donor yields `[1,1]`, missing the necessary expensive-law action switch. Three other new sources, and all four previously fitted donor sources, choose `[1,0]` correctly. The expensive-law oracle route0−route1 cost is approximately **−0.000583**, but the frozen receiver's corresponding route0−route1 *score* margins for new donors range from **−9.54e−6 to +1.34e−5**. This is a genuine fragile near-tie donor/receiver interaction, not evidence of error-free unseen-donor generalization.

The smallest next mechanism change, if needed, is to increase the limiting seed's correctly signed route-margin robustness to **previously unseen donor-replicate perturbations** without altering the 209-parameter architecture, the two laws or topology, and without trading off invariant action selectivity. Do this using training-only comparisons; pre-register a genuinely untouched unseen-world cohort before subsequent claims.

## Reproducibility, exact evidence, and qualification boundary

Independent replayer **did not import H34 training/evaluation code**. It reloaded genuine trained donors and frozen receiver tensors, independently reconstructed quadrature features and every neural score, verified all receipt-world hashes against the original H27 training hashes, and replayed all 30 cell metrics. Maximum absolute logit difference against stored receipts **3.814697265625e−6**; **one** incorrect all-new-donor pair reproduced exactly. A clean-directory replay verified all **36/36** manifest entries. Clean-room rerunning all eight donor training jobs reproduced **8/8 exact checkpoint SHA-256s**, followed by deterministic evaluation reproducing raw receipt SHA-256 `5331f0ff43db5653d03466832d8bae694e14766595c39c5b19f0da51ec2e1e31`.

**Portable complete evidence:** `DG020-H34-withheld-genuine-donor-trainingworld-PASS-20261118.zip`, 8,211,119 bytes, SHA-256 **`ea7e6fcb31a787d7497585278bf22ad7a5780fe4febb615841f55e1b9a16afd7`**. Contains locked H34 source, independent replayer, original H27/H23 generation dependencies, 8 previously trained H21R and 8 newly trained H34 genuine donor checkpoints, all 3 exact frozen H33b receiver checkpoints, 7,680 full raw original observations, actions and logits, original training hashes, training/evaluation logs, cell reports, audit, and 36-file integrity manifest. The ZIP is retained in the owner handoff. **Full archive and H34 exact-source code are not yet in GitHub custody and H34-specific exact-HEAD Actions replay is not qualified.** Do not claim otherwise.

**Scientific inference:** H34 establishes bounded withheld-donor-source stability of an already-trained rule on *historically exposed* training-world placements, with one retained negative. It does **not** override H27 FAIL or establish causality across genuinely unseen world families, donor-weight surgery, arbitrary material-law transplantation, or any CyberSand game integration. CyberSand/DRIV3R remain read-only; no paid compute used.
