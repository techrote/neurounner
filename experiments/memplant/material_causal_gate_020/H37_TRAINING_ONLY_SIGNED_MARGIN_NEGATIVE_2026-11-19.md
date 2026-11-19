# DG020 H37 — training-only bounded output-head calibration: FAIL (19 November 2026)

**Disposition: H37 TRAINING-ONLY MECHANISM FAIL.** The latest preregistered unseen-family × genuine heldout-donor scientific verdict remains **H36 FAIL, 29/30** (two missed mandatory divergent flips for seed 26011641). H37 is a diagnostic repair attempt using only **already exposed** H27 training worlds, H36 formerly prospective worlds, and H21/H34/H35/H36 genuine donor checkpoints. H36 worlds and donors have become TRAINING examples and cannot be reclassified as independent acceptance. **PR #253 remains Draft/HOLD, unmerged.**

## Exact method

- Three original 209-parameter, genuinely trained H36 dry-anchored receivers (seeds `26011639`, `26011640`, `26011641`); all **200 hidden parameters frozen**, only 8 output-head weights eligible for change, constrained to L1 distance **≤0.002** from each original H36 head. No model-size, topology, material-law, or world-family expansion.
- **16 individual genuine donor profiles**: four each from H21, H34, H35 and H36. Retain each trained replica separately, never average away same-law source variation. Both inherited laws remain wet cost 0.35 / 2.30, dry cost 1.0.
- Per seed: original **2,560 exposed H27 compact-margin worlds** plus **1,600 now-exposed H36 worlds** (spiral_spoke and ladder_weave). **133,120 both-law signed route-margin constraints per receiver** (4,160 worlds ×16 donor pairs ×2 laws). Symmetric max-min linear programming with the same source-blind dry-reference anchoring. Geometry, goals, fees, legal masks, receiver hidden weights identical within pairs; only the genuine donor-derived material coordinate changes.
- **No** law/family/source identifiers, oracle distance, path labels or answer features enter receiver forward inference. The true route action is used solely as a training constraint and offline scoring. CPU PyTorch 2.10.0, NumPy 2.3.5 and SciPy 1.17.0 HiGHS. No paid compute.

## Training-only outcomes

| Seed | L1≤0.002 LP | H27 exposed both-correct | H36 exposed both-correct | Source changes | Wrong invariant flips |
|---|---|---:|---:|---:|---:|
| 26011639 | FEASIBLE; worst signed margin +0.000392616 | 2560/2560 | 1600/1600 | 0 | 0 |
| 26011640 | FEASIBLE; worst signed margin +0.000506215 | 2560/2560 | 1600/1600 | 0 | 0 |
| **26011641** | **INFEASIBLE** | 2560/2560 (original H36) | **1598/1600 (original H36)** | 0 | 0 |

For the two feasible fitted seeds, source-independent genuine-PyTorch replay of all **3,200 exposed H36 world pairs** confirmed the counts and unchanged hidden tensors. On each seed's 640 divergent pairs, zero-donor and swapped-law controls achieved **0** both-correct; independently shuffled donor controls produced **166/640** and **152/640** both-correct. These are **exposed-world training diagnostics only**, not scientific PASSes. No H37 checkpoint was produced for 26011641 and no fresh prospective cohort was opened.

### Quantitative feasibility finding

For 26011641, the max-min LP was infeasible under L1 caps **0.002, 0.004, 0.01 and 0.05**. The independent minimum-L1 feasibility calculation required **0.364001189** even to make all 133,120 signed constraints nonnegative, **0.372026875** for margin +0.000001, or **0.524518709** for margin +0.000020. This is **≥182×** the intended 0.002 trust region before requiring a positive margin. A large L1=0.5 diagnostic solution has positive ~0.000016945 margin, but was **not promoted**: unrestricted head displacement would depart substantially from the bounded calibration mechanism and offers no unseen-family evidence.

The retained original H36 failure cases are 26011641 / spiral_spoke / expensive_edge_divergent, world indices **58** (`c6bcaf20f0513f6eb3a4de56d3397152ff5ebed5f063cb11595c19614b96efa3`) and **94** (`d749bd1bac28e65519e1d27f24ebbb6644318c407fb6d38d614c016aabec9177`). Both require an expensive-law action switch, and all original H36 trained donors agreed on the wrong action. Inference on original H36 checkpoints remains unchanged.

## Independent review, custody and next gate

Independently regenerated and verified all **4,800/4,800 original H36 observation SHA-256 hashes**, replayed all 4,800 original genuine-neural predictions with **0 action mismatches, 0.0 logit drift**, and reproduced the two negative hashes and the H36 FAIL. Both new 209-parameter output-head-only checkpoint states retain **bit-identical 200 hidden parameters**. H37 files are sealed in portable handoff archive `DG020-H37-trainingonly-signed-margin-INFEASIBLE-20261119.zip` (SHA-256 `86d39d08fd401b11dfaa04713a14b60c43a1e87db96f64683a908eec913614eb`): original H36 28MB source/checkpoint/receipt capsule, H37 exact scripts, 2 new checkpoints, logs, LP feasibility results, independent audit and checksum manifest (13/13 entries verified). Full binary GitHub custody and H37-specific exact-HEAD CI remain outstanding. Additional numerical head custody is committed separately.

**Smallest next change:** a training-only adaptation of existing hidden-layer activation knees jointly with both-law signed action-margin loss, first on the exposed 26011641 negatives, with the SAME 209-parameter architecture, 16 individual genuine donor profiles, invariant selectivity and zero/swapped/shuffled controls. Do not open any new prospective world-family × donor-source test until this smaller mechanism passes training-only qualification. Existing 29/30 scientific FAIL remains the governing result. No CyberSand/DRIV3R writes, branch/PR/issue proliferation, paid GPU/API, Ready or merge. 
