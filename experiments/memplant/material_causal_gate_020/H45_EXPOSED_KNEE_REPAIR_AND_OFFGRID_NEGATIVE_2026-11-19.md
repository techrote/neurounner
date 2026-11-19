# DREAMGRAFT DG020 — H45 exposed-grid knee repair; off-grid robustness FAIL (19 Nov 2026)

**Classification:** Training-only exposed-world feasibility PASS on the preregistered 65-position grid; **post-hoc off-grid robustness FAIL**. **NO new prospective scientific result.** H40's restricted 30/30 prospective PASS remains a historical narrow result subject to H41–H44 methodological limitations; no continuous-law, new-family or game capability is established.

H45 was [preregistered in this existing PR before execution](https://github.com/techrote/neurounner/pull/253#issuecomment-7004400446). Ownership and safety: PR #253 only, Draft/HOLD; do not merge/Ready. No new issues, branches or PRs; CPU-only, no paid compute, and CyberSand/DRIV3R untouched.

## Input identities and allowed mechanism

* Original **4,800 H40 exposed matched world pairs** (no new holdouts); raw receipt SHA-256 `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`.
* Eight genuine trained donor checkpoints (four source/replicate profiles, two authentic endpoint laws); three genuine trained **209-parameter** receiver checkpoints.
* One or two first-hidden-layer ReLU bias indices changed, along with the original eight output-head weights. **All other hidden/input/output weight matrices and output bias preserved bit-exactly.** No architecture, law, maze or family expansion.
* Analytic oracle actions provide external training constraints, never inference features. Inference still receives only original observation and genuine-donor-conditioned tile-cost channel; no family ID, law label, donor source ID, answer or oracle route cost.
* First one-knee sweep gave feasible refits for seeds `26011639` and `26011640`. An additional bounded two-knee sweep was needed for seed `26011641`. The fitting LP enforced **hard signed action constraints on every exposed world/donor/65-position alpha**; no candidate was allowed to pass by average-loss tradeoffs.

## Locked 65-position exposed training screen — PASS

| Screen | Result |
|---|---:|
| All 3 receivers × 4 donor profiles × 65 alphas × 1,600 worlds per receiver | **1,248,000/1,248,000 correct** |
| Genuine donor-law endpoints | **38,400/38,400 correct** |
| Invariant-world interpolated actions | **748,800/748,800 correct** |
| Inappropriate invariant transitions | **0** |
| Both-arm correct divergent endpoints | **7,680/7,680** |
| Same-law genuine donor-source sham action changes | **0** |
| Illegal actions | **0** |

Independent verification executed the **complete PyTorch donor and receiver networks** rather than using the LP feature reconstruction. All float32 route logits and actions were retained, and only permitted tensor edits were observed.

Selected original checkpoint → new candidate SHA-256:

| Seed | ReLU bias indices | Candidate SHA-256 |
|---|---|---|
| `26011639` | `[1]` | `0647f98131e30026e5f8211570cda73adcf6172a8e6238430f7b73efbac5df1f` |
| `26011640` | `[1]` | `7eaae96ba2b8e73bbfdb144f1d86eb78d101d887463b5002e6a2291d819458a6` |
| `26011641` | `[1,2]` | `acec1cc9ac7ac221131e60b515d2c3025f20cd9b95afd134c27b48d23f63a542` |

Exact float32 numerical tensor changes are preserved in `H45_RECEIVER_FLOAT32_BIT_PATCHES.json`; the original checkpoint bytes and all newly serialized checkpoints are in the owner's self-contained reproducibility capsule. The second receiver's **minimum absolute direct float32 score margin was 0** on the 65 grid (correct action-0 tie), making these fits numerically fragile even when classifications are correct.

Actual neural controls: real zero tile-cost tensor, neutral `1 + fee` cost, and reversed donor-law arms each achieved **0/7,680** divergent both-correct, as mathematically expected (not a strong shortcut-exclusion control). Independently pre-inference shuffled arm assignments achieved **1,895/7,680 (24.67%)**, compatible with random-arm 25%. Same-law genuine-source sham changes were 0.

## Post-hoc off-grid falsification — FAIL

After fitting to the original 65 values `alpha=k/64`, run the frozen H45 receivers on the previously **unfitted** 192 values `alpha=n/256` with `n % 4 != 0`, reusing exactly the same exposed H40 worlds and donor endpoints. These are **synthetic donor-feature interpolations**, **not** actual donors trained at intermediate laws.

| Seed | Off-grid errors / 1,228,800 | Invariant errors | Divergent errors |
|---|---:|---:|---:|
| `26011639` | **757** | 0 | 757 |
| `26011640` | **2,573** | 0 | 2,573 |
| `26011641` | **1,842** | 0 | 1,842 |
| **Total** | **5,172 / 3,686,400** | **0** | **5,172** |

Thus **causal selectivity at known 65 points is not a learned continuous material-law rule**. An additional post-hoc 257-point hard LP, holding newly moved knees fixed and permitting only the original eight output-head weights to change, reported **infeasible for all three seeds** (positive convex numerical-witness residuals ~1e−16). These are numerical diagnostics, *not* certified exact rational proofs and not impossibility results for the entire 209-parameter architecture.

## Reproduction and claim boundary

CPU: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, SciPy 1.17.0. Original executable sources, all genuine checkpoints, all 4,800 raw receipts, all 1,248,000 float32 neural logit margins/actions, unsuccessful variants, full negative controls, dense-grid reports and SHA-256 manifest are retained in the H45 reproducibility capsule delivered to the owner. Manifest was checked on clean extraction, and clean-directory independent neural replay reproduced **0 finite-grid wrong, 0 invariant flips and 5,172 off-grid wrong**.

Smallest next mechanism: **joint hard-constrained nonlinear knee placement and output-head calibration at divergent transition boundaries**, preserving genuine endpoints, invariant worlds and donor-source shams, with **held-out interpolation positions** and predeclared off-grid acceptance. Do not create a new unseen-family cohort until the exposed robustness screen is genuinely solved; do not broaden networks, physics laws, maze structure or general memory-transplant claims.

**Scientific disposition: H45 finite training-grid feasibility PASS / post-hoc off-grid robustness FAIL; NO prospective gate opened. PR #253 remains open Draft/HOLD, unmerged and not Ready.** Full binary GitHub custody, exact-head H45 CPU CI and external methodological review remain outstanding.
