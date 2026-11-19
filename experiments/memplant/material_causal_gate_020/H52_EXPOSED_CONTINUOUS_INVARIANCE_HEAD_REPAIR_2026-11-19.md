# DG020 H52 — constrained continuous-invariance receiver repair (2026-11-19)

**Disposition: exposed-only mechanism PASS; prospective scientific gate NOT RETESTED. Draft/HOLD.**

H51 found five exposed invariant worlds on which the genuine H45 trained receiver (seed `26011640`) selected a wrong action over narrow intermediate donor-cost intervals. H52 repairs those intervals by changing **only eight output-layer weights** of this same 209-parameter genuinely trained receiver. Every hidden weight, activation knee and output bias is frozen. No synthetic divergent-law labels were fitted. The other two H45 receivers are untouched. No new law, family, maze complexity, receiver scale or game integration.

## Provenance and method

- Authentic H40 receipt SHA-256: `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`.
- Original H45 checkpoint SHA-256: `7eaae96ba2b8e73bbfdb144f1d86eb78d101d887463b5002e6a2291d819458a6`.
- H52 exact candidate checkpoint SHA-256: `b1662812437ae4a83e16f0f345e2c8291acf48e8748776340d98f4be9a9fc9fc`.
- H40 worlds used: 1,600 **previously exposed** worlds for this receiver; 640 divergent, 960 invariant.
- Four independently trained H34 donor profiles, with genuine endpoint costs from the previously validated H47 donor probes; donor inference was **not independently rerun in H52**.
- Hard linear constraints: 12,800 genuine-donor endpoint actions and 101,442 ReLU-knot evaluations on the 960 invariant worlds, covering the entire min/max genuine donor-cost interval. The paired geometry, goals, legal masks and receiver weights are identical within each intervention pair.
- Since the hidden ReLUs are frozen, the route margin is piecewise affine in inherited material cost. A linear program (scipy HiGHS) minimizes the largest absolute change in the eight output-head weights while requiring positive signed margins at every genuine endpoint and every invariant knot.
- No oracle cost, family label, source ID, action answer or oracle distance is supplied to the receiver. Only the existing observation and donor-derived material-cost scalar are used.

## Numerical counterexample retained

The first candidate imposed a `1e-6` margin and was feasible in ideal arithmetic, but actual float32 neural replay produced **one genuine endpoint tie/misclassification**. This candidate is **FAIL**, retained as checkpoint SHA-256 `9fe81ab7529228f68e60c8aa334395294a43cada1f748d119e9c44cc3bb7bd57`.

The repaired candidate imposed a `1e-4` signed margin, feasible at maximum output-head weight adjustment `0.044657853119908`. The exact eight float32 replacement weights and source-identity/sha verification are recorded in `h52_rebuild_head_patch.py`. The H52 owner evidence archive retains both checkpoints, executed scripts, complete raw decisions and SHA-256 manifests.

## Results — actual independent float32 PyTorch receiver

| Measurement | Original H45 seed 26011640 | H52 |
| --- | ---: | ---: |
| Wrong genuine donor endpoint actions / 12,800 | 0 | **0** |
| Minimum signed genuine endpoint margin | 0.00000762939 | **0.00009918213** |
| Wrong invariant knot actions / 101,442 | 29 | **0** |
| Minimum signed invariant knot margin | -0.00020217896 | **+0.00009918213** |
| Distinct invariant worlds failing | 5 | **0** |
| Correct actions at 20 original H51 negative witnesses | 0 | **20** |
| Additional off-knot invariant stress (seed 2601195201) | not needed | **121,920/121,920** |
| Genuine both-direction correct / 2,560 divergent profile pairs | 2,560 | **2,560** |
| Both arms correct / 3,840 invariant profile pairs | 3,840 | **3,840** |

**Controls:** with identically seeded (26011952) actual float32 donor interventions on the H52 receiver, genuine divergent pairs were 2,560/2,560 both-correct; reversed-law pairs 0/2,560; literal all-zero donor tensor, constant-zero-wet-base, and neutral-unit same-cost shams 0/2,560 (mathematically inevitable, so per-arm correctness is retained); shuffled law assignment 646/2,560 (25.23%, expected random two-arm baseline near 25%). All 3,840 genuine invariant pairs were both-correct under genuine, swapped, actual literal-zero cost-tensor, wet-base-cost-zero, neutral-unit and shuffled conditions. Literal zero means every third-channel material cost is exactly zero, not c=0 plus the retained fee.

## Interpretation and next action

**Exposed-only continuous-invariance head-repair screen: PASS. Primary scientific gate: NOT RETESTED / no promotion.** All five H51 exposed failure worlds are fixed; same authentic receivers and donors are used. The interval certificate concerns an ideal piecewise-affine real-arithmetic model. Float32 inference at all 101,442 knots and additional off-knot stress passed with positive guard bands; every possible cross-CPU intermediate point is not formally certified.

H46 showed why trying to fit all synthetic **divergent** interpolation labels is non-identifiable; H52 avoids that impossible target. Historical H40 30/30 restricted prospective PASS and its H41 spatial-generalization limitations remain unchanged. A successful exposed-only LP is not evidence of unseen-family generalization or general memory transplant.

Next smallest justified move: independently re-probe authentic trained donors; complete binary GitHub custody and pinned CPU action-level replay; preregister an untouched, structurally unseen family cohort with strict **both-direction action correctness**, **no invariant-law flips**, matched-pair identity, informative sham/shuffled controls and predeclared generalization thresholds, before opening that cohort.

**Governance:** PR #253 only; open Draft/HOLD, never Ready/merged. Neurounner is the only writable repository. CyberSand and DRIV3R were not modified. No paid API or GPU expenditure, new issues, branches or PRs.
