# RETURN-GEN-001 — variable-depth recovery-capacity ablation

**16 November 2026 — development-only, model-side research; Draft/HOLD.**

## Question / controls

The preceding fixed six-cell corridor experiment gave an engineered capacity/depth
comparator a perfect synthetic recovery-fork score. Does that transfer to new
corridor lengths and return directions if the network sees only *raw* host counters?

Four **real 30-parameter linear-softmax NumPy/Adam controllers** were trained
from identical per-seed initialization, TRAIN examples, minibatch draws and
36 → 72 → 128 curricula (30 updates per stage, **90 optimizer updates per
checkpoint**). Two independent initialization seeds, 596 and 597; **eight
checkpoint binaries remain out of Git history**.

* `blind`: neither host recovery budget nor explicit path depth.
* `budget_only`: action-conditioned remaining recovery counter.
* `raw_counters`: action-conditioned remaining budget and path depth as separate numeric observations.
* `engineered`: both raw counters plus explicit `remaining >= path_depth`
  feasibility flags. This deliberately provides the teacher's decision rule
  as an input; its success measures *representation*, **not acquired planning**.

TRAIN: 1000-series worlds, east/north tunnels of depths 5/7/9 with caps
2/6/10/12. DEVELOPMENT: disjoint 3000-series worlds, west/south tunnels of
depths 6/8/10, with paired caps (6:5/9; 8:5/9; 10:9/12). All
corrections, forced legal error moves, painted material and history follow
the unchanged source-faithful synthetic `World.menu/apply` contracts.
At the fork, both budget counterfactual states have identical terrain,
state digest and three host-legal choices: BACKTRACK, RELOCATE_START, STOP.
These are synthetic research overlays, not native gameplay maps.

## Source-replayed original results

**Two seeds, 192 original DEVELOPMENT episodes, 26,489 sequential
source-replayed decisions, 210 verified SHA-256 artifact entries, eight restored
checkpoints.** 128- and 160-decision rollouts are repeated on the same
two material contexts, three depths and budget pairs. They are **not**
192 independent world topologies.

| Arm | First-fork teacher choice | Early STOP | Mean ink, seeds 596 / 597 | Mean neural decision ms, seeds 596 / 597 |
|---|---:|---:|---:|---:|
| Blind | 24/48 | 10/48 | 115.417 / 112.667 | 0.099755 / 0.101472 |
| Budget-only | 24/48 | **0/48** | 138.750 / 138.750 | 0.112753 / 0.113538 |
| Raw counters | 28/48 | 1/48 | 137.583 / 138.042 | 0.107272 / 0.114469 |
| Engineered | **48/48** | 2/48 | 139.833 / 137.958 | 0.114519 / 0.110972 |

**Independent post-return audit of original trajectories:**

| Arm | ≥8 new outside-corridor cells by decision 46 | Mean recovery actions before leaving corridor | Mean new outside-corridor moves at step ≥64 |
|---|---:|---:|---:|
| Blind | 40/48 | 7.00 (successful resumptions) | 78.25 (successful resumptions) |
| Budget-only | 48/48 | 5.17 | 79.92 |
| Raw counters | 48/48 | 5.67 | 79.48 |
| Engineered | 48/48 | **4.00** | 78.90 |

The engineered comparator matches the predetermined strategy at every tested
fork and saves ~1.17 recovery actions versus the budget-only arm.
**But it has two early STOPs while budget-only has zero.** Wrong initial
BACKTRACK decisions may be corrected with later RELOCATE_START decisions, so
teacher agreement is not sufficient to establish survival or game competence.
The raw-counter arm is mixed, and the simple blind controller has 10 early
stops. All reported 160-step runs are synthetic extrapolation, not native
acceptance or exemption from the native 64-logical-move limit.

## Verifiable evidence and negatives

| Seed | Checkpoints | Original episodes / decisions | Hashes | CPU campaign | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 596 | 4 | 96 / 13,304 | 105/105 | 30.859 s | 298,668 KiB |
| 597 | 4 | 96 / 13,185 | 105/105 | 30.906 s | 299,288 KiB |

Each per-run artifact includes checkpoint SHA-256 IDs, original per-step
softmax probabilities, complete host-legal menus, material/ink/position
transitions, original replay SHA-256, action histograms, measured RSS and
latency, PNG gallery, source hashes and a manifest. Restored-checkpoint
neural probabilities and original host-state transitions replay exactly.
The new eight-test focused suite passes locally. First-pass overly broad
evaluation exceeded its execution envelope; it was reduced to a bounded,
predeclared matched depth/cap design, **not counted as a successful run**.

The exact executed source file has SHA-256
`58edb9d7314bca22c4fff0ac321d01a22b2177563807144caed4c1b7a72a73ca`;
the published Git blob is `0a89e46ffee217f3d83b0249aaa1ddab235fcd25`.
Other source dependencies retain their preceding qualified hashes.

No 9000-series HELDOUT family was created or inspected; zero native receipts,
zero paid GPU/API usage, zero actual d1/LFM/Sprocket adapter loads, no
CyberSand game modification. Keep PR Draft/HOLD and unmerged.

**Conclusion:** the feasibility inequality is useful *when explicitly
supplied*, but this experiment does not establish that the trained worm
can infer it from raw counters, let alone recover on novel topologies.
Next distinguish learned cost-to-go from handcrafted feasibility on
multi-branch maps while evaluating survival, recovery tax, and motif quality
separately.
