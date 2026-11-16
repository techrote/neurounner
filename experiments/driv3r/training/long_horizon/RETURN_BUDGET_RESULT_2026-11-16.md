# RETURN-BUDGET-001 — competing host-legal recovery routes

**16 November 2026. DEVELOPMENT-only controlled training.**

## Research question
Can a small trainable controller select between two simultaneously host-legal recovery actions when their feasibility differs solely because of the remaining recovery budget? Previous STOP penalties, local sensors, frontier sensing and recovery-balanced sampling did not consistently prevent upstream trap exhaustion.

## Experimental isolation and nonclaims
- Real NumPy Adam: 36→72→128 stages, 40 updates per stage and arm, two independently initialized seeds 594/595 (4 restored 29-parameter checkpoint binaries). Equal labels, examples, minibatches and initial shared parameter draws per seed; only capacity-derived action features are exposed in treatment.
- Two matched counterfactual states at the six-step corridor end have **identical cursor, material, live path, checkpoint input features in the blind arm, and legal menu** (`BACKTRACK`, `RELOCATE_START`, `STOP`). Their external recovery allowances are **2 vs 8**. The correct *handcrafted teacher choice* is RELOCATE_START when 2 remain vs BACKTRACK when 8 remain; capacity comparison is explicitly engineered as a feature, **not emergent planning**.
- TRAIN uses separate 1000-series seeds, DEVELOPMENT uses separate 3000-series seeds. Structural HELDOUT 9000-series families are not constructed or inspected. The challenge corridor is the same six-cell overlay across eight DEVELOPMENT material contexts; **this is not a new-topology generalization claim**.
- The host action menu is always authoritative and externally forced mistakes are six genuinely legal `MOVE_E` operations. Experiment changes `max_recoveries` only in the synthetic World interface (2 or 8), **never** the native game contract or native 64-logical-move ceiling.
- Backtrack success is defined against a deliberately chosen *strategy teacher*, not a globally optimum trajectory; relocation can yield more newly painted cells because it consumes fewer recovery decisions.

## Source-replayed original results (eight DEVELOPMENT material contexts per seed/arm/cap/horizon)

| Seed | Budget | Model | 128 fork matches / rescue / early stops | 160 ink mean / rescue / early stops |
|---|---:|---|---|---|
| 594 | 2 | blind | 0/8, 0/8, 8/8 | 6.000, 0/8, 8/8 |
| 594 | 2 | capacity_visible | 8/8, 8/8, 0/8 | 159.000, 8/8, 0/8 |
| 594 | 8 | blind | 8/8, 8/8, 0/8 | 155.000, 8/8, 0/8 |
| 594 | 8 | capacity_visible | 8/8, 8/8, 0/8 | 155.000, 8/8, 0/8 |
| 595 | 2 | blind | 0/8, 0/8, 8/8 | 6.000, 0/8, 8/8 |
| 595 | 2 | capacity_visible | 8/8, 8/8, 0/8 | 158.750, 8/8, 0/8 |
| 595 | 8 | blind | 8/8, 8/8, 0/8 | 155.000, 8/8, 0/8 |
| 595 | 8 | capacity_visible | 8/8, 8/8, 0/8 | 155.000, 8/8, 0/8 |

**Primary result:** With budget 2, the blind policy always chooses BACKTRACK at the shared fork and exhausts its final recovery opportunities in the tunnel: **32/32 early stops**, **0/32 successful resumption** across seeds and horizons. The capacity-conditioned policy instead chooses RELOCATE_START and resumes painting in **32/32**, with **0/32 early stops**. Both arms handle budget 8 correctly in **32/32**.

The failure is **not a general policy deficit proven on new maze topologies**. It is a purposely identifiable observation-aliasing failure: a budget-blind model receives indistinguishable inputs yet is required to choose opposite actions. The visible model receives explicit feasibility-related features. Results show that preserving host recovery-capacity information can be essential to control, not that the tiny worm learned robust planning or artistic intelligence.

## Checkpoint provenance, latency and negative controls

| Seed | CPU wall s | Peak RSS KiB | Episodes replayed | Host decisions replayed | SHA256 entries |
|---|---:|---:|---:|---:|---:|
| 594 | 13.329 | 122076 | 64 | 7056 | 71 |
| 595 | 12.265 | 122556 | 64 | 7056 | 71 |

Mean decision latency over original 160-decision rollouts (ms):

| Seed | Budget | Blind | Capacity-visible |
|---|---:|---:|---:|
| 594 | 2 | 0.083521 | 0.132811 |
| 594 | 8 | 0.108633 | 0.134428 |
| 595 | 2 | 0.071339 | 0.112651 |
| 595 | 8 | 0.106369 | 0.113180 |

Full original per-action probabilities, exact host menus, first-actions, intervention details, painted cells and trajectory hashes are in per-run immutable artifacts. Every original episode was independently replayed from **restored** weights, with per-step softmax maps, source World pre/post hashes, material edit invariants and trajectory SHA checks. Verifier rejects forged original menus, model probabilities, checkpoint hashes and material states.

Checkpoint SHA-256 identities:

| Seed | Arm | SHA-256 |
|---|---|---|
| 594 | blind | `8129151aa346438b18980db2368670751a0106672b0cf365d40bcce7dc7dc92a` |
| 594 | capacity_visible | `6530441c4a29232e7a96bb7370cbe86e1b54cfcade9e60631268a7ded4993e19` |
| 595 | blind | `013aa2b3fd3d77c89fdbf8bdf84648fd13a6fc2bcb0661ef9edb683e026dfd8b` |
| 595 | capacity_visible | `35451bb5d93ac914a7bb45eac4e5f303002b5409d77d5589d6ad67e40dd48b90` |

Source dependency SHA-256:

```json
{
  "policy.py": "f3e6469b592e68c0d3410dca88e107b215f1bdb1c16f7ebd73232ea9470cbb82",
  "return_budget_lab.py": "5a02d02a5d9aefd6bfc23b95367e8587f7a1507ce9f3be818241d78133d12846",
  "train.py": "9ed2c9ac49fa397c12f0b8256e4d4439bd3a9294b002b54276b702a1e3ddcacb",
  "world.py": "6801c401c356869ed6535fec5da4a627871cb05fff9c95d6726224a492db678a"
}
```

## First-pass correction and remaining limits
- A first attempt of the new module failed at parse time because of two unmatched parentheses in the verifier. No first-pass checkpoint/result was counted. After repair, the full local experimental campaigns and verifier completed, including the new eight-test regression subset.
- This experiment is unusually easy and intentionally diagnostic: the corridor creates exactly one high-signal fork; the desired strategy is encoded as a bounded capacity/path-depth comparison. Larger unseen tunnel depths, dynamic painting constraints, more than two host alternatives, physical native traces, real models and artistic style retention are **not established**.
- 160 sequential synthetic decisions are *outside* the 128-decision training horizon. They do **not** authorize any 160-action native game run.
- Draft/HOLD, no merge/Ready. No native, game, material/asset or production setting edits; no paid inference/GPU; no structural held-out reads; no large SLM weights trained.

## Next falsification
Generalize the budget/depth mechanism to **variable-length corridors and distinct return junctions**, then vary cap independently of length on structurally disjoint DEVELOPMENT maps. Test whether strategy selection and post-return motif retention hold without directly encoding `remaining < path_depth` as a feature; compare a budget-only feature arm against the engineered-ratio arm.
