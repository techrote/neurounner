# FRONTIER-VIABILITY-001 — Bounded topology versus local escape sensing

Date: 2026-11-16. Model-side research only. Source location:
`experiments/driv3r/training/long_horizon/frontier_viability_lab.py`.
All results below are original **local CPU** runs. Only GitHub Actions-generated
output can be described as an immutable Actions artifact; local archives are
not substitutes for repository-source-matched Actions qualification.

## Falsifiable question and controlled design

Prior TRAP-PREVENT-001 local free-degree/two-turn features did not reliably stop
premature option exhaustion. Can a new **bounded connected-frontier estimate**
help an explicitly trained worm avoid small reachable regions and recover after
forced legal detours, while retaining long-horizon interestingness?

Three matched 39-weight linear softmax models are trained with **identical
initialisation, teacher-labelled TRAIN states and Adam update schedule**:

1. `blind`: 24 existing action features, 15 extra features zeroed.
2. `local`: the same base plus existing nine local escape-risk features.
3. `frontier`: base, nine local features, and six 12-cell-radius / 96-node
   flood-fill occupancy features for each currently legal candidate action.

The sensor treats the candidate target as **new ink** and samples reachable
empty successor cells, respecting material solids, protected cells, existing
ink and the 96×96 bounds. It does not choose paths, perform A* or grant actions.
No future world state, future teacher commands or held-out label leaks are used.
Training teacher mixes original wall/route/motif steering with a **one-step,
state-only connected-frontier preference** and imperfect jitter, and sees the
same current host menu as the models. The teacher's explicit frontier access
creates a representation-favouring comparison; it is **not** an independently
fair comparison of unconstrained algorithms. All trained models run without
live teacher assistance in DEVELOPMENT rollouts.

TRAIN stages: 36 → 72 → 128 decisions, 50 Adam updates per stage, 150 per
checkpoint. Two seeds (588 and 589), six checkpoints (39 weights each), three
arms, two matched 128/160-decision horizons, intact and forced replay. The
TRAIN corpus contains pickets/ridges/islands/zigzag plus train-only synthetic
corridor variants and 24% opportunistic imperfect legal movement. Development
uses **pickets and ridges only**, plus a predeclared corridor variant; three
scenarios per seed, NOT a broad structural generalisation study.

Forced schedule: on corridor worlds request eastward movement for decisions
0–4 whenever legal; later request an alternative legal movement at 12, 35, 71
and 109 when available. The synthetic host alone validates every action.

## Original local results (forced DEVELOPMENT, three scenarios per arm)

| Seed | Horizon | Arm | Mean ink cells | Mean branch cells | Early STOP | Mean decision latency |
|---|---:|---|---:|---:|---:|---:|
| 588 | 128 | blind | 126.333 | 0.000 | 0/3 | 0.144 ms |
| 588 | 128 | local | 126.333 | 3.667 | 0/3 | 0.224 ms |
| 588 | 128 | frontier | 124.000 | 29.667 | 0/3 | 0.688 ms |
| 588 | 160 | blind | 158.333 | 0.000 | 0/3 | 0.157 ms |
| 588 | 160 | local | 158.333 | 3.667 | 0/3 | 0.236 ms |
| 588 | 160 | frontier | 146.667 | 39.333 | **1/3** | 0.698 ms |
| 589 | 128 | blind | 126.333 | 0.000 | 0/3 | 0.149 ms |
| 589 | 128 | local | 126.333 | 0.667 | 0/3 | 0.222 ms |
| 589 | 128 | frontier | 126.333 | 0.667 | 0/3 | 0.814 ms |
| 589 | 160 | blind | 158.333 | 0.000 | 0/3 | 0.152 ms |
| 589 | 160 | local | 158.333 | 0.667 | 0/3 | 0.232 ms |
| 589 | 160 | frontier | 158.333 | 0.667 | 0/3 | 0.800 ms |

Each matched seed/horizon has three forced scenarios, plus three intact scenarios
not shown in the table; full action distributions, recovered decisions, source
hashes, STOP occurrences, per-step logits and PNGs are inside the evidence.

**Result: no demonstrated continuation or recovery advantage.** The connected
frontier can produce much more branching (seed 588), but sacrificed survival,
including a premature forced 160-step STOP when both simpler controls ran to
160 decisions. Seed 589 frontier performed essentially identically to local.
More branches are not evidence of artistic quality or successful navigation.
The 160-step tests are extrapolation beyond the training horizon, not native
qualification. The extra sensor makes decisions roughly 3–6× slower than the
local-feature control in the evaluated CPU environment.

A negative but useful observation: the hand-built `trap_corridor` is **not
sealed at the starting position**. The world constructor clears the 44–52
centre square, so the first eastern move still sees open space. The bounded
frontier detects its narrow compartment only on the following move. The
initial test incorrectly asserted that the sensor distinguishes pocket/open
from start; it failed and was corrected to test the real x=49 choice. The world
contract was not edited to manufacture a better result.

## Qualifications and immutable-result provenance

- New isolated test inventory: **9/9 PASS**.
- Available historical source subset (`test_long_horizon`,
  `test_trap_avoidance_lab` plus new tests): **32/32 PASS**.
- Six restored, genuine CPU-trained `npz` checkpoints; two source-replayed
  campaigns, **72/72 independently audited episodes**, **10,346/10,346 original
  sequential host actions** and **92/92 original file SHA-256 manifest entries**.
- Seed 588: 19.753s training-plus-rollout; 126,120 KiB peak process RSS;
  36 episodes / 5,162 source-replayed decisions.
- Seed 589: 23.772s; 126,808 KiB peak RSS; 36 episodes / 5,184 decisions.
- Every recorded choice and probability is re-derived from the **restored
  checkpoint**; the verifier rejects altered original probabilities, host menus
  and material/geometry transitions. No native receipt is fabricated.
- Checkpoints remain outside Git history; publish under the dedicated
  source-matched GitHub Actions workflow's artifact retention, never via Git.
- No structural HELDOUT worlds have been evaluated in this campaign. There is
  no claim of native acceptance, true game engine/material integration, or
  source-verified real CyberSand event-trace ingestion.

Original checkpoints (SHA-256):

| Seed | Arm | Checkpoint SHA-256 |
|---|---|---|
| 588 | blind | `0114a9fcf4f5c4ce204c7e85fbf3db12699c9f5ae27b6dbad7b619b24888ecf1` |
| 588 | local | `4d2d18d3f8a592cb90f6dca136c40f81d26628c96d31d23f541ce351c11793c2` |
| 588 | frontier | `3b2b95c65f32f014ab7f982bd1b27dade48e262e80e2702e76aad2ba5f4a86b3` |
| 589 | blind | `e32dc305332fbc937969426de56700b224f415d6a6e5e27f4712b969b56f15b2` |
| 589 | local | `a67818afd1056c3bb487ae62c6c8a9a277a79c2e7c9916e78b107c72e70529f5` |
| 589 | frontier | `6e7b9b46cc2d77e5358761ed604c461716c0977c433c3efa11730e197487166d` |

The original **local** snapshots above are not claimed to be GitHub-generated
artifacts. Their source and checkpoint byte hashes are recorded in both
`report.json` files. Recomputed Actions artifact checkpoint hashes must be
recorded separately, because NPZ ZIP timestamps/packaging can differ.

## Implication

Topological free-space detection alone is insufficient. Next investigate an
explicit *escape-debt/return-policy* and a separate style state: distinguish
short detours with viable recovery from irreversible consumed branches, while
penalising gratuitous branching and retaining stylistic diversity. Avoid an
A*-driven path that merely makes the neural policy decorative.

Research boundary: 96×96 read-only-derived synthetic host; 64-logical-move
native ceiling unchanged. No `native/**`, `godot/**`, Workbench, scene, asset,
production provider, game default, or deployed model changed; no paid SLM/GPU.
