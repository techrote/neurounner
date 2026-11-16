# MZ-014 — spatial-frame transfer audit of the MZ-013 trained controllers

**Research-only, Draft/HOLD; no model retraining this pass.** Parent MZ-013 source HEAD
[`66b30e6fd3799bc331f2ce066ed106720bbcb262`](https://github.com/techrote/cybersand/commit/66b30e6fd3799bc331f2ce066ed106720bbcb262),
[successful 700-update training run](https://github.com/techrote/cybersand/actions/runs/43923845089),
immutable artifact ID **14255566274**. Exact `models.pt` SHA-256:
`10d9c62370abebaa9d2f644cfe620dda77bbaf651491701b01a78dff7b5aed11`.

## Question

Can the MZ-013 recurrent controller execute the same legal spatial game
when its observation is expressed in a different coordinate frame? The
physical protected-cell rules, source-pinned weights, delayed memory cues,
paint mutation, stop/retrace actions and underlying generated worlds remain
identical.

The intervention transforms **all** of (1) the 5×5 categorical local view,
(2) absolute x/y position channels, (3) host-legal action mask,
(4) previous action tokens, and (5) output action symbols back to physical
host actions. Beacon/inversion, progress and appearance nuisance features
are invariant. Incomplete mask/action remapping would invalidate the test.

## Locally independently reproduced, 16 November 2026

Three untouched shifted-spacing seeds 7000..7002, two cues × two inversions
= 12 episodes per condition. Models loaded from **the original 700-update
checkpoint**, not random initialization.

| Frozen controller | Normal clean | Horizontal mirror clean | Vertical mirror clean | Rotate 180° clean |
| --- | ---: | ---: | ---: | ---: |
| Diagonal-state 209,012 | 12/12 | **0/12** | 12/12 | **0/12** |
| GRU 237,428 | 6/12 | **0/12** | 6/12 | **0/12** |
| Reactive 21,310 | 0/12 | 0/12 | 0/12 | 0/12 |

The diagonal and GRU mirrored-x agents produced **no PAINT actions** and
prematurely STOPped at step 5. The diagonal agent made 12 raw illegal
proposals across 12 horizontal-mirror episodes. Normal diagonal completed
all 36 required paints with zero collisions, as expected.

### Causal controls that separate frame symmetry from absolute x position

Two additional interventions were run on the same frozen weights and episodes:

| Controller | Mirror X but retain original absolute-x scalar | Mirror absolute-x scalar alone, keep view/actions unchanged |
| --- | ---: | ---: |
| Diagonal | **0/12** clean | **12/12** clean |
| GRU | **0/12** clean | **6/12** clean |
| Reactive | 0/12 clean | 0/12 clean |

Therefore, simply changing absolute x does **not** explain the catastrophic
failure. It follows the horizontal transformation of the categorical view,
legal-action/last-action tokens and action names. Those factors are still
transformed together: their individual causal contributions are not isolated.

**Interpretation:** the learned policy is strongly dependent on the
training environment's fixed left-to-right travel orientation, even though
the vertical (mineral choice) reflection transfers. Prior 100% clean
corridor results are not proof of spatially robust gameplay control.

This is an **orientation-transfer failure**, not a test of maze navigation:
no arbitrary obstacle layouts or true route discovery were introduced.
It also does not show recurrent-memory failure; memory-cue dependence was
separately demonstrated in MZ-013. The transformer is intentionally
excluded from this bounded replay audit because it was already rejected
under its MZ-013 training configuration.

## Reproduction

The dedicated workflow downloads the exact artifact by immutable Actions
run and artifact name, checks checkpoint SHA-256 and parent source, verifies
frame-transformation invariants, runs all 216 closed-loop
(controller × mode × cue × inversion × seed) synthetic rollouts, checks
reproduction against locally measured baseline, and retains metrics
plus environment provenance in a new immutable Actions artifact.

```bash
python -m experiments.driv3r.model_zoo.frame_audit_mz014.audit \
  --checkpoint path/to/mz013/models.pt --out mz014-evidence --seeds 3
```

## Decision

**No model promoted.** Before scaling the diagonal controller or claiming
general spatial competence, train against at least horizontal and vertical
direction-balanced paths and obstacle layouts, then evaluate disjoint
generated topologies and actual PAINT/recovery paths. The next model-zoo
training curriculum should compare a spatial encoder with recurrent state
against a recurrent MLP on randomized maps, not optimize another
left-to-right corridor score.

Boundary: no edit to CyberSand game, Godot, native, Workbench,
materials, map assets or production configuration. No paid GPU/API.
Do not merge, Ready-promote or auto-merge.
