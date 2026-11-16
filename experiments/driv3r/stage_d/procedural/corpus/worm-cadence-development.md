# D-2 DEVELOPMENT: drawing-worm steering duty cycle and arrival release

**Offline synthetic only; Draft/HOLD.** No d1 model, native painting, GPU/API, training, physical material history/Undo, paid jobs or experiment-matrix dispatch. The separately authorised native wormbrain #611 is not modified.

## Hypothesis

Continuous +16384 guidance forced the prior synthetic worm into precisely the geodesic-oracle path in 12/12 reused DEVELOPMENT scenes. Explore intermittent steering and release-after-arrival rather than treating shortest-path imitation as artistic success. Frozen starts: six 80×80 obstacle layouts (open, slats, ring, canals, baffled, crucible) × seeds 23/79, tick 60. Identical initial visited state, ink, heading, deterministic noise, 32 cardinal-movement decisions, host-like legal action list, original scalar field and pen cadence. Waypoint is source-derived and reachable. All eight arms predeclared: sham, full +8192/+16384, +8192 on the first 1/2/3 of each four turns, and +8192/+16384 switched off after the first goal hit. Release does **not** STOP, teleport, edit history or force a move.

## Results (12 matched cases per policy; DEVELOPMENT)

| Arm | Reached | Sum closest goal distance | Oracle-step matches /384 | Turns | Fresh painted cells | Nonzero guide decisions | Post-arrival near-goal visits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sham | 0/12 | 142 | 11 | 117 | 260 | 0 | 0 |
| 1/4 +8192 | 0/12 | 108 | 20 | 176 | 263 | 96 | 0 |
| 2/4 +8192 | 2/12 | 63 | 60 | 178 | 267 | 192 | 2 |
| **3/4 +8192** | **12/12** | **0** | **132** | **173** | **260** | **288** | **53** |
| Continuous +8192 | 12/12 | 0 | 263 | 146 | 249 | 384 | 105 |
| Continuous +16384 | 12/12 | 0 | 384 | 207 | 182 | 384 | 192 |
| Release +8192 at hit | 12/12 | 0 | 196 | 67 | 303 | 192 | 16 |
| Release +16384 at hit | 12/12 | 0 | 196 | 67 | 303 | 192 | 16 |

**Finding:** 3/4 intermittent guidance reaches all 12 source-reused waypoints but follows the oracle at substantially fewer aligned steps than continuous guidance, yielding more turns. Lower duty cycles lose endpoint reliability. Releasing guidance on arrival avoids prolonged goal-local oscillation and permits additional new marks, but produces fewer turns; both release gains collapse to identical sampled trajectories. These are measurements of a deterministic hand-coded policy, not d1-2B results. No claim of aesthetic quality, novel generalisation, or host-native success. A different geometry/seed cohort must be frozen *before* any promotion of 3/4 cadence. Existing blind image preference ballot remains genuinely unrated.

## Custody and replay

Use `python -m unittest -v test_worm_cadence_study` and `python worm_cadence_study.py --out /tmp/d2-cadence/episodes` from this procedural directory. The 96 cases preserve original/final PNGs, full source-replayed cursor/legal-candidate/pen/guidance/material event chain, null native/model evidence, source hashes and byte-verified deterministic manifests. All six new unit tests pass locally against the exact derived source. Tamper-plus-rehash is rejected. Original prior sham/full-control paths/material must match exactly. Captured local canonical summary SHA-256: `0137fc795d0d5096ba0ed8d10d8e6fef1366a0e5a1a48f34c17a8c05f881470e` (source hashes are part of this digest). The earlier frozen eight-case source confirmation, complete inherited D-2 suite and human visual review are still outstanding. BACKTRACK_CURSOR and RELOCATE_TO_BRANCH remain logical/material-preserving; UNDO_MATERIAL_EDIT is unimplemented without host-owned history and native receipt.
