# D-2 / sustained soft-waypoint steering (DEVELOPMENT, synthetic only)

**Study:** `d2.worm-waypoint-steering/1`. This is an independent offline controller test, **not** a learned-model, native World, material Undo or aesthetic acceptance result. Original six 80×80 obstacle geometries, seeds 23/79, guide fields, pen cadence and strong-attraction source policy remain unchanged. The registered decision state is tick 60 for each of twelve geometry/seed pairs. The goal is a legal, geodesically reachable northeast waypoint, chosen from the *pre-intervention* cursor and geometry; it is not adjusted after observing any policy results.

Each arm receives an identical 32-movement legal cardinal budget, RNG stream, starting material, visited counts and hard permissions. A source-replayed, soft `u16` preference boost to **one legal move with shortest obstacle-aware distance to the waypoint** is applied for 0, 1, 4 or 12 decisions. The score/trajectory and material outcomes are independently reproduced from committed Python source. The soft guide never overrides legal candidates, grants native material permissions or invokes host Undo. Resulting drawings are *not* objective art-quality ratings.

| Arm | Goal reached (of 12) | Sum minimum distance to goal (lower better) | First choices changed | Subsequent cursor mismatches vs sham |
|---|---:|---:|---:|---:|
| Sham (0 guide increment) | 0 | 142 | 0 | 0 |
| One +4096 pulse | 0 | 145 | 7 | 148 |
| Four +4096 nudges | 0 | 136 | 7 | 329 |
| Twelve +4096 nudges | 1 | **50** | 7 | 368 |
| Twelve +8192 nudges | 1 | **35** | 9 | 373 |

**Observation:** With an identical 32-decision budget, twelve-step soft steering improves the *closest approach* to a source-frozen waypoint for all 12 matched states at both +4096 and +8192, but actually reaches the waypoint in only 1/12. One-turn steering does not consistently improve the approach (better 1, worse 1, unchanged 10), despite differing first decisions in 7/12. Short-horizon direction changes are not a substitute for ongoing goal-directed controllability. The held arms also alter downstream drawings and can reduce sustained autonomous wandering. These are synthetic, highly related DEVELOPMENT conditions; not independent held-out samples, learned results, native painting, or claims of aesthetic merit.

**Known limitations:** The target is artificially selected, not an authored goal or human-drawn line. The minimum-distance improvement favors the held policies' own optimization target and cannot establish quality, topology or route usability. Some held cases diverge materially after preference is restored. Wall geometry remains a synthetic permission mask. The stage D original eight-case frozen policy confirmation, complete exact-head suite, and genuine human blind ratings remain separate gates.

**Evidence construction:** `python -m unittest test_worm_waypoint_study.py -v` and `python worm_waypoint_study.py --out <new-directory>`. All case records, material-before/after hashes, scalar steering magnitudes, cursor histories, legal candidate lists, input/algorithm source hashes, pristine/replayable synthetic before/after PNGs and honest native/model-null provenance are committed in the generator, not in the repository as large PNGs. `verify()` independently regenerates every artifact from the source and rejects rehashed tampering. The GitHub Actions 2/4/8-runner matrix remains inert and was not dispatched.
