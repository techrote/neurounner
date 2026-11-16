# CyberSand Game Corpus Factory — multi-topology recovery v3
**16 November 2026 — Draft/HOLD, research only**

## Executed training
The baseline student fitted on **12,939 privileged-teacher decisions across 48 original-generator TRAIN worlds** with 36,332 candidate hard negatives. It then controlled a synthetic host for **5,706 actual decisions** on 12 other original-topology TRAIN worlds and 12 horizontal-barrier TRAIN worlds. The geodesic teacher labelled the student's visited states, not its proposed actions; hidden whole-world geometry was excluded from prompts. Normalised source-aware deduplication yielded **1,075 original-only** versus **1,884 mixed-topology TRAIN correction examples**. Three independent 24-epoch CPU SGD fits were executed (source baseline, original-only correction student, mixed-topology correction student). No checkpoint binaries entered Git.

## World-disjoint closed-loop result
Two fitted policies controlled 12 unseen original worlds, 12 unseen horizontal-barrier worlds, and 12 unseen **vertical-gate slalom** worlds, each with 3 correlated family variants, for **216 host-applied trajectories** and a 160-action safety bound.

| Unseen evaluation topology | Original-only successes | Mixed-topology successes | Mean two-step returns (original → mixed) |
|---|---:|---:|---:|
| Original | 11/12 worlds | 11/12 | 16.306 → 4.083 |
| Horizontal barrier | 9/12 worlds | 9/12 | 34.500 → 20.500 |
| Vertical slalom, untrained generator | 0/12 worlds | **3/12** | 102.333 → 36.778 |

For slalom, family-rollout successes were 0/36 → 4/36 and mean visited cells 40.139 → 97.722. The slalom's reachable oriented shortest paths require 134–136 moves, leaving only 24–26 spare steps. **Nine slalom worlds remain unsolved.** Mixed training does not improve success on either previously represented generator. The intervention changed both the number and topology of examples, so this is not a size-matched causal ablation.

## Integrity and reproduction
All 36 evaluation-world geometries are SHA-pinned, never used as training examples, and evaluated separately from the TRAIN-only corpus. Five focused regression tests passed, including duplicate-geometry split rejection, unreachable/illegal-action checks, and rejection of non-TRAIN rows in the export. Two separate complete CPU executions reproduced the exact same output hashes:

- Source Git blob: c284f09de420f05f95746b141e5d990dddc22cc1
- Test Git blob: 35a643a31b2e627f55b08417ef9c3e104a7615ce
- Report SHA-256: 0c0d8a36fe358ffdfe5c177ee455354e6813de11bc39894f504b70c5b63eba3a
- TRAIN ZIP SHA-256: 61a3fa64b44346dbd000670d45e4f992f1c9f85e9e755bded2352ff6f4cbd26f

The deterministic report includes all 216 action summaries, per-world source hashes, hard-negative provenance and 12 original generated PNG receipts. Run the domain_recovery_v3.py script with --out set to a new directory, and test_domain_recovery_v3.py using unittest. For reproducible bounded CPU behaviour use OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1.

**Strict qualification boundary:** synthetic research only, not native CyberSand/W0, official DEV or held-out evaluation, Sprocket/LFM model behavior or artistic competence. Data and PNG evidence were locally verified but are not immutable hosted Actions artifacts. Do not train on slalom evaluation rows. No game-side, provider, paid compute, schedule, merge or Ready transition.
