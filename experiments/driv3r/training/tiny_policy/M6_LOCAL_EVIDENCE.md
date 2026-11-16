# M6 blind-topology transfer — executed local evidence, research HOLD

**16 November 2026 · PR #634 · provisional source HEAD `63383391f98849e6473c44814017dbf779b1babb`.** This report records an actually executed, audited CPU-only experiment under the `experiments/driv3r/training/tiny_policy/**` ownership boundary. **No M6 optimizer, LoRA, paid GPU/API, game/native/Godot/Workbench/provider, FINAL, merge or production action.**

## Experiment and contracts

Compared four **real reloaded M3/M5 trained scratch decision-head checkpoint models** with legal random, deterministic wall, novelty greedy and local oracle on 12 novel synthetic world seeds `27000–27011`, three family styles per seed, six geometry families: chambers, crossroads, pocket-stair, hairpins, guarded-plaza and sawtooth. Each policy got the same 96×96 blocked/protected geometry, 64-action window, unrestricted STOP, 96-action synthetic host budget and legal actions **`UP DOWN LEFT RIGHT STOP`**. Policy observation is 162 local features; global reachability is audit-only. 36 matched episodes per policy, 288 original PNG+JSON pairs, **18,432 actual decisions**. No training updates. This is not native CyberSand, W0 Sprocket or admitted FINAL.

| Policy | Mean cells /64 | Mean branches | Mean retraces | Early STOP /36 |
|---|---:|---:|---:|---:|
| Random legal | 24.917 | 6.806 | 40.083 | 0 |
| Wall deterministic | 32.083 | 0.333 | 32.917 | 0 |
| **Novelty greedy** | **59.917** | 0.750 | 5.083 | 0 |
| Local oracle | 57.194 | 1.028 | 7.806 | 0 |
| M3 MLP (trained) | 53.694 | 0.750 | 11.306 | 0 |
| M5 linear (trained) | 52.583 | 0.694 | 12.417 | 0 |
| **M5 MLP (trained)** | **55.611** | 1.306 | 9.389 | 0 |
| M5 convolutional (trained) | 55.444 | **2.083** | 9.556 | 0 |

M5 MLP's mean gain over M3 MLP is **+1.917 cells**, but the 12-world, seed-cluster bootstrap 95% CI is **[-1.167,+5.500]**; not a statistically established improvement. Against novelty greedy, M5 MLP is **-4.306 cells**, CI **[-9.833,-0.222]**. Treat the **12 geometry seeds**, not 36 repeated family-labelled episodes, as independent. Family-conditioned mean three-way footprint IoU: M3 MLP 0.262, M5 MLP 0.187, M5 conv 0.354, M5 linear 0.690, wall/novel 1.000. Character is conditional but not yet useful artwork.

Failure topology: **hairpin corridors** — M3 MLP 24.33, M5 MLP 29.00, M5 conv 36.83, novelty 41.00 cells. Conversely, open guarded plazas are near-saturated by all. Evidence includes original rather than generated six-topology raster comparison galleries.

## Independent qualification and provenance

The independent auditor reloaded and SHA-validated all 6 M3/M5 checkpoints, recomputed **18,432/18,432** actual model/control decisions and host replays, checked **288/288 original PNG raster pairs**, and all **584/584 internal SHA-256 manifest entries**. Illegal applied actions **0**. **6/6 positive/adversarial source tests** passed, including protected-only STOP, split identity, forged state, and altered legal-mask rejection. A second fresh process recreated the original **580/580 core evidence files byte-identically**; four additional gallery files were then added and audited.

Immutable M3 model hashes: linear `a5d1ec8f5f956c05968f7b5b538952e68d89b861d337a48559e0de759fc32435`, MLP `1b2c9cea0b99fffde0b3ebd39942835540880d93a6e70ce1518ff39c508ceeb5`, conv `b44a97d0a6af84010280bf83e22c481afa15546d8b1575ef4c855d5bb6661d6d`. M5 hashes: linear `8382459268e258025ef5e0324ad2aceea385a8ae084f0fac8d5f793f4a90db8d`, MLP `55f0ea469278086bef2c91b533a8451e1938d1739d596dd3f0ed8542a7267940`, conv `39d0c2eb067eb0a12cb454f859de8f6434294704f7e3494af1999d25b484f636`. M5 synthetic TRAIN split identity: `2728c8ae225cc122dd39102d0a80f7866a31922fd3f6b39868aa038f60ea6f4c`.

**Important qualification gap:** Original executable `m6_blind_transfer.py` SHA-256 `bed8b30ad97be3d0511aef88cdb7e5385524e85878966f338d87f44c5bc86be6`, independent auditor `bbdaf2280415242c73ed3666cdebe0aa754c96179413092a5c86f6c545352ab1`, tests `d939790d130c927b3cb9eac205ef8dabf2aac359ffbe7bffc31f0dd398d3506f`. These scripts and the evidence/checkpoint archives are delivered as a **local ZIP in the originating conversation**, not yet committed or hosted as immutable GitHub Actions artifacts. They used the locally archived M3/M4/M5 code, which is behaviourally replayed but **not byte-identical to the current repository's M5 source**. Do not label M6 exact-source-hosted or production-qualified.

**Next:** commit exact local M6 source under tiny_policy only, reconcile with repository host/M5 source, source-match/host the immutable evidence, then run new protected-cell recovery, active branching and line-texture quality evaluations. No promotion until earned. PR remains Draft/HOLD; do not merge.
