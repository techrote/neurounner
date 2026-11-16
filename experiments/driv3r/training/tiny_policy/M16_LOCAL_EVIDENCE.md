# M16 — OOD recovery-conditioned training and local tabu rescoring

**2026-11-16 · Local exploratory evidence only · No model promoted**

## Hypothesis

Does relabelling off-trajectory legal states during training improve long-horizon waypoint completion? Can short-history legal-action rescoring reduce loops without sacrificing spatial coverage?

Four scratch heads received **160 real AdamW updates each (640 total)**: BC MLP, recovery-enriched MLP, BC GRU and recovery-enriched GRU. Separate untrained MLP, goal-greedy, novelty, global privileged BFS oracle and two local tabu-reranked MLPs were included as matched controls. All model logits were constrained by the host-supplied cardinal/STOP legal menu; no native CyberSand execution. The tabu variant only downweights moves visited in the previous twelve steps (plus an additional previously-visited penalty). It does not consult the global privileged BFS oracle.

**Synthetic research model:** independent 32×32 obstacle/protected-cell worlds, 7×7 local patches, single-cell painting, six host-projected motif waypoints, 64-turn budget. Training seeds 91000–91015; DEV 92000–92003; OOD research EVAL 93000–93011. OOD worlds have independent U-pocket, offset-ring and comb generators, paired across three family labels. This is *not* the 96×96 CyberSand native/Workbench ABI or sealed FINAL.

## New 36-matched-world OOD rollouts per controller

| Controller | Unique cells | Waypoints | Retraces | Branches |
|---|---:|---:|---:|---:|
| Privileged BFS oracle | 58.94 | **3.25** | 6.06 | 1.03 |
| Greedy | 8.94 | 0.31 | 56.06 | 0.78 |
| Novelty | **64.33** | 0.25 | **0.67** | 0.42 |
| Untrained MLP | 25.53 | 0.00 | 39.47 | 0.00 |
| BC MLP | 50.14 | 0.22 | 14.86 | 1.64 |
| Recovery MLP | 41.22 | 0.08 | 23.78 | 1.22 |
| BC GRU | 42.03 | 0.00 | 22.97 | 0.75 |
| Recovery GRU | 32.58 | 0.17 | 32.42 | 0.19 |
| BC MLP + 12-step tabu | 59.97 | 0.22 | 5.03 | 1.69 |
| Recovery MLP + 12-step tabu | 59.39 | 0.19 | 5.61 | 1.81 |

Paired seed-cluster 95% bootstrap intervals for waypoint difference: recovery MLP − BC MLP **−0.139 [−0.333, +0.028]**; recovery GRU − BC GRU **+0.167 [+0.028, +0.333]**; BC tabu − BC MLP **0.000 [0,0]**. Those are exploratory uncertainty estimates after inspection of the research EVAL, not valid sealed FINAL claims. The absolute best trained waypoint completion remains below 0.25 of six goals; all policies remain unsuitable for promotion.

**Negative result:** Data augmentation does not repair navigation in this OOD setting. Tabu rescoring markedly increases coverage/decreases retraces but does not increase waypoint success. Better line texture statistics are not equivalent to meaningful goal progress. The next experiment should model explicit multi-step topology/recovery state, rather than reweight single-step imitation losses.

## Local source/evidence provenance

Standalone executed Python source (now committed in this PR): `m16_recovery.py` SHA-256 `648e0ba4e3f1768726716d085c207e0bfa6040c63a4d95df7e55fa30c6c794c1`, with `audit_m16_recovery.py` and `test_m16_recovery.py` in the local research bundle. Local ZIP SHA-256: `22f5b6debca5d631695c176bbd4cd2297f1aa9d006dd9378ea7b272bbdfbb2f6`. No checkpoint binary committed. This PR entry is a **results ledger, not exact-HEAD executable qualification**.

Audit passed 360 original state/PNG/receipt replays, 23,040 host-legal applied actions, 728 internal manifest hashes, four safe checkpoint loads and four adversarial/contract tests. A second full execution reproduced **729/729 files byte-identically**, including the manifest.

**Open custody/qualification gap:** M16 source and weights presently remain in the downloadable local evidence bundle. No new immutable GitHub Actions artifact or hosted job was created; the committed source still needs explicit hosted requalification from Neurounner HEAD. Earlier M15 source and immutable hosted artifacts remain preserved by the migration manifest; this entry does not claim new pretrained-backbone or LoRA updates.

CyberSand is read-only; production game, native, Godot, Workbench and provider files untouched. Neurounner PR #252 remains Draft/HOLD and unmerged.
