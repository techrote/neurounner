# C8 — topology transfer and correction feedback, 16 Nov 2026

Research-only; Draft/HOLD. This experiment trains a 25→16→1 BACKTRACK-only residual on the frozen C7 worm head. The residual activates exclusively when both the host recovery cue and goal-change bit are present. No other action logit can change. Without both signals, the entire C7 output is bit-identical. The synthetic host alone validates legal actions and protected materials; BACKTRACK moves the cursor without removing painted material.

## Model and actual optimization

Training topologies are new hairpin and staircase trails with forks and productive exits; unseen hook and zigzag topologies are withheld from fitting. Across the input conditions, the experiment matches World, seed, legal menu and material hashes for no/weak/strong guidance and cue-off/cue-on observations. Both personalities, four rotations and mirroring are represented independently of training family. A changed goal can legitimately require retracing history or continuing to paint; the latter is a deliberate false-cue condition.

240 training source groups × 2 goal outcomes × 3 guidance strengths = **1,440 observations**; **210 actual CPU AdamW optimizer steps** across 14 epochs; regularized binary loss **0.319629 → 0.126437**. New adapter checkpoint SHA256: **b1d485773fd33d760d7a41f1bb90b3aa23e208f1241c7dfcabb04cef1aa697bd** (2,340 bytes).

Frozen prerequisite hashes: cue-calibrated base 8c8625a1bd8daed03e00188687dbba9732634c7bf7b8e7be5a4fa21cb310ac09; C4 cue adapter f1d538200c0d5e183d46463a9058f73db162a5dcccf13daeb521073da9bfc0fa; C7 personality gate 9ddadf372465abca9634eca8bf450cb03e5903434da8065de588511b44b5da9a.

Training array SHA256: 8680389668fc069572f2512d4411b305a3198e31287736b00145055dec8ba97a. Training source-list digest: e62b0720630d83bbc5f48df55fc3a63478055426bd86cfb11e0b113f2ae68c2f.

## Matched greedy first actions, with cue and goal change

Each pair uses 48 held-out seeds per family and strength; strength arms are matched, not independent topology samples.

| Condition | C7 | C8 |
| --- | ---: | ---: |
| Trained hairpin/staircase: reverse onto history → BACKTRACK | 246/288 | **288/288** |
| Untrained hook/zigzag: reverse onto history → BACKTRACK | 144/288 | **276/288** |
| Trained continuation: false warning → permitted exploratory INK | 288/288 | 288/288 |
| Untrained continuation: false warning → permitted exploratory INK | 288/288 | **270/288** |

The 18 false-cue failures in unseen topology are a **negative result**. The INK criterion permits a branch exploration, not necessarily the shortest path to the goal. First-choice correctness is not full recovery.

## Host-applied stochastic evaluation

**4,608 episodes:** four families × 24 fresh seeds × both opposite personalities on the same geometry × three guidance strengths × two goal outcomes × cue/no cue × C7/C8; up to 24 real synthetic-host action applications per episode. Model and random stream start matched, but trajectories are not same-state comparisons after divergence.

| Cue-present goal outcome | C7 reached | C8 reached |
| --- | ---: | ---: |
| Trained topology, goal reversed | 144/288 | **130/288** |
| Untrained topology, goal reversed | 96/288 | **112/288** |
| All reversed | 240/576 | **242/576** |
| False-cue productive continuation | 552/576 | **541/576** |

Cue-off trajectories match exactly, with 826/1,152 goals reached under each model. This is an architectural gate guarantee, not improvement. Active-cue opposite-personality turn-bias gaps vary in magnitude and sign across families and arms: unchanged forward-action relative logits do **not** prove personality survives changed trajectories. There were zero raw illegal top-1 choices in held-out greedy starts, zero host-rejected stochastic actions, protected-material hashes were constant, and initial painted ink was never deleted.

**Conclusion:** C8 improves the previously unseen hook/zigzag first action and modestly improves unseen-goal arrival, but the overall recovery gain is negligible and false cues degrade productive paths. Not ready for native transfer or production default selection.

## Reproduce and qualification

From the repository guidance_recovery directory, using exact hashed prerequisite weights outside Git:

1. Run Python unittest discovery for test_*.py.
2. Run topology_transfer.py with --base, --c4, --c7, --out, --train-n 240, --heldout-n 48, --epochs 14, --rollout-n 24.
3. Run topology_gallery.py with the generated stochastic.json and an output trajectory_gallery.html path.

Two full local CPU optimizer/evaluator/gallery replays reproduced **byte-identical checkpoint, numerical metrics, first-action receipts, 4,608 full trajectory receipts and 16-panel source-reconstructed HTML gallery**. New module SHA256: f1533adb5129ab6d7b4f76d4c55143051e07aa64541de3b1bfd6b5e6918d926a. Metrics SHA256: f57dc9d3ee9abd059321ff6278c177fa72fe3cf2155152ec5d748e5cb97fd455.

**79/79 available offline regression tests pass** (71 C7-available plus eight new C8 tests). The previously omitted 11 C5 tests were not present in the available C7 archive, so **complete exact-HEAD test coverage is still outstanding**. The committed C8 Python files were matched independently to the locally executed Git blob SHA-1. Model weights, raw receipts and gallery remain off Git in a manifest-verified local bundle. Immutable Actions custody requires workflow-owner authorization and has not been achieved.

No native game/runtime/Workbench/contracts, shared workflow files, model provider defaults, paid calls, real material Undo or resident worm streams were modified or invoked. Preserve this model as a negative-tradeoff research checkpoint; next prioritize multi-turn correction context and independently novel false-cue geometries rather than claiming broad generalization.
