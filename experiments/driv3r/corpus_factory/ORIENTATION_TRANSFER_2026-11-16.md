# CORPUS-ORIENTATION-TRANSFER-v5 — executed source-pinned synthetic comparison

16 November 2026. Research only. Draft/HOLD PR #638. No CyberSand game/native/Godot/Workbench/provider changes.

## Research question

Does orientation-matched on-policy correction data (vertical combs rather than horizontal barriers) improve transfer to a **different, evaluation-only** vertical-slalom generator? The two students use identical source-label episodes, equal correction world-group counts, **616 unique correction observations per arm**, the same 24-epoch CPU log-loss SGD optimizer and the same host-legal action contract. Vertical comb TRAIN worlds are deliberately different geometries and generator rules from the held-out slalom worlds.

## Actual, reproducible run

Command: `python -m unittest -v test_transfer_orientation_v5.py && python transfer_orientation_v5.py --out <fresh-path>` from `experiments/driv3r/corpus_factory/`. Input source files are Git blob-pinned in the script; refusal on mismatches, existing output directory, duplicated split IDs, repeated geometries, mismatched world groups, or infeasible evaluation paths.

- 6 source TRAIN worlds, 2 shared original correction worlds, 2 horizontal or 2 vertical correction worlds per student; 4+4+8 previously unseen world seeds across original/horizontal/slalom EVAL.
- **1,626 original teacher decisions; 616 deduplicated corrections per arm; 2,858 weighted training rows per arm; 24 CPU optimizer epochs per arm**, plus 24 baseline epochs for correction collection. Each student sees the same shared correction worlds; neither receives evaluation labels.
- Each evaluation world has three family replicas, which are correlated and **not** independent experiments. World-level success is the unit of comparison.
- Oracle shortest-route feasibility was checked separately: 4/4 original, 4/4 horizontal and 8/8 slalom worlds achievable in the 192-action budget.

| World-disjoint EVAL | Horizontal corrections | Vertical corrections | Fully-informed oracle |
|---|---:|---:|---:|
| Original generator | 2/4 worlds solved | 3/4 | 4/4 |
| Horizontal-barrier generator | 4/4 | 4/4 | 4/4 |
| Unseen vertical-slalom generator | **3/8** | **2/8** | 8/8 |

On the unseen slalom, horizontal-trained student averaged **133.0 unique cells** and **13.4 two-step returns**; vertical-trained student averaged **49.7 unique cells** and **126.1 two-step returns**. Vertical correction exposure **did not improve** unseen vertical-slalom transfer in this controlled pilot; it worsened both exploration and cycling. The successful seeds differ (horizontal 9500/9504/9506; vertical 9501/9505), so the curriculum can alter *which* layouts work. These are eight independent worlds, not a reliable performance rank for population-level claims.

## Provenance, artifacts, limitations

- Synthetic 96×96 local material grids with blocked/protected cells, legal STRAIGHT/LEFT/RIGHT/BRANCH/RETRACE/STOP and goal-conditioned 5×5 observations. All decisions were host-applied; no illegal moves or protected-cell penetration accepted.
- Source names and exact blob pins are in `SOURCE_PINS` in `transfer_orientation_v5.py`; all seed partitions and geometry SHA-256 digests, complete per-family trajectories, optimizer statistics, negatives, original PNG paths, and artifact hashes are in the execution JSON.
- Two complete independent local runs reproduced byte-identical `report.json`, `train_horizontal.zip` and `train_vertical.zip`. Only TRAIN correction ZIPs are model-training eligible. **Evaluation trajectories must never enter either TRAIN ZIP.**
- This study is synthetic and small. It is **not** native CyberSand evidence, primary SLM checkpoint inference, official DEV/heldout qualification, or statistically significant superiority. Global-geodesic privileged oracle is a *teacher and upper bound only*, never student input. Host-facing integration proposals remain separate from game changes.
- Host-artifact publication in immutable GitHub Actions remains outstanding. Keep generated datasets and checkpoints out of Git history; publish source and small methods/report only. Do not promote Draft PR #638 or merge.

**Negative result retained:** superficial obstacle-orientation similarity is insufficient for cross-generator competence. The next discriminating experiment is matched correction-family count over multiple TRAIN seeds, plus explicit cyclical recovery vs spatial-memory feature ablations and 24+ independent evaluation worlds. Do not select curricula on this small repeated EVAL set.
