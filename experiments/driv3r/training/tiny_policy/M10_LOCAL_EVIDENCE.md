# TINY GAME POLICY M10 — two-head style/novelty router (local synthetic research only)

- Candidate: 3,937-parameter (24-wide) observation-responsive sigmoid router over immutable M8 novelty/style 20,549-parameter heads. **300 real CPU AdamW steps**; no M8 optimizer updates.
- Input: local 7x7 blocked/visited/protected W0 synthetic patch plus heading/family/time/geometry (162 floats); host legal action menu UP/DOWN/LEFT/RIGHT/STOP. The policy guard disallows voluntary STOP while movement remains legal.
- Frozen exact M8 checkpoint SHA256: novelty `68defcf794d57392d898bf217efe2fd5afe73f1b91a0303a702000b2ed4c39e2`; style `9a8f416c61d3bb3ee4e5c06ae0be0e566d9da02a2896a7c8526363481fc0033a`.
- Train 11,664 labelled states, seeds 44000-44011, shapes ratchet/branch-rail/windmill. Dev 3,888 states seeds 45000-45003. EVAL untouched during training: seeds 46000-46005, disjoint geometric designs double-ladder/nested-U/fan-spokes; 54 worlds (6 seeds x 3 shapes x 3 families) per controller.
- Nine controllers, 64 turns: **486** original PNG/JSON paired trajectories, **31,104** legal actions; 0 illegal, 0 early STOP across all controllers.

| Controller | Mean unique cells | Mean traversed-edge branch nodes | Family footprint IoU (lower = more distinct) |
|---|---:|---:|---:|
| M8 novelty | 58.17 | 2.33 | 0.825 |
| 25% style fixed mixture | **58.30** | 1.74 | 0.478 |
| 50% style fixed mixture | 57.46 | 1.91 | 0.463 |
| M8 style | 44.76 | 2.15 | **0.216** |
| **Trained M10 router** | **51.91** | 2.06 | **0.299** |
| Local style/novel teacher | 49.24 | 0.93 | 0.116 |
| Deterministic novelty greedy | 50.89 | 0.78 | 1.000 |
| Random legal | 24.52 | 9.98 | 0.300 |

The trained router is **6.39 unique cells worse than the simple 25% style mixture** (seed-cluster 95% bootstrap interval [-7.70,-5.07]). It achieves lower family IoU (0.299 versus 0.478), but **does not demonstrate superior overall utility**. The M8 novelty head remains better for pure coverage. Neither model makes paintings comparable to intentional authoring; the gallery illustrates small, repetitive tracks.

Independent local audit: 976/976 manifest hashes, 486/486 original PNG/state replays, 31,104/31,104 model+controller actions recomputed from source and loaded checkpoint, plus three adversarial checks. A second entire optimizer-and-evaluation run reproduced all **977/977** files byte-for-byte, including checkpoint bytes. Local source SHA256 `e125b623fd54d6c2f02e18fb8e987181d121ed9abbce69eb042be8bd6586f0f8`; checkpoint SHA256 `1f69046a143f83b68c5942a550d464a40ad55c30fe147690158e7cebda0c8001`.

**Qualification boundary:** source copy uses archived core/M5 bytes, which do **not** match the PR's current `core.py` / `m5_topology_train.py`; M8/M9 scripts are archived but uncommitted. Thus this is not exact-HEAD GitHub Actions or original CyberSand native evidence. No model should be promoted. Source-checkpoint provenance and the failed candidate remain in the downloadable immutable local bundle. Next action is source/host reconciliation and hosted source-matched baseline, *not* further optimization of this weak router on M10 EVAL.
