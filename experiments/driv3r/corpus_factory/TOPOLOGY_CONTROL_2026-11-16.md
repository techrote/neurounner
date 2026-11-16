# GAME CORPUS FACTORY — size- and world-matched topology ablation v4

**Classification:** external synthetic research, TRAIN-internal controlled comparison; NOT native CyberSand, official DEV/held-out, primary Sprocket/LFM, or proof of artistic quality. Draft/HOLD; no production or game files edited.

## Question

The v3 mixed-topology student improved from 0/12 to 3/12 unseen slalom worlds but also trained on 1,884 correction states versus 1,075 original-only states. Is there a directional benefit after matching correction count, number of correction worlds, optimizer settings and source pool?

## Actually executed (2026-11-16)

- Same 24 seeded source TRAIN worlds; 6,436 host-applied synthetic teacher rows, 18,082 action-level legal hard-negative distractors; same 24-epoch baseline SGD source student.
- Independently gather corrections on 12 original-only worlds and 6 original + 6 staggered-barrier worlds. Six original correction worlds intentionally shared between arms, never evaluation. Disjoint source and evaluation seeds/geometry.
- Apply source-aware and cross-family normalized-state deduplication. Extract **exactly 1,145 unique correction states per model, drawn from 12 distinct world groups each** using deterministic world-balanced hashed selection. Domain mix and action/history distributions are recorded exactly in the JSON manifest.
- Both fitted correction students use the same 6,436 source rows, 1,145 additional states repeated twice, identical optimizer settings, 24 real CPU epochs, 363 features and 8,726 weighted rows each.
- Three independently seeded synthetic evaluation domains: original (4300–4307), horizontal barrier (6300–6307), vertical slalom (9300–9307), each × three correlated family labels × two fitted policies = **144 host-applied closed-loop inference trajectories**, each capped at **160 actions**. All 24 distinct evaluation worlds are world-group-disjoint from TRAIN. Slalom geodesic at 9300–9307 is 136 steps, so an earlier 128-step pilot was **invalid as a goal-success test**. Source now *fails closed* if any evaluated world has shortest oriented route longer than decision budget.

## Matched evaluation (successes are independent worlds, not family rollouts)

| Synthetic geometry | Original-only 12-world correction | Mixed 6 original + 6 barrier correction |
| --- | ---: | ---: |
| Original | 8/8 | 8/8 |
| Horizontal barriers | **8/8** | 7/8 |
| Vertical slalom | 0/8 | **1/8** |

Vertical slalom exploratory behaviour: unique cells **33.542 → 70.333**, mean two-step returns **110.583 → 70.292** (original-only → mixed). The mixed model moves more on previously unseen geometry but is **not broadly goal-competent**. On horizontal barriers, the matched mixed model performs **worse** on world-level success. No causal gain in general-purpose goal success is established, and the 8-world/domain sample is small. Action-label and history distributions still differ despite sample and world-count control.

## Qualification, reproducibility and gating

Six focused regressions PASS, including deterministic world-balanced selection, no duplication, legal targets, grouped split overlap rejection, pinned parent source, and reject-impossible evaluation budgets. Two independent complete runs produced byte-identical `report.json` SHA-256 `ca30fd86898c9c4b0b02dcb9b14fb7d72f0e68a3d5e4fcc08b402f71fb731baa`, original-only TRAIN ZIP SHA-256 `acce3c9e6017b69b2d106a0d64bf9d3a7eb7e9785e87cae88bc5b6056317310a`, and mixed TRAIN ZIP SHA-256 `6ed133fb7cef8e9aa181f791033a01e262c018df5348864a831ca556946985af`.

Reproduction (Python 3, NumPy, scikit-learn; CPU only, no API or external GPU spend):

```bash
python3 -m unittest -v test_topology_control_v4.py
OPENBLAS_NUM_THREADS=1 python3 topology_control_v4.py --out UNIQUE_EMPTY_DIR
```

The Python files must be beside the source-pinned v3 dependencies in `experiments/driv3r/corpus_factory/`. Only correction TRAIN ZIPs are model-training eligible. Evaluation trace, report, PNGs and trajectories are quarantined from any trainer. All candidate model weights remain local/in-memory; no checkpoint promotion. **Actions artifact not yet published**. Do not merge, mark Ready or modify game code.
