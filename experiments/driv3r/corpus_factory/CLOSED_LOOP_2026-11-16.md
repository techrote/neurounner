# Family closed-loop synthetic corpus — 16 November 2026

**Scope:** `experiments/driv3r/corpus_factory/**`. External standard-library Python only; no game, Workbench, native, physics, materials, asset or provider mutation. Draft PR #638 remains HOLD. **These are actually executed synthetic host-legal walks, not native PAINT, model inference, learned-model updates or architecture-planning demonstrations.**

## Source-pinned execution

- Git source blob `family_closed_loop.py`: `4430d92bfb4a27e54445c33d537b893d99ffd4f9`; source-file SHA-256: `b72cc77f1935f69ef2602a703fa8041ad989b14ca7399eede5e7a941006618a3`. The source manifest records that SHA-256.
- Git test blob `test_family_closed_loop.py`: `b90c1564b9352dbfcce276f8de0caa31f799c857`.
- Local exact-source `python -m unittest -v test_family_closed_loop.py`: **6/6 PASS**. Tests cover 128-step seeded reproducibility, legal applied/receipted movement and protected-cell safety; matched randomness and label-only ablation; strict splits and prompt-safe TRAIN; forged/rehashed source receipt replay; forged/rehashed TRAIN-eligible label; input bounds.
- Independent data audit reruns all 180 generated episodes from world seed and controlling family; checks world ledgers, legal original receipts, eligible-row identity, world-group split, 5×5 input collisions, and equal-world footprint metrics. **PASS**, 60 world groups, 49 inventoried files.

Run from `experiments/driv3r/corpus_factory`:

```sh
python -m unittest -v test_family_closed_loop.py
python family_closed_loop.py --output /tmp/cf_family_v1 --worlds-per-scenario 10 --budget 192 --image-worlds 2
python family_closed_loop.py --output /tmp/cf_family_v1 --audit-only
python family_closed_loop.py --output /tmp/cf_family_v1 --export-train /tmp/cf_family_v1_train --per-episode 16
```

## Actual observed outcomes

| Measurement | Exact result |
|---|---:|
| Independent 96×96 world groups | 60 (48 TRAIN, 6 DEV, 6 held-out) |
| Six scenario families / actual closed-loop family episodes | 6 / 180 |
| Original source receipts / applied synthetic actions | 34,560 / 34,560 |
| TRAIN full receipts / deduplicated eligible | 27,648 / 27,167 |
| DEV full receipts / evaluation-eligible | 3,456 / 3,406 |
| Held-out full receipts / evaluation-eligible | 3,456 / 3,385 |
| Cross-split observable collisions quarantined | 121 |
| Within-TRAIN duplicate prompts dropped | 481 |
| Original drawing PNGs generated | 36 |
| World groups with three distinct painted-cell footprints | 60/60 |
| Label-only ablation, fixed pipework steering, three identical footprints | 60/60 |
| Mean pairwise painted-footprint Jaccard, matched world | 0.072 |
| Procedural budget STOPs | 180/180 |
| Reachable spatial goals actually visited | **0/180 — negative result** |

The family effect survives a *matched stochastic stream*: three action-scoring modes on the same seeded world generate distinct applied drawings, while changing the **observation family label only**, with pipework steering fixed, leaves identical footprints. Greebles produces local wandering/branches; pipework generally longer sparse orthogonal runs; architecture more terraced/intermittently branched traces. This **does not show** that any trained model understands the family instruction. The procedural teacher sacrifices pathfinding: zero goal visits despite generated goals being reachable. Do **not** train goal-success behaviour from these style runs as if it were successful; retain the older v1 goal/recovery curriculum and independent DEV tests.

## Optional strictly TRAIN-only message export

- **2,303** prompt-safe messages from **48 independent TRAIN worlds**; each has a separately indexed original record provenance line.
- Label histogram: STRAIGHT 772, LEFT 569, RIGHT 430, BRANCH 331, RETRACE 170, STOP 31.
- `train.messages.jsonl` SHA-256: `a98a9d81027fe8a3095ab85e32a02786be626da1953fa3da4730f52dde5ed244`.
- `train.provenance.jsonl` SHA-256: `c4330ef63d7422ebbc827c0257f42c8f4df4c0ee63069b5c3a680e354355111a`.
- Prompt-visible allowlist: position, heading, local symbolic/material/protected grids, legal actions, recent actions, family, tick, budget, goal delta/reached. **No** entire source rows, receipts, optimizer scores, hidden solutions, negatives, world ledgers or any DEV/held-out record may be passed as a model input.
- Original source and TRAIN export are distinct directories with immutable-output refusal and per-file SHA-256 inventories. Local results remain locally qualified; **GitHub Actions archival/hosted exact-source execution is not yet complete**.

## Boundaries and research handoff

The standalone world generator is a new source-pinned **synthetic variant**, informed by read-only v1 corpus and W0 action semantics; it is not a faithful Godot material solver or a native receipt. `BRANCH` takes the first fresh host-legal right/left/straight arm, `RETRACE` pops a real preceding pose, and `STOP` occurs at the fixed 192-decision budget bound rather than as autonomous learned continuation. Protected/obstacle cells remain non-traversable. The two score teachers are explicitly **heuristics**; no model checkpoint or optimizer output is produced.

The most useful next controlled trainer experiment is **family-style model response on equal-seed, non-TRAIN worlds**, compared against this source-pinned procedural reference, with separate goal/recovery scoring. A classifier's one-step family accuracy or three coloured PNGs alone is not proof of learned closed-loop steering. Preserve negatives and do not contaminate DEV, held-out or original W0 quarantined evidence.
