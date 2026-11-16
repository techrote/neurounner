# D-2 independent post-freeze structural robustness study (15 Nov 2026)

**Evidence class:** `POST_FREEZE_EXPLORATORY_SYNTHETIC_ONLY`. **PR #598 Draft/HOLD.** This is **not** a native World, model, paint-receipt, physical-strength or visual-art acceptance result, and does **not** qualify the original eight-case frozen confirmation policy.

## Chronological input custody

1. New input-only deterministic source `holdout_fixtures.py` was committed at `77713e3d0e715c1557ae87805b12fc85cb9ce1ad`, source Git blob `d2a3dec9df4984c2cd139e0020150d76bca4a449`.
2. **Before policy results were generated**, twelve novel 32×32 geometries, seeds **21001–21012**, positive u16 scalar fields, hard-permission masks, actual material hashes, three policy arms and 20-edit / radius-3 limits were frozen in `corpus/holdout-prereg-v1.json`. The corrected exact input freeze is commit **`65d4b8e477cccedfe8b9a25f0617d20ead0c820d`**, blob **`d1f89893be15170aee5f0871bcb66043ce203be9`**, SHA-256 **`7160002b2cee9f9d97210acff17afa11326f8dcb6fdaa8bfe78cd13d8c0d109d`**. The correction fixed a two-character transcription error in one hash **before** evaluation.
3. `holdout_study.py` was added **after input freeze**; it fails closed on manifest or Git-blob mismatch, checks generated mask/guide/material hashes, compares the committed reinforcement implementation with equal-actual-cell guidance-first and seeded random-frontier arms, preserves all hard permissions and original material, and writes the original and all three resultant **32×32 RGBA PNGs** per case.

The actual locally executed Git blobs match branch files for `holdout_fixtures.py` (`d2a3dec9`), `holdout_study.py` (`d15d4758`), `test_holdout_fixtures.py` (`c2c39e49`), `test_holdout_study.py` (`1773fd7a`), `reinforcement_study.py` (`fb4a1f20`), and `fragility_study.py` (`cb1fdf1c`). The post-freeze policy is source-qualified in this study only; the earlier archived eight-case policy remains **unverified**.

## Research results

Twelve structurally distinct input instances spanning seven explicitly defined geometry families, with one no-write/permission-closed negative control. All cases were retained, including STOP and disconnected outputs. **Eleven cases made 20 edits per arm (220 edits/arm total); the negative control made zero.** All arm edits were matched within each case, not merely equalised by maximum budget.

| Exact synthetic topology metric | Initial | Reinforcement | Guidance-first | Seeded random frontier |
| --- | ---: | ---: | ---: | ---: |
| Sum of 4-neighbour material articulation points, all 12 cases | 879 | **658** | 895 | 898 |
| Newly edited cells below soft preference threshold | — | 114/220 | 0/220 | 113/220 |
| Sum of filled 2×2 material squares | — | **275** | 69 | 95 |

Reinforcement produced fewer articulation points than **both** controls on each of the eleven editable instances; it did not invent success on the no-permission negative control. On crossbars-A, 56 initial articulation points became 37 with reinforcement, versus 59 guidance-first and 60 random-frontier. On gated-choke, 66 became 47, versus 70 and 70.

**Important failures / scope:**
- Staircase-A finished with **five** disconnected material components, staircase-B with **three**, and each perimeter case with **three**. Minimising articulation points within components does not connect islands.
- The disjoint-blocks negative control has five components, zero legal new cells, and correctly STOPs without painting.
- Reinforcement used below-threshold u16-guidance cells for **114/220** additional edits, compared with zero for guidance-first: topology robustness can conflict with preferences.
- A 32×32 synthetic raster is not a representative playable scene; filled 2×2 squares, articulation counts and mask correctness are **proxies**, not mechanical stability, playable continuity or artistic quality. Geometry families were devised with knowledge of prior D-2 limitations; they are new and preregistered, but not independent blind artistic validation.

## Reproduce offline

From `experiments/driv3r/stage_d/procedural`:

```sh
python -m unittest -v test_holdout_fixtures test_holdout_study test_reinforcement_study
python holdout_study.py --manifest corpus/holdout-prereg-v1.json --out /tmp/d2-holdout
```

Executed locally against matched committed blobs: **20 test methods, 20 passed with optional historical provisional archive present; without that archive, 18 passed + 2 explicitly skipped**. No full historical D-2 exact-head test suite was executed. Repeated full report generation including original PNGs was byte-identical; `holdout-results.json` SHA-256 **`7189986fbc955c041e9da3492a87aeb2d1c87f43de886182d548cdfbb6d9655a`**. These were *local* offline tests, not GitHub Actions job results.

The input freeze and methods can be reproduced entirely from this branch without any GitHub Actions experiment matrix, inference, GPU or paid service. All files in this milestone are under the assigned D-2 procedural directory.

## Outstanding gates

The original eight-case freeze still requires full exact-head D-2 Python qualification and an actual `pareto_study.py`-blob, source-verified regeneration of the **original preregistered** cohort; the previous shadow archive must not be promoted. Further study should test width/route realism and material-island reconnection with actual host-owned receipts **only** following separate owner approval. No native D-2 integration, material Undo, workflow matrix dispatch, model inference, training, merges, Ready or provider changes were performed.
