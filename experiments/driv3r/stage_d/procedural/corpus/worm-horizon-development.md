# D-2 DEVELOPMENT — input-frozen sustained escape-horizon drawing worm

**Status:** deterministic **offline synthetic only**; not native, learned, independently held-out, or human aesthetic evidence. **Draft/HOLD; no native/model/Undo/Actions experiment dispatch, training, paid computation, or merge.** Source remains wholly inside D-2's procedural ownership.

## Input-first provenance

- Input-frozen commit: `d754242a98f38a1f95f96eb8104d36feb6e42eae` (created **before** any of the new five-arm outcomes were run); immutable `worm_horizon_fixtures.py` blob `9ebd7af5d72245b26caf737997ee14f3507b5a10`, input test blob `351c12d868863a5192a3d5f8f7f7e7027c19810a`.
- Preregistered input manifest SHA-256 (canonical UTF-8 JSON): `f0e0446d0fc200c36b034b9d55277c51f5272066420ed89099154d9bdf56e3e6`.
- Twelve independent original inputs: three legacy environments (open/slats/ring) and three new masks (canals/baffled/crucible), seeds 23 and 79. Explicit identical hard geometry, origin, soft scalar u16 guide, pen rhythm, step budget (192), and tick-indexed random draws across all arms. Each case embeds the original hard-mask and scalar hash.
- Five policy arms: prior strong-attraction `x16`, exact-parity prior short hysteresis, a **new sustained-escape** intervention, a sham sustained-escape control, and an extreme always-novel control.
- The sustained arm uses a 12-position revisit window (trigger at ≤10 unique), 12-step minimum escape, 12-fresh-cell exit or 64-step timeout, 12-step cooldown; *during escape only*, attraction gain drops to 0 and visited-cell penalty rises by 200. No new hard permissions or unseen native shortcuts are introduced.

## Offline results (DEVELOPMENT, descriptive only)

Totals are summed over twelve episodes per arm, each capped at 192 legal moves. The **same inputs and independent per-tick RNG** permit paired descriptive differences; trajectories after the first differing action are different Worlds and later action catalogues need not match.

| Arm | Distinct painted cells | Repeated cursor visits | Turns | Escape activations / escape-weighted steps |
| --- | ---: | ---: | ---: | ---: |
| Strong attraction ×16 | 1,367 | 789 | 680 | 0 / 0 |
| Earlier short escape | 1,406 | 732 | 660 | 27 / 187 |
| **Sustained escape** | **1,503** | **618** | 670 | 22 / 275 |
| Sham sustained (no score intervention) | 1,367 | 789 | 680 | 25 / 0 |
| Always novelty-biased | 1,979 | 39 | 519 | 0 / 0 |

Against strong attraction, sustained escape adds **136 painted cells (+9.95%)**, reduces repeated cursor visits **171 (-21.7%)**, while retaining 670 turns versus baseline 680 (-1.5%). Across pairs: **10 improved, 1 worsened, 1 identical** in painted cell count. Notable negative cases: `baffled/23` lost seven painted cells and `baffled/79` never triggered. The sham exactly matches the original *trajectory and material* on all twelve cases, despite recording its internal trigger/exit events. Always-novel covers more cells but substantially changes the squiggly line character, notably fewer turns.

A simple mean chosen-cell guidance advantage over legal-catalogue mean falls from ~286 u16 units (strong) to ~240 (sustained), versus ~91 (always-novel), suggesting a trade-off between following scent and exploring fresh areas. These values are directional synthetic proxies, **not calibrated steerability, aesthetic ratings, or proof of neural/model benefit**.

## Qualification and limitations

- **4/4 input-only and 7/7 new policy tests pass locally**; 22/22 frozen legacy worm tests also passed locally. New-policy tests compare **all six legacy scene/seed pairs against their exact original strong, short-escape, and always-novel trajectories and material**. Twelve sham pairs are bit-identical in paths/material.
- Thirty sampled full-event episodes check cardinal legal movement, actual synthetic material readback/hash chain, explicit pen states, no native receipts and refusal to execute Undo. All 60 episode recipes, original hard-mask views, scalar fields, events, full trajectories, result images, metrics and hashes are deterministically regenerated and independently compared to archived bytes; altered view plus rehashed manifest is still rejected.
- Twice regenerated corpus index SHA-256: `41f1148a76304bd3c1f543c436319878ace8aacad061492bb75d5df731b5f262`. Both new source/test Git blob IDs must match the locally tested files before qualification can be promoted: `9527a438165e13d024ae11966eaf0e7a851dbd6c` and `61140351fc47ed29f121465cc5d00b9640b4e1f4`.
- **No aesthetic preference ranking was collected** and all cases are DEVELOPMENT. Novelty, more ink and fewer repeats are not objective improvements in line art. An owner/human should compare original PNG panels blind before selecting a default. The frozen older eight-case source-policy confirmation and full exact-head D-2 suite remain separately outstanding.

## Reproduction

From `experiments/driv3r/stage_d/procedural/` with Python stdlib:

```bash
python -m unittest -q test_worm_horizon_fixtures test_worm_horizon_study test_worm_sketch test_worm_scent_study test_worm_escape_study
python worm_horizon_study.py --out /tmp/d2-worm-horizon-development
```

The emitted artifacts are local only; the 2/4/8 opt-in matrix remains disabled and was **not dispatched**. Do not infer authority for cross-lane/native/model integration from these synthetic observations.
