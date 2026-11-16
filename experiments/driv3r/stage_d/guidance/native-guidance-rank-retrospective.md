# Archived Q8 scalar guidance rank: null-calibrated retrospective

**Date:** 15 Nov 2026. **Evidence class:** read-only analysis of the original #611 paired native/Q8 source; no new model or native execution. The executable source is `native_guidance_rank.py` and its eight cheap adversarial tests are in `test_native_guidance_rank.py`.

**Original source:** `313b836b3c9ce2c60e4659e3763df140f84347a6`; approved M1 contract: `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6`, schema SHA-256 `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`. Original Actions [run 43678354515](https://github.com/techrote/cybersand/actions/runs/43678354515). Free ZIP artifact `14147579972`, SHA-256 `33083b5642c9ae6af04d7530147818453dd4513b9cf4e6f9576e4ef79e0d9aac`; soft ZIP artifact `14147233975`, SHA-256 `82d64e708e5706d4532900c60c52eaee7de04105d64888aaf436270136e7eca1`. Both underlying archives were independently native/readback and Q8-choice audited previously (see `archive-reverification-2026-11-15.json`).

## New finding — 13/36 is not an attraction effect estimate

Among the 36 actual **soft** MOVE choices, 13 select a legal destination tied for the highest 16-bit scalar. The appropriate trivial null *conditional on making a legal MOVE* must account for each turn's offered MOVE count and ties. Summing `number_of_maximum_ties / number_of_legal_MOVEs` over these actual 36 legal menus gives **15.333333333** expected maxima under uniform legal-MOVE choice. Observed minus this null is **−2.333333333**; chosen tie-aware guidance percentile is **0.476851852** (neutral 0.5). Mean chosen hint = **25,442** u16; mean offered legal-MOVE hint = **25,231.060185**. Mean original Q8 probability mass on highest-guidance legal moves, conditional on choosing a MOVE, is **0.406496169**.

The original **free** arm painted 33 moves and recovered 3 times. With its entirely zero-valued scalar, all 33 choices are trivially tied for maximum, as is the conditional uniform null (**33**); this cannot indicate guidance following.

At turns **0–4** (five matched native decision states and host legal catalogs), the two runs present identical geometry but differing scalar values. The Q8 choice remains identical through turn 3; turn 4 is BACKTRACK (free) versus MOVE_S (soft), the first consequential divergence. Largest original option-probability differences within those five matched turns were **0.039859567, 0.027618205, 0.036276578, 0.013830800, 0.034413313** respectively. After this divergence, the distinct native Worlds preclude treating later decisions as matched-state causal effects.

**Interpretation:** this particular seed offers no evidence that Q8 greedily follows highest guidance; the observed highest-hint count is even below the simple conditional uniform expectation. The probability shifts and eventual path divergence are interesting but are **not** a controlled demonstration of causal scalar responsiveness. They do not show that soft guidance is useless, nor prove any artistic preference. Presentation wording, chance, original model behavior, and subsequent state changes remain confounded.

## Reproduction and qualification boundary

```bash
python3 -m experiments.driv3r.stage_d.guidance.native_guidance_rank FREE.zip SOFT.zip
python3 -m unittest discover -s experiments/driv3r/stage_d/guidance -p 'test_native_guidance_rank.py' -v
```

The executable first invokes the lane's **entire existing original archive/native receipt/Q8 choice audit** and only then calculates the calibration. A local read-only reconstruction of the two original ZIPs and eight source-blob-matched unit tests produced the values above, but that local reconstruction substituted previously established archive custody for the actual committed `audit_pair` invocation; **the new complete command has not yet been executed on an exact committed full checkout**. Similarly, the full **199-test** D-1B suite and central M1 five-record integration conformance remain outstanding. No true visual d1 support (`--no-mmproj`), material Undo, critic execution, training, paid resources, production code or formal new A/B work.

**Suggested next controlled test after owner authorisation:** replay identical initial native states with legal catalogs/order frozen and vary only scalar values; measure the Q8 response distribution and the first consequential drawing divergence across multiple independently frozen seeds. Do not conflate differing post-divergence Worlds or a single Q8 argmax with a causal effect.
