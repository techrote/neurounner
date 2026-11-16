# D-2 DEVELOPMENT — small wayward drawing-worm sketchbook (15 November 2026)

**Evidence class:** `DEVELOPMENT_SYNTHETIC_ONLY`. **PR #598: Draft/HOLD.** This is not a model outcome, an actual CyberSand Workbench capture, a native material receipt, or a human aesthetic rating. No native/model/API/GPU/matrix job was dispatched.

## Owner-intent correction and scope

The 15 November owner clarification in [issue #570](https://github.com/techrote/cybersand/issues/570#issuecomment-6889195672) sets the hypothesis as a **slightly deranged but steerable line-drawing worm brain**. Successful line art need not minimise articulation points or match shortest paths. Existing bridge-strength experiments remain engineering topology diagnostics; neither their one-cell articulation penalties nor strong A* performance should be imposed as an aesthetic objective.

This **independent post-freeze** synthetic sketchbook adds `worm_sketch.py` without touching the frozen eight-case policy, the 12-case structural cohort, M1, native, #568, or another research lane. A deterministic `wayward` four-neighbour controller makes mildly persistent, soft-guide-responsive, seed-dependent, sometimes irregular turns. Compare it against `stubborn` (direction persistence) and `random_legal` (seeded uniform cardinal choices). Each arm has exactly the same original hard-permission geometry, positive u16 scalar field, cursor, seed, 192-step cap, and periodic four-step pen lift. The policy cannot override permissions.

Three 80×80 scenarios (open, slit walls, chamber ring) × two seeds (23,79) × three policies = **18 DEVELOPMENT trajectories**. Save before-material/geometry and after-material synthetic PNG, exact brush-up/brush-down steps, ordered offered legal candidates, per-step before/after synthetic material hashes, original u16 guidance, recipe, manifest and replay. No simulated material Undo: `UNDO_MATERIAL_EDIT_UNAVAILABLE` is explicit. Pen-up moves preserve the painted set. No fake native or model receipts.

## Locally observed descriptive statistics

| Policy | Cases | Turn-count range | Median turns | Distinct final painted cells | Episode lengths |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stubborn straight bias | 6 | 3–5 | 4 | 168 each | 192 each |
| Seeded random legal | 6 | 78–141 | 109.5 | 34–92 | 108–192 |
| Wayward steering | 6 | 30–44 | 38 | 161–167 | 192 each |

All six `wayward` trajectories have distinct trajectory hashes, as do the six random trajectories. The low-turn straight controller has only three distinct trajectories across scenes and seeds. Three random seed-79 episodes terminate early on the explicit repeated-local-oscillation STOP condition; they remain in the denominator. The output gallery shows longer purposeful-looking but sometimes awkward loops in the wayward arm, versus near-rectangular perimeter strokes for stubborn and small turbulent clusters for random. This is **descriptive visual inspection only**, not a quantitative preference or proof of neural benefit. The soft guidance and the chosen turn score are engineered developmental preferences, **not** a calibrated reward or a model.

## Qualification and custody

- **8/8 new local stdlib tests passed** under Python; all 18 development cases generated and all regenerated from recipe with byte-identical original/material PNG, scalar guidance, trajectory and event files; tampering with a PNG then rehashing its manifest was rejected.
- Both locally executed source Git blobs match the committed branch: `worm_sketch.py` **`b419282d2556e20cc7ca26e62f5dea3c2827dc0a`** and `test_worm_sketch.py` **`c50bbaf711cf3fddafcea6d71815d7eca3c2fa36`**. This qualifies the **new standalone files locally**, not the complete exact-head repository suite.
- Original 18-case synthetic index SHA-256 **`c71789cec97c2bd7ba1bb9a122009b021ca052b68d9e85511335ca25bc6d8f77`**; no generated raster/report committed to Git.
- Reproduce: `python3 -m unittest discover -s experiments/driv3r/stage_d/procedural -p 'test_worm_sketch.py' -v` and `python3 experiments/driv3r/stage_d/procedural/worm_sketch.py --out /tmp/d2-line-worm`. All fixture outputs remain development-only.
- Frozen M1 pin remains owner-approved `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6`, schema SHA-256 `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`; no modifications to the contract or approval status.

**Open qualification:** complete exact-head D-2 suite, independently source-verified regeneration of the *older* eight-case frozen confirmation cohort, original-gallery human evaluation, and any real d1-2B/natively accepted drawing-worm sessions. Native/model work remains a separate owner gate; the existing M3/M4 approval for other lanes is not inherited here.
