# M9 — 96-turn source-pinned local endurance and protected-gate stress (16 Nov 2026)

**Draft/HOLD, synthetic research only. No game-side edits, no optimizer, no merge.**

## Experiment actually executed
- Exact locally executed script SHA-256 `2d4c03e8e71059b66cebc019bc6391970a75e52d5b856aa2186f5fe68490834a` (`m9_endurance.py`, provided in the original evidence ZIP, not yet committed). SHA check covers the corrected loader where each M8 MLP receives a **different model instance**; a preliminary loader reused an MLP object and invalidated its results, and the preliminary outputs were discarded before the final full rerun.
- **12 previously untested world seeds 36000–36011 × 2 obstacle variants (baseline / additional deterministic protected gates) × 3 family labels × 9 controllers = 648 genuinely executed 96-turn rollouts, 62,208 original legal-action selections.** Source-original 96×96 PNGs and action-by-action state/observation/legality receipts retained; no new AdamW steps.
- Host action strings: `UP DOWN LEFT RIGHT STOP`. All four trained policy families load the original immutable M5/M8 head checkpoints; M8 novelty/style MLP are separate real forward instances. M5 MLP, M5 conv, three M8 trained heads, teacher-style, novelty greedy, wall, random compared at identical world, protected geometry, budget and initial position.
- The M8 six shape-generators are reused with independent seeds; therefore **new seeded world instances, not novel generator families**. Added seeded protected gates stress access paths. The local archived `core.py`, `m5_topology_train.py` sources **differ from current repository blobs**. No exact-HEAD GitHub Actions run and no native CyberSand qualification.

## Actual matched 96-turn results

| Policy | Mean cells | Mean branch nodes | Mean retraces | Gated cells | Early STOP |
|---|---:|---:|---:|---:|---:|
| Novelty greedy | **67.04** | 0.92 | 30.0 | 63.42 | 0/72 |
| **M8 novelty MLP** | **65.72** | **3.22** | 31.3 | 62.89 | 0/72 |
| M8 style MLP | 56.85 | 1.50 | 40.2 | 42.64 | 0/72 |
| M5 MLP | 55.88 | 2.01 | 41.1 | 48.22 | 0/72 |
| Wall deterministic | 53.58 | 0.46 | 43.4 | 42.58 | 0/72 |
| M5 conv | 49.79 | 1.61 | 47.2 | 35.06 | 0/72 |
| M8 style conv | 48.39 | 1.71 | 48.6 | 42.86 | 0/72 |
| Teacher-style synthetic | 59.62 | 1.99 | 37.4 | 48.50 | 0/72 |
| Random legal | 30.06 | 10.14 | 66.9 | 30.14 | 0/72 |

Each policy: 12 seeds × 2 variants × 3 families = **72 trajectories**. All retain negative/retraced behaviour.

Seed-cluster (12 distinct seed groups) 95% bootstrap intervals for coverage differences:
- M8 novelty MLP − novelty greedy: **−1.32 cells** [−9.76, +7.29]; no established superiority.
- M8 novelty MLP − M5 MLP: **+9.85** [−1.64, +20.88]; positive mean but CI crosses zero.
- M8 style conv − M5 MLP: **−7.49** [−15.19, −0.56]; underperforms.
- M8 style MLP − novelty greedy: **−10.19** [−20.61, −0.42].

Cross-family mean painted-footprint IoU: M8 novelty MLP **0.8796** (44/72 pairs identical action traces); M8 style MLP **0.3273** (6/72 identical); M8 style conv **0.3587** (11/72); deterministic novelty **1.000** (72/72 identical). **The best learned coverage is largely family-invariant.** Extra branch nodes are insufficient evidence of a better painting.

## Audit and evidence
- Independent disk audit read and SHA-256 verified **1299/1299** evidence manifest entries, replayed **648/648 original state/observation/legal action receipts**, applied 62,208 legal actions, and compared **648/648 original PNGs pixel-for-pixel** to the replayed synthetic-host canvas.
- Complete second evaluation reproduced **1299/1299** evidence files byte-identically; independent audit of receipts includes geometry and checkpoint SHA controls.
- Original evidence ZIP SHA-256: `f68a9a98b42ae0fc66c7943ed3c710ceda581a5204aefdbc99c6c0d18274eb4b` (`tiny-policy-m9-endurance-evidence.zip`, attached to the originating ChatGPT automation conversation). Contains original checkpoints, archived execution dependencies, M9 executable, independent audit, five adversarial regression tests (5/5 PASS), all 648 original PNG and JSON receipts, complete manifest and visual gallery.

**Decision:** M8 novelty MLP is a **research lead**, not a promotable CyberSand gameplay/authoring policy. Retain adverse gait, retraces, weak family-style separation and source-exact limitations. Next action: commit exact tested M8/M9 source and qualify it against current-HEAD dependencies and hosted immutable artifacts; then genuinely new topology families for blind promotion tests. Never merge/mark Ready without owner approval.
