# DG020 H22 — independent local post-hoc donor ablations (18 November 2026)

**Disposition:** H22's original, preregistered **restricted causal-selectivity PASS 30/30** remains unchanged. This is a new, **read-only post-hoc methods/falsification analysis** of the already exposed 4,800 H22 observation pairs. It is **not** a prospective H23 run, independent confirmation cohort or new scientific gate PASS. Keep PR #253 Draft/HOLD; never merge or mark Ready.

## Frozen inputs, custody and audit

- Original exact H22 source SHA-256: `a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d` (pre-registered on PR #253).
- Owner-handoff H22 evidence ZIP SHA-256: `5a60ca1119a22916bda18ed46c0c6acecf77b4d891f5b1b841854a81533d1ca0`. Full 30.3MB archive remains in owner's handoff, not GitHub Actions.
- Re-extracted frozen authentic H21R neural donor and H19 209-parameter receiver checkpoints; reran independently written H22 neural checkpoint/receipt auditor on **CPU PyTorch 2.10.0+cpu and NumPy 2.3.5**. Auditor matched 4,800 original observations, genuine/sham/dry-placebo/shuffled/zero/blind actions, all 30 acceptance cells and original 31/31 evidence-manifest hashes. This was **local**, not pinned exact-HEAD Actions.
- New independently authored read-only diagnostic `H22_METHODS_ABLATION_2026-11-18.py`, SHA-256 `96b43a00482998703f03ec328dd74da6719733ee6ee7ea1c915b72e543bbffe2`, replays H22 on those 4,800 *same* saved observations, preserving geometry, goals, wet masks, fees, legal actions, dry scalar anchor and receiver checkpoint weights in each comparison. The source, result JSON, complete replay log and manifest are in the owner handoff `DG020-H22-posthoc-methods-ablation-20261118.zip` (13,332 bytes, outer SHA-256 `f673f45cd03f9ace7875eac841f61b75a61e8ced48bc8eb226623b5d10fd392c`).

## Quantitative falsification

| Post-hoc wet-only inference intervention | Divergent both-direction correct /1,920 | Invariant both-law correct /2,880 | Wrong invariant flips /2,880 |
|---|---:|---:|---:|
| **Genuine trained donor forecasts (frozen H22)** | **1,920** | **2,877** | **3** |
| Replace expensive-law wet signal with cheap-law wet signal (no law swap) | **0** | 2,880 | 0 |
| Reverse the two genuine-donor wet-law signals | **0** | 2,877 | 3 |
| Untrained randomly initialized donor pair (seeds 727001/727002) | **0** | 1,668 | **180** |
| Known exact wet law costs (oracle-informed **diagnostic ceiling only**) | **1,920** | **2,877** | **3** |

The randomly initialized donor pair is **one negative control**, not a representative random-donor population. The known-law input is intentionally oracle-informed and **may not count as learned-donor success**. The zero/no-change and reversed signals prove that a *correctly directed* inherited wet-material signal matters in the fixed two-route task.

**Mechanistic finding:** On all **9,600 individual law-arm decisions**, substituting direct known-law wet scalar constants for genuine trained-donor forecasts produced **zero action disagreements** (the logits differed by at most `0.02368927`). Genuine learned primary source wet scalar estimates were `0.35014015` (cheap) and `2.30236959` (expensive) compared with exact synthetic wet costs `0.35` and `2.30`; the anchored dry scalar was `1.0012010`. Thus the successful conditioned receiver effectively uses a near-exact **supervised scalar material-cost rule**, not demonstrably a richer hidden law. The **same three** H22 near-tie equal-wet invariant errors persist even using exact known costs, localizing them to receiver ranking errors on these observations.

**Methods limitation:** For divergent worlds the two correct oracle actions are opposite, so any identical-action zero/source-blind control must obtain 0 both-direction-correct by construction. Those controls establish need for a law-dependent signal, but are less informative than the pre-inference shuffled, independently trained same-law source sham, dry-only placebo, and invariant selectivity controls.

## Scientific boundary and smallest next step

H22 still **passes its locked narrow gate**, not a general material-law transplant. The task is two preselected wet costs, supervised donor training, an explicit 5×5 material-specific quadrature, fixed 2×12 routing geometry, and constructed decision categories. No weight-level edit of a single donor, arbitrary geography or CyberSand integration is demonstrated. Retain earlier H17/H18 FAILs and H22's three real negative cases. All controls here reused exposed data and are therefore **diagnostic only**.

Priority: durable exact original H19/H21R/H22 source and frozen-checkpoint custody in this existing Neurounner lane, then pinned **exact-HEAD CPU** Actions checkpoint/receipt replay with published immutable artifacts, followed by genuinely external methodological review. Do not expand laws, models or maze complexity pending qualification. No new PR/issue/branch, paid GPU/API or CyberSand/DRIV3R write.
