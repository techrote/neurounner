# DG020 H26 — prospective near-margin counterfactual selectivity (2026-11-18)

**Restricted scientific gate: PASS, 30/30 preregistered nonpooled cells, local genuine-checkpoint CPU replay.** This does not demonstrate general physical-law inheritance or game competence.

## Protocol lock and ownership

- [Immutable-timed PR preregistration, before H26 holdout generation or evaluation](https://github.com/techrote/neurounner/pull/253#issuecomment-6980705287). H26 training had already finished; acceptance worlds had not been generated or used.
- H26 source SHA-256: `94ca0ccbd2f9105ecdc7abe0c95e2a67ad9fbb84deeeb94079dea1339a662540`.
- H25 dependency SHA-256: `b1a385dc4c9bceea4aa5b3bf2f8f22382db1e9f0af7b646b755f631815a00853`; H23 dependency SHA-256: `41475d535ce4005e1f94c57385891b2f90b87f01e5f265d4ee0eec482bfb8358`.
- Original H25 genuine receiver checkpoints checked before training. Original H21R genuine donor checkpoint manifest checked before inference.
- Newly trained 209-parameter H26 receiver checkpoint SHA-256 values:
  - seed 26011639: `55427ff362e971603414ff2883cfe203a99fa5fe739c531042cff4477c6a8bd6`
  - seed 26011640: `dbc83bad61db889966bf233276dab6f817aa71ed316b19173384ff5cf7e8aff7`
  - seed 26011641: `d8b4c691cb3e490907036d8f9225f315102aee114e0270d60b262fdcc64338ad`
- Single bounded training correction: append training-only near-tie fee perturbations (minimum true margin 0.003–0.035) to original H25 training families; 1,400 CPU AdamW updates, LR 0.00007, exact original H25 tile/absolute/delta/equal-wet/trust loss coefficients; original 209-parameter architecture; same two donor law values 0.35/2.30, two 12-tile routes.
- Geometry, texture, fees, goal, legal mask and receiver weights held fixed within paired law arms; **only donor-derived inherited wet-material cost coordinate changes**. Training oracle targets are excluded from inference.

## Predeclared H26 holdout and hard gate

The two new synthetic wet-position families `quad_stripe` and `interleave_rim` each contain five original counterfactual buckets, 160 new pairs per family/bucket/seed, three receiver seeds, 4,800 total. Fee-profile distributions remain the two earlier Beta families; no increase in complexity or law families. Oracle minimum decision margin is 0.008–0.030, more difficult than H25's original 0.06–0.19 range.

Per cell: divergent both correct ≥152/160, required flips ≥157/160, each law correct ≥157/160; invariant both correct ≥157/160, inappropriate flip ≤1/160, each law correct ≥157/160; same-law independent donor-source changes ≤1/160; equal-wet p99 route-margin drift ≤0.010; illegal decisions 0. Donor forecast MSE ≤0.020 globally. Zero/source-blind and shuffled-donor divergent controls retained as **limited-power sanity checks**, not independent proof.

## Locked prospective outcome

| Statistic | Value |
|---|---:|
| Independently scored cells | **30/30 PASS** |
| Divergent pairs, correct under both laws and flipped when needed | **1,920/1,920** |
| Invariant pairs, both-correct and no law-induced change | **2,880/2,880** |
| Incorrect invariant flips | **0** |
| Foreign trained same-law donor action changes | **0/4,800** |
| Worst equal-wet p99 absolute margin drift | **0.000006114** |
| Maximum paired margin-additivity residual | **0.005810** |
| Genuine donor forecast maximum squared error | **0.000006737** |
| Zero/blind divergent both-correct | 0/1,920 |
| Shuffled donor divergent both-correct | 460/1,920 |
| Illegal actions | 0 |

Independent CPU replay *from the original frozen trained donor and receiver checkpoints* recomputed the 4,800 observation SHA-256 hashes, all raw primary and foreign logits/actions, legal masks, true oracle actions/costs and each nonpooled cell: **zero neural logit drift or verdict disagreement**. Every H26 holdout differs from all 19,200 H22/H23/H24/H25 exposed acceptance observations, all 47,040 H25 posthoc transformed observation receipts and all 15,360 H26 training samples. The union of previously exposed hashes comprises 66,239 distinct observations (one H25 posthoc transform reproduces an original H25 observation).

## Evidentiary restrictions

This establishes conditional use of an **engineered trained-donor scalar material-cost interface** over new arrangements and harder, but still synthetic and fixed-topology, two-law near-boundary worlds. The same-law donor wet-estimate differences are tiny (~0.000019 cheap and ~0.000087 expensive) compared with the ~1.95 real law contrast; this sham is therefore weak. The shuffled/zero controls have similarly limited causal discriminating power. No general model-memory transplant, physical law discovery, alternate physical-world mechanism or CyberSand actor-policy competence is claimed.

**Past negatives are not erased:** H25's posthoc 0.01-margin test had 150/2,880 invariant failures. H26 uses freshly sampled families/margins, not the exposed H25 0.01-margin examples. H25's 0.0001-margin stress failed very badly; H26 has not demonstrated robustness at 0.0001. H23 scientific FAIL, H24 protocol INVALID and all historical negative cohorts remain unchanged.

**Qualification limit:** original H26 source, frozen checkpoints, 4,800 raw observation/decision/logit receipts, independent audit, prior-collision hashes and exact SHA-256 manifest are preserved in the owner's separately attached self-contained evidence ZIP, **not yet published into durable GitHub checkpoint custody**. There is **no exact-HEAD H26 GitHub Actions neural replay** and **no external methodological signoff**. The H22 Actions replay is not an H26 qualification.

**Next smallest steps:** secure original H25/H26 checkpoint and receipt bytes in GitHub, run frozen-checkpoint replay at exact HEAD, then obtain independent methods review emphasizing powerful nontrivial donor controls and generator shortcut risks. Further 0.0001-boundary investigation, if justified, must be a new preregistered cohort with no H26 reuse. Keep existing PR #253 **Draft/HOLD, unmerged/not Ready**. No new PRs/issues/branches, no paid GPU/API, no CyberSand/DRIV3R changes.
