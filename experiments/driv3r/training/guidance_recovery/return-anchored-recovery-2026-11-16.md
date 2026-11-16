# C19 — return-anchored attested guidance (16 November 2026)

**Research-only; Draft/HOLD. No promotion.** C12 remains the preferred trained checkpoint and C18-A0 the strongest untrained host-attestation control. This is external CyberSand-like synthetic gameplay, not native Workbench or resident-worm inference.

## Causal question and architecture

Can a small learned BACKTRACK correction head improve on the successful **untrained** C18-A0 host-attested trail gate without over-obedience? The model receives three additional **host-verifiable executed-history facts**, not hidden map geometry. C12 remains frozen. Only the BACKTRACK score may change, bounded from **+0.2 to +1.8 logits** when the host currently gives signed advice and the goal is attested on the executed trail. Unguided, expired, illegal BACKTRACK and off-trail predictions are identical to C12; legal action masks and protected materials remain host-owned.

Added an adversarial **bridge** topology: a historical goal lies one legal forward step away, yet the historical cursor stack offers a much longer retrace. On-trail evidence cannot be treated as proof that BACKTRACK is shortest.

## Actually executed optimization

576 matched first-decision menus from hairpin, staircase, shortcut and bridge geometry, across weak/strong recovery advice, three scalar soft-hint arms and alternating personality. **2,304 host-executed, stochastic counterfactual suffix trajectories** were compared under the frozen C18-A0 controller. Realized arrival, progress, actions and ink costs supply reward differences. Of 576 comparisons, **225 favor BACKTRACK, 66 favor forward and 285 are near ties**. Ahead/off-trail controls stay non-gradient-active by design.

Two independently trained heads (FULL 32→12→1 and LOCAL 10→12→1) received **360 CPU AdamW updates each** using a reward-weighted contextual-policy surrogate, source-group maximum objective and drift-to-A0 regularizer. Objective FULL **+0.000024 → −0.001413**, LOCAL **+0.000034 → −0.001373**. Two independent optimizer replays were **byte-identical** for each checkpoint.

- FULL safetensors SHA-256: `2139da0f70a6b3cd284ede34a741666ca28ae2fc67a4b9236bb013fabe67e3dc`
- LOCAL safetensors SHA-256: `4b3a95e2e9c84a25263342189ab02676116686f2db26a1073ad27ff7e1cc804f`

Neither checkpoint belongs in Git history; both, their exact prerequisites, raw receipts and complete reproducible trainer are in the separately delivered SHA-256-manifest-verified owner archive.

## Fresh held-out, host-applied outcomes

Three independent seed cohorts (`196001`, `196777`, `197555`) produced **6,912 stochastic complete episodes**, 1,728/model. Results are goal completions out of 144 per model/condition, across all scalar arms and both personality settings:

| Task / host directive | Frozen C12 | Untrained A0 | Trained FULL | Trained LOCAL |
|---|---:|---:|---:|---:|
| Goal reversed / valid BACKTRACK | 105 | 128 | 129 | **131** |
| Goal reversed / misleading CONTINUE | 114 | 135 | 135 | 137 |
| Goal ahead / misleading BACKTRACK | 136 | 136 | 136 | 136 |
| Goal ahead / valid CONTINUE | 124 | 124 | 124 | 124 |
| Shortcut / BACKTRACK | 54 | 69 | 70 | 67 |
| Shortcut / CONTINUE | 107 | 130 | 133 | 133 |
| Adversarial bridge / BACKTRACK | 109 | 121 | 122 | 121 |
| Adversarial bridge / CONTINUE | 122 | 128 | 128 | 128 |

Improvements relative to **A0**, rather than only to the older C12, are **small, inconsistent and not promotion-grade**. In particular LOCAL loses two net shortcut-BACKTRACK goals (67 vs 69). The bridge's one-step forward route is geometrically optimal, but C12's own stochastic continuation may fail to select it; these numbers are not proof of optimal routing.

All **576 cue-off matched trajectory/ink pairs per model are exact matches to C12**, and all reported held-out worlds had zero synthetic host rejection, zero illegal raw top-1 and protected-cell mutation. Different models begin from the same source/menu/RNG; their futures are no longer same-state comparisons after diverging.

**Personality negative:** active-BACKTRACK aggregate opposite-personality turn-gap C12 **+0.0216**, A0 **−0.0129**, FULL **−0.0306**, LOCAL **−0.0320**; the unguided value remains exactly **+0.1268** for all models. Conditional style varies substantially across topologies, so personality retention under correction is *not established*. No candidate promoted.

## Qualification/custody limitations

The model-only [C19 inference head](return_anchored_core.py) and [10 independent source guards](test_return_anchored_core.py) are in this directory. Locally available archived suite **140/140 PASS**, including 10 new guards; previous C5/C10/C15 tests are unavailable in this archived run. **Full repository exact-HEAD qualification and immutable GitHub Actions model-artifact custody are still outstanding.** Reproducible source and full 12-panel synthetic SVG/HTML gallery are preserved outside Git in the owner archive.

No CyberSand `native/**`, `godot/**`, Workbench, material/map/scene/asset files, simulation, provider defaults, unrelated workflows or shared contracts were edited. No paid compute, real ink Undo, game integration, merge, Ready or auto-merge.

**Decision:** preserve C19 as a well-controlled but mixed research result; next target a hard active-personality gate and more varied multi-turn false-recovery cues rather than further increasing BACKTRACK compliance.
