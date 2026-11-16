# C11 — signed host correction / safe release (16 November 2026)

**Status:** model-side synthetic research only · Draft/HOLD PR #639 · no merge, Ready, native changes, native Undo, paid compute or production defaults.

## Question and design

Can an already trained persistent-backtracking worm recover faster when the host **reverses** a bad recovery instruction and explicitly says CONTINUE, without turning a steerable worm into a deterministic backtracking automaton? C10 tested revocation/grading. This C11 ablation compares simple C9 cancellation with an extra signed, host-issued CONTINUE cue after a false BACKTRACK warning. It does **not** assume that CyberSand's accepted native schema exposes this field.

The external 15×15 synthetic host retains all original 25 candidate features and legal action masks. Three additional, explicitly host-owned research features extend the C9 26-feature observation: signed directive (+1 backtrack, −1 continue, 0 absent), confidence [0,1], bounded age/12. A host can countermand after 0, 1, 2 or 4 decisions; this clears its previous recovery TTL, but never retracts ink or changes materials. The adapter freezes all C9 and prerequisite weights, and changes **only the BACKTRACK candidate logit**. Inactive/revoked scores and all other action scores are exactly preserved. In the bounded model positive/backtrack advice is intentionally passed through unchanged to C9; only a new negative/CONTINUE advisory is learnable.

Both tried variants use genuine CPU AdamW training on **6,912 menus** (144 training source groups × two true/false goal outcomes × four host-replayed history depths × two strengths × three scalar guidance arms), **540 optimizer steps each**. Hairpin and staircase topology families train; hook and zigzag families remain withheld by topology, with fresh source-seed domains. Legal material/movement checks are enforced separately by `World.apply`, never by model scoring. Host-applied cursor-history BACKTRACK does not remove prior ink; no real native Undo is implemented.

| Variant | Learned residual | Regularized training loss, first → last | Model SHA-256 |
|---|---|---|---|
| C11-A (negative control) | ±5 logit on both BACKTRACK and CONTINUE cues | 0.013264 → 0.002731 | `a3541e80dbdeedd3d62d5af09ca336819589a4e1c634bf2689855985bbae10c4` |
| C11-B (retained comparator) | ±1 logit, **CONTINUE only**, positive recovery passed through | 0.018690 → 0.018437 | `a402c30bf628ae1cfffc3bba788654ce460ac9c3029ef4ced7dd7d5223a11bc7` |

Both variants reproduced byte-for-byte on an **independent full 540-step optimizer replay** with the same frozen prerequisite bytes. This is a deterministic optimizer replay, not a second independent final evaluation.

## Final-source held-out host-applied outcomes

The source domain below contains 12 independent source seeds per topology, two personalities, three strengths, and two models. Each model receives a source/seed-matched legal-action menu and RNG stream; once actions diverge subsequent states cease being same-state counterfactuals. Counts are **goal reached within 28 host-applied decisions** (not first-action labels), pooled across 288 episodes per model/outcome/condition. False warnings occur in productive/ahead worlds; release ticks count model decisions before the host supersedes its warning. C9 receives a simple cancellation; C11 receives an explicitly signed CONTINUE instruction. Thus the intervention protocol is *not identical* after countermand: these are the two host-policy strategies being compared.

| Productive/ahead world | Frozen C9 | C11-A full residual | C11-B bounded CONTINUE |
|---|---:|---:|---:|
| No warning | 251/288 | 251/288 | 251/288 |
| False BACKTRACK warning remains live | 263/288 | **177/288** | 263/288 |
| Host supersedes at tick 0 | 262/288 | 262/288 | 262/288 |
| Host supersedes at tick 1 | 266/288 | 249/288 | 266/288 |
| Host supersedes at tick 2 | 256/288 | 214/288 | 257/288 |
| Host supersedes at tick 4 | 256/288 | **161/288** | 255/288 |

Genuine goal reversals with a live BACKTRACK warning: C9 **249/288**, C11-A **288/288**, C11-B **249/288**. C11-A improves reversal compliance but devastates productive continuation; in live reverse the small left/right turn personality gap collapses to **0.000**, from C9's −0.036341. **Do not promote C11-A**. C11-B restores exact predecessor behaviour when the host asks for backtracking; it is safer, but has **no demonstrated positive effect on complete-goal success**, despite slightly fewer post-release BACKTRACK actions.

Two additional *fresh evaluations* of **3,072 episodes each** (eight new seeds per family, 28-decision horizon) use C11-B versus frozen C9:

| Cohort | Release 0, C9 → C11-B | Release 1 | Release 2 | Release 4 |
|---|---:|---:|---:|---:|
| Fresh full strength, seed 2990001 | 170→168/192 | 172→172 | 174→174 | 182→182 |
| Fresh 0.45 confidence, seed 3990001 | 173→173/192 | 173→174 | 168→167 | 180→179 |

These countermand outcomes are at best near-equivalent, not evidence for a useful additional CONTINUE head. The *directional advice* idea remains interesting, but must be evaluated against cancel-only under richer misleading geometry before further optimizer spend. All feedback arms are recorded individually in raw trajectories, rather than only the pooled table.

## Source, invariants and custody

- **96/96 available archived-source and C11 regression tests pass** locally. This includes exact no-cue logit equality, nonbacktrack logit equality, zero confidence, host-controlled countermand, no material modification, illegal-action rejection and real AdamW weight updates. **Not the complete repository HEAD suite:** the separate C5 clearance tests and C10 source/tests are absent from the available local C9 archive. Do not call this full exact-HEAD qualification.
- Source: `signed_feedback.py`, `test_signed_feedback.py`, `signed_gallery.py`. Original source/candidate menu/material/observation hashes, all actions, per-step signed directives, endpoint and ink hashes, both learned checkpoints and prerequisite model SHA-256 values are retained in the separately verified local bundle. The 24-panel deterministic SVG/HTML gallery selects the lowest-seed member for each topology×release-time combination; it is not curated by successful completion.
- Immutable GitHub Actions checkpoint custody remains **unmet**: no workflow mutations are permitted inside this research-only ownership. Local ZIP/checkpoint verification does not substitute for an immutable Actions artifact.
- No d1-2B weights trained, no resident worms, no model dispatch to native CyberSand, no native material rollback or owner-accepted observation schema change, no paid API/GPU spend, no merge/Ready.

**Disposition: continue *investigation* only, hold both C11 model variants for integration.** A later training iteration should create hard countermand cases where the correct response is actively *not* BACKTRACK at a misleading near-wall junction, but where context occasionally still requires a legal retrace. Evaluate the host's reliable evidence independently from the learned policy and preserve personality distributions over a longer drawing horizon. No shared or game-side change is requested.
