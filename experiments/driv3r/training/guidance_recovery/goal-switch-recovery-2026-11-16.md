# DRIV3R guidance/recovery C6 — goal-switch cue arbitration

16 November 2026 · Issue #570 · owner roadmap #570 comment 6877408001 · Draft/HOLD PR #639

**External model-side synthetic research only.** No CyberSand native, Godot, Workbench, shared contracts, maps, assets, painter lanes, training-weight Git commits, paid GPU/API calls, production/default changes or real material Undo. Host applies legal actions; BACKTRACK reverses only cursor history and never deletes ink. This is not resident d1 inference.

## Hypothesis and architecture

The C4 adversarial cue adapter sometimes paints *forward* when a goal flips behind an already painted corridor and the host warns the worm to recover. Can a small adapter learn to arbitrate corrective cues without rewriting its independent drawing policy?

The new 25→16→1 learned residual sits on the **frozen C4 25-feature base + adversarial adapter**. It activates only when BOTH original goal-change feature 18 and correction-cue feature 23 are present. With either absent, new output logits are bit-identical to the frozen C4 policy, including after real training. The legal menu and 25-feature schema are unchanged. This is an independent C4-derived research candidate, **not integrated with C5** (the 28-feature optional branch-clearance head); never silently stack the two adapters.

Train 240 world/seed groups × reversal versus still-ahead objective × three no/weak/strong guidance strengths = **1,440 labelled cue-active action menus**. Half the groups have a viable upper-side branch; use personality-sensitive soft labels across two productive INK choices rather than force one path. Train rotations 0/1 and withhold rotations 2/3, fresh seed streams and the lower-branch geometry. Two complete optimizer/evaluation executions produced byte-identical checkpoints, metrics and four first-action/rollout receipt streams.

**Main selected checkpoint:** 180 CPU AdamW updates across 12 epochs; regularized loss **1.418053 → 0.581600**. Full C4+C6 local 61-test suite PASS (52 original + nine new); exact Git blob matching verified for the three new Python modules. The separate C5 tests were not locally loaded; full exact-HEAD Python suite remains **unqualified**. No fake CI/native acceptance claims.

## Held-out first legal actions

These are **correct/40 source worlds per scalar strength, prior C4 → trained C6**, with a real synthetic host cue. For changed-goal reversal, correct means BACKTRACK; when the new goal remains ahead, either genuinely productive INK branch is allowed.

| Scenario | No guidance | Weak | Strong |
|---|---:|---:|---:|
| Corridor reversal | 0→40 | 0→40 | 0→40 |
| Upper branch reversal | 0→30 | 0→30 | 0→20 |
| Withheld lower branch reversal | 0→40 | 0→40 | 0→40 |
| Corridor goal still ahead | 40→40 | 40→40 | 40→40 |
| Upper branch goal still ahead | 40→40 | 40→40 | 40→40 |
| Lower branch goal still ahead | 40→40 | 40→40 | 40→40 |

Initial source World/material/legal menu identity is paired between cue/sham and before/after models. Later rollout states diverge and are not matched-state causal observations. Zero synthetic host-rejected actions in these recorded first-choice cohorts.

## 12-decision recovery and style limitations

On 12 held-out synthetic histories per scalar arm with cue active, corridor goal completion is 12/12 before and after; upper-branch completion changes **0/12 → 12/12**; lower-branch completion changes from 0/12 (none), 6/12 (weak/strong) to **12/12 in all arms**. None of this establishes long-horizon native recovery, interesting art, or actual material rollback.

Important negative: in a same-geometry *opposite personality* 32-world comparison, upper-branch cue-active **strong-guidance action divergence falls from 32/32 to 0/32**. Autonomy without a cue is preserved exactly, but personality *under correction* is not. A stronger personality-labelled training sweep (two additional **180-update** runs, 10/90 and 2/98 branch-choice supervision instead of 35/55) restored only 16/32 divergence and reduced upper-branch strong-guidance goal completion **12/12 → 6/12**. Retain the balanced candidate and this negative finding; do not describe the worm as reliably personality-preserving.

## Exact provenance and reproduction

From experiments/driv3r/training/guidance_recovery/ with CPU PyTorch, NumPy and safetensors:

    python -m unittest discover -s . -p 'test_*.py' -q
    python goal_switch_recovery.py --base /path/to/guidance_recovery_cue_calibrated.safetensors --previous /path/to/guidance_recovery_adversarial_adapter.safetensors --out /tmp/driv3r-goal-switch --mix-branches
    python goal_switch_gallery.py --before /tmp/driv3r-goal-switch/goal_switch_walks_before.json --after /tmp/driv3r-goal-switch/goal_switch_walks_after.json --out /tmp/driv3r-goal-switch/goal_switch_gallery.html

- Frozen C4 base SHA-256: 8c8625a1bd8daed03e00188687dbba9732634c7bf7b8e7be5a4fa21cb310ac09
- Frozen C4 adversarial adapter SHA-256: f1d538200c0d5e183d46463a9058f73db162a5dcccf13daeb521073da9bfc0fa
- **C6 trained adapter SHA-256: 84e14a18182d49779fe04cd6d0df1251dfef1c87289085b6b92ece0c16df002d**
- Exact goal_switch_recovery.py SHA-256: 9fe567eec2cdf9938aa8895a41c9ac13435dc4752d374befbdb9d6a1f711cbe7
- Reproduced metrics JSON SHA-256: 9b365472e68aa57e31c158ee43e40e2548f2e55012800cc4383f5a5e05e9f799
- Exploratory style-head adapter SHA-256s: 76c76c703616c16e66f06297ae0ecca7d1f073de6b8daad389fe55f94a18f851 and 7f78d215c73f98b23751728549d34ebb042136891cc601f72487ea70845e9db4

All source-level World/material/menu/decision provenance and full 12-turn original model traces are preserved in the external local bundle, including a source-reconstructed 18-panel SVG/HTML gallery. Model checkpoints are **not in Git history**. Immutable GitHub Actions artifact publication is still a workflow-owner task: .github/workflows/** is outside exclusive ownership. Do not change shared workflow files, merge, or mark PR Ready.
