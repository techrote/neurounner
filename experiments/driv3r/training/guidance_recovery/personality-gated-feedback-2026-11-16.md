# C7 — preserve worm character during active goal-switch correction

16 November 2026 · CyberSand #570 · Draft/HOLD PR #639 · training-lane research only.

## Question and trained method

C6's unrestricted correction residual recovered goal switches but collapsed the strong-guidance, cue-active upper-fork personality contrast from 32/32 to 0/32 matched opposite-personality first decisions. Can a trained feedback head keep the original worm's *relative drawing-action preferences* intact while still learning timely cursor-history recovery?

C7 freezes the exact C4 policy (base SHA-256 `8c8625a1bd8daed03e00188687dbba9732634c7bf7b8e7be5a4fa21cb310ac09`, C4 cue adapter SHA-256 `f1d538200c0d5e183d46463a9058f73db162a5dcccf13daeb521073da9bfc0fa`) and trains a separate **25→12→1 residual acting ONLY on the BACKTRACK candidate**, multiplied by the existing changed-goal and host-cue flags. All nine other candidate logits are bit-identical to C4 even when a recovery cue is active; all ten logits are identical without a simultaneous cue and goal-change event. Native/material authority remains with the host; this gate cannot authorize painting or material rollback. This is a distinct parallel C7 experiment, **not** stacking C6 or the C5 optional-clearance adapter.

Training used 240 seeded source groups × 2 goal outcomes × 3 guidance strengths (0/0.35/1.0): **1,440 matched supervised action menus** from **480 distinct synthetic World hashes**, with upper forks mixed into training. The target is binary BACKTRACK versus all other legal choices, *not* a hard-coded drawing direction. **360 actual AdamW CPU updates**, regularized binary loss **1.077176 → 0.260701**. Six bounded exploratory epoch/rate variants were checked on a separate development seed group; the 24-epoch, 0.002 learning-rate configuration was selected. Do not treat those reused template/dev results as blinded generalization.

## Reused C6 comparison fixture (40 independently seeded examples per condition)

| Cue-active upper-fork metric | C6 | C7 |
|---|---:|---:|
| Correct reversal, no scalar | 30/40 | **40/40** |
| Correct reversal, weak scalar | 30/40 | **40/40** |
| Correct reversal, strong scalar | 20/40 | **30/40** |
| Correct productive forward continuation (all scalar arms) | 40/40 | **40/40** |
| Strong-scalar opposite-personality divergence, same-state ahead goal | 0/32 | **32/32** |
| Correct goal arrival, 12-turn greedy upper-fork reversal (each arm) | 12/12 | **12/12** |

The improved personality result is a **guaranteed non-BACKTRACK ranking invariant**, not broad artistic or personality understanding. With a reversal cue, only the BACKTRACK-vs-other-action odds may change. The strong-scalar first-action reversal still fails **10/40**, a retained negative. Both policies reuse a narrow junction template; lower-junction transfer is withheld only from this trainer, not a fully new game family.

## New independent-seed, stochastic 16-turn host-applied episodes

Used seeded legal-action softmax selection at temperature 0.7 (no forced successful step, identical random stream per matched source) on **24 independent seeds per topology/arm/cue/model = 1,296 episodes**, contrasting frozen C4, selected C6 and trained C7 on 3 topologies × 3 scalar strengths × cue absent/present. The host independently validates each selected action. After model choices diverge, the trajectory states are *not* same-state comparisons.

| Reversal goals reached, cue present | C4 | C6 | C7 |
|---|---:|---:|---:|
| Corridor, 72 episodes | 60 | 71 | **71** |
| Upper fork, 72 episodes | 3 | 33 | **42** |
| Lower fork, 72 episodes | 42 | 65 | **66** |
| **All 216 cue-active episodes** | **105** | **169** | **179** |

All models achieved 161/216 on the cue-absent matched cohort; C7's **full uncued trajectories are identical to C4** for every matched source and random stream. No host-rejected actions, protected-material modifications or synthetic ink deletions were observed. Stochastic upper-fork C7 recovery is **42/72**, still poor in absolute terms; the result does not prove long-horizon robust recovery or novelty of drawing composition.

## Reproducibility, boundaries and custody

- `python -m unittest discover -s . -p 'test_*.py' -q` against the locally available exact-Git-blob-matched C4/C6 source, plus C7 source/tests: **71/71 pass** (61 prior + 10 new). Four new C7 source/test files were individually Git-blob matched before publishing. **The full current PR suite is NOT qualified** because separate C5 modules/tests were unavailable locally; do not count 71 as full exact-head coverage.
- Two complete C7 CPU train/evaluate runs produced byte-identical model bytes, metrics and first-action/turn-by-turn receipts. Two independently executed stochastic comparisons likewise reproduced byte-identical metrics and 1,296 receipt records.
- C7 checkpoint SHA-256: `9ddadf372465abca9634eca8bf450cb03e5903434da8065de588511b44b5da9a` (1,836 bytes). Dataset SHA-256: `80401b37fcc85b4303e5bf4ae2ad3811b990d4e36878c5f38a58cedd98acaac0`. Source-group receipts SHA-256: `6cfef02d703d60c08817368e7ef176d7be08d4c8742472319de88bc30ccc5154`. Stochastic receipt index SHA-256: `fbd4ef6b2d3129fd046f9225b46c5b8fec97505634b1ad845036d3171eef6ab3`.
- Local reproduction (install PyTorch CPU, NumPy, safetensors) from this directory: `python personality_gate.py --base /path/to/guidance_recovery_cue_calibrated.safetensors --previous /path/to/guidance_recovery_adversarial_adapter.safetensors --out /tmp/c7` then `python stochastic_probe.py --base ... --adapter ... --c6 /path/to/guidance_recovery_goal_switch_adapter.safetensors --c7 /tmp/c7/guidance_recovery_personality_gate.safetensors --out /tmp/c7`. Exact prerequisite checkpoint hashes above; original C6 SHA-256 `84e14a18182d49779fe04cd6d0df1251dfef1c87289085b6b92ece0c16df002d`.
- Trained weights, full JSON receipts and a rendered 18-panel trajectory gallery are held in a locally SHA-256-verified bundle, **not Git history**. **Immutable GitHub Actions artifact custody is not complete**: workflow ownership lies outside the permitted `training/guidance_recovery/**` path. Request an authorized workflow owner to run exact-source reproduction and upload artifacts before calling this hosted-qualified.
- No modifications outside this training lane; no native/Workbench/simulation/game modifications, paid API/GPU services, model-2B training, formal native A/B, real ink Undo, merge or Ready promotion. All snapshots are synthetic; the host owns permissions and action execution.

**Next research:** increase recovery transfer across structurally new obstacle/branch families and assess active personality with stochastic style metrics beyond merely preserving two forward logits. Full current-HEAD C5-inclusive regression and immutable Actions custody remain open gates.
