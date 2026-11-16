# Guidance/recovery C5 — optional bounded branch-clearance observation (16 November 2026)

**Disposition: Draft/HOLD model-side research.** Owner: `experiments/driv3r/training/guidance_recovery/**`; related [#570](https://github.com/techrote/cybersand/issues/570), [owner roadmap](https://github.com/techrote/cybersand/issues/570#issuecomment-6877408001), [#639](https://github.com/techrote/cybersand/pull/639). **No CyberSand native, Godot, Workbench, materials, production/defaults, D-1B, D-COMMON or #568 files changed.** This is an external *synthetic* host, **not** LiquidAI d1 inference or training. No native Undo: `BACKTRACK` only changes cursor history; painted material remains. The host validates each action again and never delegates legal permission to the model.

## Question, design, custody

C4 proved that legal action menus and all 10×25 model-visible candidates were *exactly identical* between a viable branch and a deeper blocked decoy, forcing opposite ground-truth decisions from identical observations. This experiment asks whether a **read-only optional** five-cell host-verified clearance and reachable-cell count (two extra floats, plus presence flag) can disambiguate them. A bounded flood fill starts from each legal INK endpoint, excludes the cursor/history corridor, never changes original features/action menus and never operates as a permission grant. This is a **proposed future host interface**, not a claim that present native DRIV3R already emits the fields. The prior exact 25-feature contract is not silently changed: a new versioned experimental 28-feature wrapper is isolated in this lane.

The existing 25→64→64→1 neural scoring base and C4 25→32→1 cue adapter are frozen. A new 28→32→1 residual (4,404-byte safetensors) is activated **only if both a host recovery cue and optional clearance evidence are present**. Therefore both no-cue logits and missing-evidence logits fall back bit-for-bit to the previously trained policy; there is no training-mediated replacement of the worm's autonomous behaviour. With evidence present, the residual can still change choices and style. Host legality remains authoritative.

**Actual CPU optimizer experiment:** 240 deterministic, source-separated training groups × six scenarios (genuine warning, straight/corner productive+same-state deeper decoy, open false cue), one of three 0/.35/1.0 guidance strengths per group; **1,440 labelled cue-active action menus**; **216 AdamW updates** across 18 epochs, with conservative residual-penalty supervision. Regularized mean training loss **0.861953 → 0.311507**. Supervised training uses depths 2–3. Two independent *entire* train/evaluate executions produced byte-identical weights, all eight JSON receipt streams and metrics. **63/63 existing+new offline regressions passed.** Existing eleven Python sources were independently matched to published Git blobs at original HEAD `15d1f9e78e425822e02eac426b07bbb72d168377` before the experiment.

## Source-isolated depth-4 held-out results

Each line reports **correct first legal action /32 independent source seeds per guidance strength** on the primary heldout `920001+i×977`. A second disjoint heldout at `940001+i×977` reproduced the new results.

| Cue issued; synthetic host offers same legal menu | Frozen prior | Clearance-trained | Evidence withheld from new model |
|---|---:|---:|---:|
| Genuine immediate obstruction | 32/32 | 32/32 | 32/32 |
| Straight productive corridor, false warning | 32/32 | 32/32 | 32/32 |
| Corner productive corridor, false warning | 32/32 | 32/32 | 32/32 |
| **Straight deeper decoy, genuine warning** | **0/32** | **32/32** | **0/32** |
| **Corner deeper decoy, genuine warning** | **0/32** | **32/32** | **0/32** |
| Open productive corridor, false warning | 32/32 | 32/32 | 32/32 |

Identical counts at 0, .35 and 1.0 guidance strengths. Four complete matched panels each contain 1,152 actions (32 seeds × 6 scenario families × 3 guidance strengths × cue/no-cue). No synthetic host-rejected action was admitted. Results reflect the **same architectural corridor family**, with depth-4, some withheld seed/orientation combinations; these are *not* general OOD success. This is a single-choice causal observation ablation, not proof of long-horizon learned recovery.

## Negative controls intentionally preserved

- **Lookahead ceiling:** A new untrained two-step-stem family has a blocked cell beyond the radius-five observation. Viable and blocked worlds have different protected-material hashes but **identical entire 10×28 model inputs** and menus. Model correctly continues in all **96/96** viable sources per straight/corner family (32 seeds × 3 arms), but **mistakenly continues in all 96/96 blocked counterpart sources**. It cannot fix information that is not observed; do not fit it to those cases and declare generalization.
- **Misreported host evidence:** Swap the two clearance fields across source-matched viable/dead-end worlds while keeping all 25 original features and legal menus unchanged (research-only, never used to authorize actions). Both direction categories fail **96/96** per family under this deliberate evidence corruption: factuality/integrity of optional host evidence must be verified, and a model output must never be mistaken for host authorization.
- **Personality and long-run stability:** On 24 disjoint seeds × three synthetic topology families (forks/switchback/clumps), two personality settings, three strengths and cue/no-cue (**288 sampled trajectories per model**, up to 30 actions, goal change at turn 15, optional correction at turn 6), no-cue traces and measured signed-personality gaps are *bit-identical* to frozen prior. With cues, signed turn-bias separation **0.236430→0.295525** with no scalar, **0.214217→0.279295** with weak scalar, **0.229065→0.239345** with strong scalar. These exploratory aggregates are small, synthetic, and after policy divergence are **not paired same-state causal evidence**. No improvement in goal distance is uniformly established. All reported synthetic host-rejected actions: 0.

No claims of true visual d1 support (`--no-mmproj`), resident worms, material rollback, original native material compliance, or production improvement. The new field is merely a source-verified *hypothesis* for future host-owned integration.

## Exact reproduction and artifact custody

Run from this lane with Python, NumPy, CPU PyTorch and safetensors installed, supplying the **original prerequisite checkpoint bytes**, never weights from Git history:

```bash
python -m unittest discover -s . -p 'test_*.py' -q
python clearance_ablation.py --base /path/to/guidance_recovery_cue_calibrated.safetensors --previous /path/to/guidance_recovery_adversarial_adapter.safetensors --out /tmp/guidance-clearance
python clearance_gallery.py --before /tmp/guidance-clearance/sampled_before_receipts.json --after /tmp/guidance-clearance/sampled_after_receipts.json --out /tmp/guidance-clearance/trajectory_gallery.html
```

- Prerequisite trained base SHA256: `8c8625a1bd8daed03e00188687dbba9732634c7bf7b8e7be5a4fa21cb310ac09`.
- Prerequisite C4 adapter SHA256: `f1d538200c0d5e183d46463a9058f73db162a5dcccf13daeb521073da9bfc0fa`.
- **New trained residual** SHA256: `cfa6be630e657e2aeaf7becd9e8079eb523e9cd2edb63165119d8082790763cd`.
- Exact source SHA256 (`clearance_ablation.py`): `0073993cfbe8419f54ccb3290853b2bc03a02a06f68eaa9934011e2738f3276c`.
- Original exact run metrics JSON SHA256: `87b4153093fe6c6fdaa950d5e95a107711beb836cd67e1cf7917db77ea208270`.
- All training tensor/label/source hashes and eight original trajectory/choice receipt SHA256s are in `clearance_metrics.json`; independent regenerated run B matched run A **byte for byte**. The 24-panel source-reconstructed HTML gallery contains original *model-selected* paths, never invented model output.

**Custody gap:** Checkpoints, metrics, full receipts and HTML remain in the externally supplied verified local archive. They are intentionally **not committed to Git**, and **not yet deposited in an immutable GitHub Actions artifact** because `.github/workflows/**` is not owned by this lane. A separately authorized workflow owner must implement exact-source read-only reproducibility + artifact upload before release-grade custody is claimed. **Keep #639 Draft/HOLD, do not merge or mark Ready.**
