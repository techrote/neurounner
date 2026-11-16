# C10 — revocable and graded host feedback, trained offline (16 Nov 2026)

**Status:** RESEARCH ONLY · Draft/HOLD PR #639 · no merge/Ready · no game/runtime changes.

## Design and actual optimisation

This iteration tests whether a noisy host-issued recovery instruction can remain useful without becoming a hard BACKTRACK override. It extends the **frozen C9 26-feature gate** with two *research-only*, host-owned fields: confidence in [0,1] and cue age / 12. The original 25 candidate features, legality flags and source world are unchanged. A host may countermand the cue at any moment; this clears its synthetic feedback timer and context, **not** brush material or history. No claim is made that the CyberSand native host already supports countermand.

The **28→16→1** new residual changes only the BACKTRACK candidate score; other drawing and STOP logits remain unchanged. The predecessor C9 and all earlier model heads are frozen. In inactive/countermanded states, scores follow the unmodified prior; exact-gate tests cover absence of cue, revoked cue, and other-action logits.

Generated 8,640 training menu states (180 source groups × 2 real host-applied outcomes × 4 cue ages × 2 confidence levels × 3 scalar guidance strengths). The host-applied outcome is a training target, **not** an extra inference feature. Confidence .55 and 1.0 occur in both true and misleading cases. Train families: hairpin and staircase. Hook and zigzag completely withheld by family; seeds separately isolated.

Two genuine AdamW fine-tunes completed, 680 steps each, using frozen C9:

| Variant | BACKTRACK cap | Reversal-positive BCE weight | Initial → final regularized loss | Checkpoint SHA-256 |
|---|---:|---:|---:|---|
| C10-A | ±6 logits | 1× | 0.039764 → 0.009575 | d3738d260c2cd46baf7a3f9ca72fc325c955fab3c2ce4d91cc400ff5c60d2044 |
| C10-B | ±4 logits | 2× | 0.062860 → 0.023519 | 5f8f7698c0e0f032d57d6b3a70df420cc72cd7721fe592f1c24a2b3fe76581ef |

C10-B was retained as the **comparison candidate**, not promoted for integration. Independently repeated B optimiser steps produced a byte-identical safetensors checkpoint. Training examples and model binary are intentionally **not committed**.

## Source/seed-matched stochastic outcomes

All counts below mean reaching the target within at most 24 host-applied steps, not merely making the correct first choice. Source worlds, geometry, personality, action menus and initial RNG draws match within each comparison; after diverging decisions, states are **not same-state counterfactuals**. The no/weak/strong scalar-guidance arms are pooled here, but separately preserved in raw receipts and the metrics JSON.

**Initial held-out source domain** 1600401, 12 source seeds per family, 288 episodes per controller × outcome × cue condition:

| Controller / live host feedback | Goal reversal | Misleading cue, continue forward |
|---|---:|---:|
| Frozen C9, strength 1.0 | 237/288 | 268/288 |
| C10-B, strength 1.0 | 243/288 | 276/288 |
| Frozen C9, strength 0.55 | 199/288 | 270/288 |
| C10-B, strength 0.55 | **241/288** | 269/288 |

**Fresh independently seeded validation** 1893301, eight source seeds per family, 192 episodes per controller × outcome × cue condition:

| Controller / live host feedback | Goal reversal | Misleading cue, continue forward |
|---|---:|---:|
| Frozen C9, strength 1.0 | 172/192 | 169/192 |
| C10-B, strength 1.0 | 163/192 | 174/192 |
| Frozen C9, strength 0.55 | 148/192 | 171/192 |
| C10-B, strength 0.55 | **176/192** | **165/192** |

**Interpretation:** C10-B at reduced feedback strength recovers more effectively than C9 at the *same* reduced strength, including withheld families; however its misleading-cue continuation is worse on the fresh domain (165 versus 171), and its improvement over fully confident C9 is small (+4 reversals, −4 false-cue completions). At full strength, C10-B reversals regress versus C9 on fresh sources (163 versus 172). Variant A also regressed on withheld reversals despite improving false-cue cases. These are meaningful negative findings, not a generally superior recovery model.

**Personality under active feedback remains uncertain.** On fresh full-family reversed goals, the signed left/right turn-separation diagnostic is 0.199719 for C9-full versus 0.185915 for C10-B-weak. At C10-B full it falls to 0.132035. Individual family gaps can be unstable or change sign because these short recoveries contain few non-BACKTRACK turns. These measurements cannot establish intact drawing character.

**Countermand control:** after the two-decision host countermand, the synthetic host switches off its feedback context and original cue flag. This restores original inactive scores; it cannot retroactively undo the first two actions or painted pixels. All 4,608 fresh-validation episodes were host-legal; zero host rejects and zero illegal raw top-1 choices. Across 384 fresh no-cue C9-full/C10-weak matched pairs, entire action sequences and final ink hashes were identical. These are **external synthetic host guarantees**, not native material-Undo evidence.

## Reproducibility and limitations

Reproduction from repository root (Python with CPU PyTorch, NumPy and safetensors):

~~~bash
python -m unittest discover -s experiments/driv3r/training/guidance_recovery -p 'test_*.py'
python experiments/driv3r/training/guidance_recovery/revocable_feedback.py \
  --base /path/to/guidance_recovery_cue_calibrated.safetensors \
  --c4 /path/to/guidance_recovery_adversarial_adapter.safetensors \
  --c7 /path/to/guidance_recovery_personality_gate.safetensors \
  --c8 /path/to/guidance_recovery_topology_transfer.safetensors \
  --c9 /path/to/guidance_recovery_persistent_feedback.safetensors \
  --n 180 --epochs 10 --eval-n 12 --reverse-weight 2 --scale 4 --out /tmp/gr-c10
~~~

Frozen predecessor and selected C10 checkpoint hashes, complete source/seed observation receipts, all original action/ink hashes, fresh-validation trajectories, variant A and B, and a deterministic source-reconstructed 32-panel HTML/SVG gallery are in the separately packaged **local C10 bundle**. Full executable C10 and regression-test source is committed within the owned lane. The selected B checkpoint SHA-256 is 5f8f7698c0e0f032d57d6b3a70df420cc72cd7721fe592f1c24a2b3fe76581ef.

Locally executed **97/97 archived-source plus C10 tests** PASS. New C10 modules match their published Git blobs; the archived C9 prerequisites also match the checked repository source blobs. **This is NOT the complete exact-HEAD test suite**: the separately committed C5 clearance_ablation.py, clearance_gallery.py, and test_clearance_ablation.py are absent from the local C9 archive. Full checkout/CI and independent inspection remain outstanding.

Immutable GitHub Actions artifact custody is still **not established**. The current lane has no authority to modify shared workflows; do not mistake a verified local ZIP for an immutable Actions artifact. No LiquidAI d1 model was trained; no resident worm runs, native material permission changes, real native Undo, paid GPU/API, game modifications, production defaults, merges or Ready promotions occurred.

**Next discriminating test:** on-policy misleading-recovery countermand timing (0, 1, 2, 4 turns), observed before/after correction and long stochastic strokes with sufficient non-BACKTRACK turns to judge personality. Retain the conventional no-cue baseline and source- and RNG-matched comparisons. Model-side experiments stay offline until the owner authorizes native integration.
