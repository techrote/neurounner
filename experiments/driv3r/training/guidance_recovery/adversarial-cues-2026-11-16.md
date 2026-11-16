# D-1B-adjacent DRIV3R trainer: near-wall cue calibration (16 November 2026)

**Scope:** Model-side synthetic CyberSand worm research only. Draft/HOLD PR #639, #570 research lineage. No CyberSand/native/Godot/Workbench/material/default changes. The experiment does not run actual d1, resident SLMs or native material Undo. BACKTRACK only relocates the synthetic cursor, retaining ink. The model cannot grant permissions; the synthetic host independently rejects illegal actions.

## Experiment

Fine-tune the previous **frozen-base** 25→32→1 cue-gated residual adapter (on the exact 25→64→64→1 CPU-trained base), using source-matched cue/no-cue episodes across **0, 0.35, 1.0** scalar strengths. Train **384** unique scenario-source groups × four environment families (genuine impending dead end, near-wall straight viable continuation, near-wall corner viable continuation, open productive control), each with cue absent/present. Only the cue-active residual is optimized, with **280 real AdamW updates** over **3,840 weighted presentations**. Cross-entropy **0.737506 → 0.488116**. No-cue forward logits remain bit-identical because the residual is multiplied by cue presence.

Hold out **32 source seeds, depth-4 corridors** and test all three strengths, cue/no-cue, both worm personalities, rotational/mirror variants. All four rotations occur during training, so this is **not rotation-family OOD**. Depth-4 is held out but the corridor motifs remain closely related. Run separate 12-seed, 17-decision greedy same-source warning rollouts, plus the original eight-seed stochastic style diagnostic. The models see a finite host-offered action menu with protected materials, 16-bit scalar hints, history and recovery flags.

## Results (greedy first action; out of 32 held-out source seeds per arm)

| Family | Previous cue-correct | New cue-correct | Interpretation |
| --- | ---: | ---: | --- |
| Genuine immediate future dead end | 32 | 32 | Keeps true early-warning correction |
| Near-wall **straight** productive passage, false cue | 0 | 32 | Learns to ignore that warning |
| Near-wall **corner** productive passage, false cue | 0 | 32 | Learns to ignore that warning |
| Open productive passage, false cue | 32 | 32 | Keeps original easy rejection |
| **Decoy straight**: looks open but ends one cell later | 32 | **0** | New regression: rejects a genuine warning |
| **Decoy corner**: looks open but ends one cell later | 32 | **0** | New regression: rejects a genuine warning |

All values are identical across **no/weak/strong scalar strengths**. The six families total **1,152 decisions per checkpoint** (32 sources × 6 families × 2 cue arms × 3 strengths). No-cue first actions are unchanged in every family. The new checkpoint achieves **12/12 clean cursor-history unwinds and 12/12 escape-branch entries for each scalar strength** in the separate 17-turn, true-warning greedy rollout. No host-rejected synthetic actions and no raw illegal top-1 candidates were counted. Neither metric establishes native permission enforcement or stochastic multi-topology success.

### Fundamental negative result: hidden clearance aliasing

For every straight/corner productive-vs-decoy pair (both cue settings, both personalities, all 0/0.35/1.0 scalar arms), **all 10×25 candidate feature values and legal menu fingerprints are exactly identical**, while the deeper protected-material maps have different SHA-256 digests and require opposite oracle choices: continue vs BACKTRACK. The decoy labels were excluded from training. With this observation contract, any deterministic current-step policy must answer identically for both; its best balanced accuracy on the contradictory paired labels is at most **50%**. That is an information-limit result, not a model optimization failure in isolation. Proposed future *interface-only* experiment: host-supplied deeper clearance/branch viability with distinct source and observation hashes, or explicit cue confidence, never model permission authority.

### Personality and guidance limits

The small, reused eight-seed stochastic diagnostic remains **bit-identical with cue absent**. With cue present, the signed turn-personality gap declines **0.0916 → 0.0736** (zero guidance) and **0.0494 → 0.0231** (strong guidance). This is evidence of some style erosion under active correction, not an aesthetic judgement. No claim is made about long-horizon native model behavior or generalization to unseen topology families.

## Reproduction and provenance

Run in the owned experiments/driv3r/training/guidance_recovery/ directory with Python, numpy, CPU PyTorch and safetensors:

~~~bash
python -m unittest discover -s . -p 'test_*.py' -q
python adversarial_cues.py --base /path/to/guidance_recovery_cue_calibrated.safetensors --adapter /path/to/guidance_recovery_positive_cue_adapter.safetensors --out /tmp/guidance-adversarial
python adversarial_gallery.py --receipts /tmp/guidance-adversarial/adversarial_first_action_receipts.json --out /tmp/guidance-adversarial/gallery.html
~~~

- Exact base checkpoint SHA-256: 8c8625a1bd8daed03e00188687dbba9732634c7bf7b8e7be5a4fa21cb310ac09.
- Exact preceding adapter SHA-256: 1dae5dc3fa22cae7dc681c4f9702ce9d4ef84f0f1e9cb33693cbc88a5867a4f5.
- New **adapter-only**, frozen-base-dependent safetensors SHA-256: f1d538200c0d5e183d46463a9058f73db162a5dcccf13daeb521073da9bfc0fa.
- Training script byte SHA-256: cf7e056a0d76fb5e62999dd8e6eaf9a0c943f8db14c9f93b65e594c7de3a488e.
- Training tensor bytes SHA-256: 4bd41e36cb10a95b6c2b8d7f8977447b7fd5ee28848a4802ffd718a0ee590279.
- Exact prior-to-after first-action receipt digest and additional source/label hashes are in adversarial_metrics.json, not asserted from the Git file alone.
- Repeated full CPU training+evaluation runs produced **identical adapter bytes, metrics and all first-action/rollout receipt bytes**. The full local suite was **52/52 PASS**, comprising 41 pre-existing and 11 new tests, from Git-blob-matched prior modules.
- A 24-panel source-reconstructed HTML gallery shows two source seeds, before/after, all six families at weak guidance, cue present. It is **synthetic source visualization**, not CyberSand image output.

**Custody gap:** model binaries, raw receipts and gallery are in the reproducible external local bundle; **not yet immutable GitHub Actions artifacts**. No model binary is committed to Git. Workflow ownership remains separate from this lane. PR remains Draft/HOLD; no merge or Ready.
