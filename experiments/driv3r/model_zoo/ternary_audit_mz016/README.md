# MZ-016 — frozen recurrent-policy low-bit compression screen

**Research only. Draft/HOLD under [Neurounner PR #252](https://github.com/techrote/neurounner/pull/252).** No merge, Ready promotion, native gameplay modification or production inference claim.

## Research question

MZ-015 trained a 209,012-parameter diagonal-state policy and a 237,428-parameter GRU to 32/32 clean completions on direction-balanced synthetic turning paths. Can deterministic post-training ternary (three levels), four-bit or eight-bit weight substitution preserve **closed-loop** PAINT/protected-material behaviour? Which recurrent layers are most fragile?

This is a bounded **model-surgery experiment on two already-trained specimens**, not a new SLM onboarding or training run. It materially advances compression eligibility and identifies a layer-localized failure mechanism. No pretrained weights have been added to Git.

## Source custody and actual execution

- Original frozen [MZ-015 Actions checkpoint artifact](https://github.com/techrote/cybersand/actions/runs/43942288057/artifacts/14265125044) is **read-only external evidence**. Training source commit: `807ac25373c31649dd5a92a9aaa503098a86183b`.
- Input `mz015/models.pt` SHA-256: `0c3d60f6259e9b9504843883c1bc06459720b0789748d22dcdf1ac4bb3071c5f`. The script rejects mismatched input by default, and checks original checkpoint schema/source SHA.
- This audit source Git blob: `9b42923e256ded37c092fe9217d0461840341933`; tests Git blob: `ed92fc12f0587034efff26a228c6eaad42814f8a`. Both Git blob hashes matched the locally executed source exactly.
- Local CPU/PyTorch 2.10, one thread, on 16 November 2026. Source-compatible replicas of the imported MZ-013/MZ-015 model/world modules were used for local execution; source is independently present in this branch. Before the audit, frozen MZ-015 baseline reproduction matched original evidence: diagonal 32/32 clean, GRU 32/32 clean, reactive 10/32 clean with 668 collisions. **No exact-HEAD GitHub CI is claimed for MZ-016.**
- New evaluation: 16 unseen seeds (`9000..9007`, `9100..9107`), crossed with all four cue/inversion settings = **64 closed-loop rollouts per treatment**. Four model-surgery unit tests passed.
- Full compact results, seed list, CPU p50, compression estimates and source hashes are preserved in [evidence-2026-11-16.json](evidence-2026-11-16.json).

## Closed-loop outcomes

| Frozen policy / weight treatment | Clean / 64 | Protected collisions | Total PAINTs | Raw illegal proposals |
|---|---:|---:|---:|---:|
| Diagonal FP32 reference | **64** | 0 | 192 | 0 |
| Diagonal all-matrix ternary, threshold 0.5 | 9 | 1169 | 106 | 383 |
| Diagonal first recurrent layer only ternary | 12 | 1161 | 117 | 20 |
| Diagonal second recurrent layer only ternary | 59 | 10 | 191 | 0 |
| Diagonal head only ternary | 35 | 151 | 156 | 189 |
| Diagonal all-matrix ternary, threshold 1.0 | 18 | 1182 | 109 | 46 |
| **Diagonal all-matrix symmetric INT4** | **64** | **0** | **192** | **0** |
| Diagonal all-matrix symmetric INT8 | 64 | 0 | 192 | 0 |
| GRU FP32 reference | **64** | 0 | 192 | 0 |
| GRU all-matrix ternary, threshold 0.5 | 8 | 625 | 167 | 681 |
| GRU input projections only ternary | 35 | 34 | 192 | 30 |
| **GRU state-recurrent matrices only ternary** | **64** | **0** | **192** | **0** |
| GRU output head only ternary | 63 | 1 | 192 | 0 |
| GRU all-matrix ternary, threshold 1.0 | 27 | 732 | 143 | 244 |
| **GRU all-matrix symmetric INT4** | **64** | **0** | **192** | **0** |
| GRU all-matrix symmetric INT8 | 64 | 0 | 192 | 0 |

All matrix quantization uses independent **per-output-channel** scaling; biases and LayerNorm remain FP32. Ternary sets small values to zero and averages the magnitude of retained weights per row. INT4/INT8 round to a symmetric, per-row max-scaled grid. All transformed weights are evaluated **after dequantization into FP32 tensors**, with the host restricting execution to the supplied legal action menu. Hence illegal raw proposals are not hidden by the host mask.

**Finding:** post-training ternary conversion of *all* weight matrices catastrophically damages both previously competent policies. The first diagonal recurrent layer is a much more fragile part than the second; the GRU's hidden-to-hidden matrices tolerate ternary conversion when left isolated. In contrast, all-matrix INT4 exactly preserves clean completion on the tested turning-path seeds.

The analytical lower-bound **packed** size ratio is 13.84% of FP32 for diagonal INT4, 14.12% for GRU INT4 (including per-row FP32 scales and unquantized parameters); ternary-all ratios are 7.66% and 7.93%. These are not physical runtime savings. The experiment uses the same FP32 kernels for inference, so **no memory, latency, energy or throughput advantage is established**.

## Reproduction

Obtain `mz015/models.pt` from the original [source artifact](https://github.com/techrote/cybersand/actions/runs/43942288057/artifacts/14265125044) (archive member `mz015/models.pt`). From Neurounner repository root:

```bash
python -m unittest experiments.driv3r.model_zoo.ternary_audit_mz016.test_audit -v
python -m experiments.driv3r.model_zoo.ternary_audit_mz016.audit \
  --checkpoint /path/to/models.pt --out /tmp/mz016-evidence
```

The code rebuilds 16 frozen variants deterministically without producing new large checkpoints; its metric JSON is the evidence. Only explicitly invoked CPU experiments run; no new per-experiment push/PR training workflow is created.

## Decision

**Do not promote post-training all-ternary policies.** Prioritize a bounded INT4 CPU packed-kernel benchmark, then, if warranted, a small quantization-aware-training (QAT) intervention focused on the sensitive first diagonal-state layer or GRU input projections. Require independent topologies and counterfactual goal changes before declaring any compressed controller game-policy ready. No result here proves arbitrary spatial planning, native CyberSand compatibility or Windows physical acceptance.

Only `experiments/driv3r/model_zoo/ternary_audit_mz016/**` is owned and modified by this increment. CyberSand remains entirely read-only.
