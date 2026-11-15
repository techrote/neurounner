# MEMPLANT / DREAMGRAFT-001 — G0 real-model interface probe

**Status:** experimental G0 source. A real-model result requires a completed GitHub Actions run. No G1 calibration campaign or cross-model semantic-transfer claim.

This implementation starts from `main` on an independent branch, keeping the unrelated documentation proposals [#228](https://github.com/techrote/neurounner/pull/228) and [#229](https://github.com/techrote/neurounner/pull/229) Draft and unmerged.

## Actual experiment

1. Load frozen [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) at immutable revision `7ae557604adf67be50417f59c2c2f167def9a775`. Process three synthetic donor-only worlds, inspect actual K/V cache geometry, pool four layers × four temporal bins of V state, hash and record. Real cache readback is mandatory; synthetic tensors cannot be silently substituted.
2. Unload donor weights and collect garbage before loading B.
3. Load frozen [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) at `12fd25f77366fa6b3b4b768ec3050bf629380bac`.
4. **Oracle T**: give B full source text only in this identified baseline; compare correct and counterfactual answer loss.
5. **Oracle I**: optimise eight B-native continuous input positions directly to verify the soft-prefix interface and gradient path. Fitted-prefix learning on known examples is only a diagnostic.
6. With both base models frozen, fit the `V-pool → 64-wide MLP → 8 × 576-dimensional` adapter for **five masked-loss updates only**, recording nonzero gradient, time, peak RSS, and code/model hashes. This is G0 compute measurement, not G1.
7. Verify actual B-native KV continuation is reproducible for repeated identical prefix/prompt; record zero and donor-swap exploratory controls.
8. Generate **nine raw dream postcards**: three unfamiliar creative questions × true donor, swapped donor and zero-prefix. Keep all tokens and text, not curated winners. These are unscored exploratory artifacts, **not** verified transplant evidence.

## Run

A dedicated [Actions workflow](../../../.github/workflows/memplant-g0.yml) triggers from the implementation PR. It runs an offline source/contract guard followed by actual public-weight CPU inference/training on `ubuntu-24.04`. The native job has a 28-minute job cap and 19-minute process cap.

Within an isolated Python environment having CPU PyTorch 2.5.1, Transformers 4.57.6, huggingface_hub, psutil and safetensors:

```bash
python -m unittest discover -s tests -p test_memplant_g0.py -v
python -m experiments.memplant.dreamgraft001.g0_live --out runtime/memplant/g0/report.json
```

## Evidence and boundaries

The G0 report records immutable weight/tokenizer checksums, source geometry, all three cache hashes, actual oracle losses, adapter gradients/step times/RSS, native continuation token IDs, the entire 9-card raw postcard grid, and explicit interpretation gates. The workflow retains only small JSON/Markdown artifacts for 30 days; no full weights, private source contexts or credentials go into Git.

A fitted native prefix, numerical output change or five-step loss reduction is **not** evidence of contextual memory transplantation. A weak full-text Oracle T indicates the recipient may be unable to solve the relevant task even without compression.

**Gate policy:** G0 instrumentation and bounded profiling are authorised by the owner. G1 (24–48 examples with embedding, hidden-state and source-swap baselines) and G2–G5 remain independent future authorisation decisions. No automatic merge or follow-up workflow is enabled.
