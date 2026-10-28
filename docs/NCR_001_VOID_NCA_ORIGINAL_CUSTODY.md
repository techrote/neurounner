# NCR-001E void.1 original one-step NCA diagnostic custody (2026-10-28)

**Draft, frozen source-bound conformance diagnostic only; final #72 and training admission remain blocked.** Parents [#67](https://github.com/techrote/neurounner/issues/67), [#72](https://github.com/techrote/neurounner/issues/72), [#73](https://github.com/techrote/neurounner/issues/73), [#79](https://github.com/techrote/neurounner/issues/79) and [BYTE-EOS #92](https://github.com/techrote/neurounner/issues/92).

## Exact source and physical execution

Source implementation [draft #172](https://github.com/techrote/neurounner/pull/172), head
`2810e800ef890f0c99f9ea44adca29f879b5d0ea`.
[Original complete Actions run 40765025332](https://github.com/techrote/neurounner/actions/runs/40765025332) completed both jobs successfully:

| Original job | ID | Exact artifact ZIP / SHA-256 |
|---|---|---|
| Source-only sentinels (11 pass) | `123086735654` | [12850046593](https://github.com/techrote/neurounner/actions/runs/40765025332/artifacts/12850046593) / `b153d177497f15227545bfe6b717a6430219a662f2b0b868e7a68b89682a90d0` |
| Original pinned Qwen3 physical host | `123087080911` | [12850675205](https://github.com/techrote/neurounner/actions/runs/40765025332/artifacts/12850675205) / `8e9152042fb909f18b13589e6e21552c40f3104c4cc292cf9a643c600fbf08b5` |

Both **exact original ZIP byte strings** are committed in `experiments/ncr-001/controls-evidence/void-nca-learning-20261028/` as base64; both **complete decoded original GitHub job logs**, raw internal transcripts, the original 5,697-byte Qwen3 physical `receipt.json`, and source tests are retained losslessly inside those files. No summary replaces the originals. `manifest.json` pins every ZIP and member byte size/SHA, Actions run/job/artifact identity, model source, runtime and quantitative results. Independent offline `scripts/ncr_001_void_nca_original_custody.py` checks SHA-256, exact member set, original semantic checks, non-admission and 14 adversarial falsifications. Exact-head CI also re-downloads original GitHub Actions artifacts and job logs and byte-compares them with Git custody; until that separate CI succeeds, do not call the archive independently confirmed.

**Physical runner:** GitHub Actions `1000046808`, Linux `7.0.0-1017-azure-x86_64`, Python `3.12.3`, PyTorch `2.14.1+cpu`, Transformers `4.57.6`, PID `2554`, 2 torch threads; recorded host RSS `748260 KiB`, runtime `3.10979 s` within the job. Original full `dotlabs/void.1` checkpoint 360,622,744 bytes SHA `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`, actual ByteTokenizer Python SHA `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`, future causal-EOS serializer from immutable #163 `9fbdee880697903aaf5b91cc29442c9e53479e50`. Both four-lane protocol and pins remained byte unchanged.

## Measured one-step synthetic conformance

One **public synthetic 31-unit sample** with a 10-unit UTF-8 prompt and 21 supervised completion/EOS units, genuine terminal EOS **ID 258**, was used. No held-out case or preregistered TRAIN-003 fixture was accessed. The zero NCA and disabled original Qwen3 logits were bit-exact. A single **NCA-only in-memory SGD step** (`lr=0.005`, *not* NCA/MLP comparative training) produced a nonzero readout gradient with zero upstream gradient before warmup. Subsequent backward-only evaluation yielded finite, nonzero readout/observation/perception/rule/input-normalization gradients and three distinct ordered 8×8×4 depth fields. The complete frozen original Qwen3 named tensor fingerprint remained identical before/after: `7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904`. Frozen disabled logits remained bit-exact after the step.

The one synthetic native causal-EOS loss was `3.065654993057251` before the diagnostic step and `1.981663703918457` afterwards. **This is not a model-quality experiment, a heldout improvement or learned EOS stopping:** it is a scalar overfit diagnostic of gradient reachability on one training example. The original log retains the harmless `torch_dtype` deprecation warning. No MLP gradient/update, NCA-vs-MLP comparison, adapter checkpoint, repeated optimizer schedule, new frozen model, or learned semantic behaviour was performed. **The earlier negative observation of no EOS emission in 12/12 four-step max-context generations remains unchanged.**

## Boundary and next review

The physical result is explicitly
`PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED`.
It provides original-host evidence for the **previously missing three diagnostic learning/update/loss checks** identified in #171, but no independent protocol/scientific review or fully reconciled final four-model acceptance has occurred. Independent evaluator must verify the exact new composed source, all original archive bytes, and source/runtime dependency sequence before revising the #72 lane matrix. Do not use this source diagnostic as authority to edit the frozen protocol or mark the overall source/training admission gate open.

`experiments/ncr-001/protocol.json: train_authorized=false` remains authoritative. Keep both #172 and this custody PR draft; do not merge, access sealed heldouts, retrain models or launch any comparative NCA/MLP experiment.
