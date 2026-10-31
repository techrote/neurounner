# NCR-001E — void.1 full 2048-context and HF generation reviewer

Parent issue: [#72](https://github.com/techrote/neurounner/issues/72).  
Reviewer draft: [#128](https://github.com/techrote/neurounner/pull/128).

## Decision

**PASS this finite void.1 source/runtime pre-training subgate.**  
This closes the explicit earlier evidence limit `full_2048_context_proven=false`.

It does **not** grant final #72, train a model, reinterpret historical BYTE-EOS
behavior, integrate the future EOS serializer, or authorize an optimizer.

## Exact pinned source

- `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`
- full frozen `model.safetensors`: 360,622,744 bytes,
  SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`
- independently reviewed/executed `tokenization_byte.py` SHA-256
  `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`
- original frozen Qwen3 NCA hook commit
  `5fa3dc8c0de153098d1800f7b0c809819620aef4`, taps `[7,15,23]`
- PAD/BOS/EOS IDs `256/257/258`; vocabulary 259
- pinned maximum context: **2048 byte-tokenizer units**
- fixed cache gate: `5e-4` absolute and `5e-4` relative

## Original actual-model execution

[Actions **40605678381**](https://github.com/techrote/neurounner/actions/runs/40605678381)
is the authoritative physical model run.

- source job `122544450864`: PASS
- actual frozen model job `122544503703`: PASS
- branch head during execution:
  `0bbc85270693be0ae3f6cee229976071eeeaf001`
- Actions merge checkout:
  `74e8a07f3e2a97748db8de4228c11ae985097e0d`
- runner `GitHub Actions 1000045826`, PID 2745, 4 CPUs
- Python 3.12.3, PyTorch 2.14.1+cpu, Transformers 4.57.6
- peak RSS 1,223,536 KiB; wall time about 79.97 s
- **zero optimizer steps**, no backward pass, no heldout evaluation

A long mixed-UTF8 source string tokenized to 4,440 ordinary byte units. The
reviewer deliberately selected the **first 2048** units; the tokenizer warning
about the full 4,440-unit source is therefore not a model-overlength call.

## Fixed 512 / 1024 / 2048 evidence

Frozen reference versus zero-initialized NCA logits are **exact** at all three
lengths.

After a single manually activated, **untrained** readout scalar, full forward
is compared with an `N-16` prefill plus **16 genuine sequential one-byte KV
cache forwards**:

| Context units | Full vs cached-tail max abs |
|---:|---:|
| 512 | **1.1444091796875e-05** |
| 1024 | **1.049041748046875e-05** |
| 2048 | **1.6689300537109375e-05** |

All are far below the frozen 5e-4 absolute/relative gate.

Additional max-context checks:

- strict future-suffix influence at 2048: **0, exact**
- independent A→B→A request reset: **0, exact**
- manually activated untrained effect at 2048:
  `0.06541299819946289`
- frozen backbone tensor SHA before/after:
  `2b24c42bf3c746514e56f574cac8f4698e99d4dc80eae3450b212bb63c9c4ac5`
  — unchanged
- frozen base parameters: 90,148,352
- untrained NCA parameters: 329,944

## Actual Hugging Face GenerationMixin

A 512-unit prefix runs the original wrapper's actual `generate()` implementation,
not a handwritten decoder loop, for:

- source disabled
- zero readout
- manually activated **untrained** readout

Each runs with both `use_cache=false` and `use_cache=true`, four generated
units, fixed source EOS 258.

All six generation paths emit the same IDs:

`[108, 111, 97, 100]` — ASCII `load`

Frozen source versus zero-readout processed scores are **bitwise exact**. The
largest cache-on/cache-off processed score difference over all modes/steps is:

**7.271766662597656e-06**

Each processed score has 258 finite logits and one identical intentionally
suppressed EOS score.

**EOS 258 was not emitted in these four units. This is not an EOS propensity or
historical stopping-rate estimate.** The reviewer explicitly stores
`historical_byte_eos_impact_measured=false`.

## Durable original evidence

The original Actions artifact #12779509677 has ZIP SHA-256

`4774374596d6b4a74be7e6186ca561c2cbfbd8217890387a78bffb59a6bbf88a`

and size 7,480 bytes. The ZIP remains in Actions storage. All **six original
text members** are preserved byte-for-byte in a deterministic USTAR/XZ archive:

[`experiments/ncr-001/evidence-review/void1-2048/`](../experiments/ncr-001/evidence-review/void1-2048/)

Important originals:

- `void1-2048-context.json`: 14,433 bytes,
  SHA-256 `ca560b77f8a7ca2ca5cf3a39d31460f15ca6e3c18892d0c7bed3a3177089b67e`
- `void1-2048-prefetch.json`: 1,165 bytes,
  SHA-256 `7912faf90c9b152af14e6d92b6584789dbe4802f6b95e1b2d8134263db02dcd5`
- static tokenizer source audit, package inventory, exact source checkout and
  SHA manifest are also retained
- deterministic XZ SHA-256
  `e023c81627d7f3a3ebc07d4d4c4a1bd835276c4019990b16be1535c793b090a9`

The standard-library
[`ncr_001e_void_2048_archive_audit.py`](../scripts/ncr_001e_void_2048_archive_audit.py)
rechecks the original six-member archive, all model/tokenizer/config bytes,
runner identity, exact 512/1024/2048 metrics, six generation records, frozen
tensor hashes and no-training/EOS-scope claims without executing a model.

### Preserved audit failures

Reviewer/auditor failures remain evidence rather than being removed:

- **40606479851 FAILED**: provenance/auditor transcribed the 64-character
  protocol SHA without one `b`. The original Actions hash inventory was
  correct; no model evidence failed.
- **40606704857 FAILED**: the auditor referenced invented `runtime/limits`
  fields in the static tokenizer-source JSON instead of its actual top-level
  schema. Original source/model JSON was unchanged.
- **40606809785 PASS**: corrected auditor verifies exact original six-member
  evidence and all negative tests.

These are archive-verifier defects only. The canonical physical model result is
still **40605678381**.

## BYTE-EOS relationship

Independent BYTE-EOS draft #126 accepts the **future supervised label serializer
source** against the exact ByteTokenizer. That acceptance is separate from this
inference/conformance evidence.

This reviewer does not:

- integrate that serializer into training;
- modify TRAIN-002b;
- claim historical literal-`<eos>` caused repetition or budget exhaustion;
- use four generated bytes to estimate EOS behavior.

Historical BYTE-EOS effect remains **UNMEASURED**.

## Acceptance boundary

The previous void evidence limit `full_2048_context_proven=false` is now
resolved for frozen inference/cache conformance.

Final void #72 nevertheless remains blocked until all protocol-required
learning/update/loss checks can legitimately run. `train_authorized=false`
remains unchanged, so no backward/optimizer step is permitted.

No matched NCA-vs-MLP experiment, trained adapter, heldout behavior or merge is
authorized by this result.
