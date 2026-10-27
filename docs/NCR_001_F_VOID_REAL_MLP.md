# NCR-001F — actual frozen void.1 matched gated-MLP byte-tokenizer prepass

**Parent:** [#67](https://github.com/techrote/neurounner/issues/67). **Control issue:** [#73](https://github.com/techrote/neurounner/issues/73). **Stack:** protocol [#77](https://github.com/techrote/neurounner/pull/77) → NCA core [#82](https://github.com/techrote/neurounner/pull/82) → matched controls [#87](https://github.com/techrote/neurounner/pull/87) → real frozen MLP wrapper [#96](https://github.com/techrote/neurounner/pull/96) → this draft. ByteTokenizer source/runtime authority is separately held by [#90](https://github.com/techrote/neurounner/pull/90)/[#91](https://github.com/techrote/neurounner/pull/91). Void NCA actual-model prepass [#93](https://github.com/techrote/neurounner/pull/93) is **not** a matched-MLP result. All PRs remain unmerged.

## Pinned frozen identity and independent remote-code custody

- Actual HF model **dotlabs/void.1**, revision `9c2be5087772867d45961abd11f6b35110617563`; **root** `model.safetensors`, 360,622,744 bytes, SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`.
- Qwen3 decoder, 24 blocks × 640 hidden, zero-based **post-block taps [7,15,23]**. Existing source-identical noncellular gated-MLP wrapper is fixed at [PR #96 head `f4f4b5becaebf17d89f11dc8db0952c31cbe451f`](https://github.com/techrote/neurounner/pull/96).
- Trainable conventional matched MLP **329,472** parameters vs untrained original NCA **329,944**; **472 fewer** (about 0.143%), but parameter parity is not compute/latency parity.
- Source reviewed `tokenization_byte.py` SHA-256 `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`. Reuses unchanged standalone review function `verify_reviewed_shape` from exact [#91 commit `1fdc2e7e5c7b7b77923dbc4b47dcafa132aef47a`](https://github.com/techrote/neurounner/pull/91). Actual dynamic import occurs **only after complete source-file hashing and AST structural review under HF offline mode**. Hash the *executed* Python module file again after import. Special IDs remain PAD=256, BOS=257, EOS=258; ordinary literal `<eos>` produces five UTF-8 byte IDs, **not** ID 258.
- Compare only against this same frozen model's own logits. Do not convert a Sprocket checkpoint, tokenizer, template, result or character metric into a void.1 claim.

## Bounded nontraining evidence plan

[Runtime script](../scripts/ncr_001f_real_void.py), [source/negative tests](../tests/test_ncr_001f_real_void.py) and [GitHub Actions workflow](../.github/workflows/ncr-001f-real-void.yml) operate on original pinned bytes with four CPU threads, PyTorch 2.14.1+cpu, Transformers 4.57.6, huggingface_hub 0.36.2, safetensors 0.8.0. Bounded 35-minute actual-model job; no optimizer exists and **zero updates** occur.

1. Static-only negative sentinels deny source/variant/revision/geometry/weight/byte-tokenizer drift and any changes to `train_authorized=false`. No held-out material is opened.
2. Download + hash full original checkpoint, config, ByteTokenizer Python, byte vocabulary, tokenizer and special map, writing complete or partial *original* prefetch JSON before invoking remote Python. Require pinned review commit.
3. Execute reviewed UTF-8 tokenizer on a **new synthetic technical sentence** with emoji, Japanese, accents and Greek; prove tokenizer units equal actual UTF-8 bytes, retaining character/byte/unit lengths separately. Do not use frozen evaluation prompts or attempt EOS quality scoring.
4. On original frozen Qwen3 model, verify zero-readout MLP exact frozen logits at 16/64/128 units and left/right masked batches, plus four-unit zero-init greedy parity with actual EOS **ID 258**. Check ordered three-tap hooks.
5. Take one gradient-only readout check against model-sourced synthetic tokens: finite loss, nonzero MLP readout gradient, no frozen base gradients. No optimizer is constructed.
6. Manually activate one untrained readout scalar, compare genuine 48-byte-unit full forward with 48 actual single-unit cached decoder forwards at *unchanged* preregistered `atol=rtol=5e-4`; test exact mutated-future suffix invariance, A→B→A request reset, independent adapter instance restored against **same** original frozen model, wrong adapter metadata refusal, and before/after SHA-256 of **every loaded backbone parameter byte**.
7. Always upload complete available unaltered prefetch/actual-host JSON, including failures and stack trace. Stage artifact is **not** independently audited or durable until re-downloaded byte-for-byte, SHA-bound to Actions run/job/runner/checkout/source revision, archived under `experiments/ncr-001/controls-evidence/` and reviewed using separate negative sentinels.

## Immutable gates and reporting

This is **not** a valid #72 PASS, independent new-process cold replay, full 2048-context test, matched trained NCA/MLP comparison, EOS defect effect estimate, historical TRAIN-002b correction, or admission of any training. [BYTE-EOS-001 #92](https://github.com/techrote/neurounner/issues/92) and future-only [PR #94](https://github.com/techrote/neurounner/pull/94) stay separate. Both successful and failed actual frozen-host results must be archived; no test may silently lower causal thresholds. All four model identities and preregistered held-outs remain unchanged. **`train_authorized=false` remains in force.**

**At creation:** script/workflow/source-negative tests are committed for review; actual frozen-model CI numerical results and permanent raw artifact readback must be verified separately before marking any real-host prepass as passed.
