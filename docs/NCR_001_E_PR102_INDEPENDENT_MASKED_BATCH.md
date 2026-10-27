# NCR-001E — independent PR #102 masked batch and cache reviewer prepass

Parent [#72](https://github.com/techrote/neurounner/issues/72);
research [#67](https://github.com/techrote/neurounner/issues/67);
reviewed [draft PR #102](https://github.com/techrote/neurounner/pull/102).

## Exact review boundary

- Fixed reviewed PR #102 head **`3592841802d85c442ca2521479b5d8795bee7233`**.
- Exact publisher-source commit `a003767e70d766bc14b40c89792543ee8641c8bc`, public SFT source SHA256 `e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69`. This is *source* identity, not cryptographic proof the public source generated the released training checkpoint.
- Original frozen adapter hook **`b95c010f9b2ba3cbd9f81317981316bb9c8acd19`** (root Instruct, taps 7,16,25).
- Exact **root** `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668` `model.safetensors`, 1,002,207,984 bytes, SHA256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. The Chat sibling SHA `551c6e26...` is forbidden.
- Four protocol keys remain `smollm2`, `sprocket_chat`, `sprocket_instruct`, `void1`. **`train_authorized=false`; no optimizer; no held-out fixture or comparative training.**

## Independent source/runtime questions

PR #102 demonstrates 108-position unpadded source dialogue cache parity, explicit host/action role masking, and 21 preserved synthetic generation negatives. The following source-structured **real frozen model** checks have not been independently demonstrated on that PR head:

1. Build the source-defined system/memory/user/assistant tool-call/host tool-result/assistant memory-write/assistant memory-read dialogue **without importing the reviewed serializer as authority**, then compare its actual tokenizer IDs and assistant-mask ownership to the exact PR #102 serializer. The second, unrelated ordinary dialogue is much shorter and never contains tool tokens.
2. Evaluate both dialogues together under **left and right** padding. Compare direct frozen-backbone and zero-NCA logits exactly, and activated-but-untrained NCA valid-position logits to each single-request result at fixed, unchanged absolute+relative `5e-4` tolerance. Record the numerical maximum deviation rather than just `allclose` status.
3. Change masked padding from token ID 1 to real host `<|tool_result|>` ID **7**. **Valid outputs must be bitwise unchanged**. Mutate the other user's real input while asserting cross-batch request independence. Include a *deliberate invalid-mask negative sentinel* proving that the harness detects unmasked padding.
4. On the left-padded batch, use the full attention masks and explicit position IDs for prefill and **four genuine one-token KV-cache forwards**. Compare both rows' final four positions to the full-sequence reference at source-prescribed 5e-4 absolute and relative gates; verify per-call state shape `[2,1,...]` and no remaining model hooks.
5. Independently hash the full original checkpoint and all source metadata, hash frozen tensors before/after the finite test, preserve runner, memory and any traceback on failure. **Zero optimizer steps**, no quality scores and no new held-out cases.

The new independent reviewer runner [`scripts/ncr_001e_pr102_masked_batch_review.py`](../scripts/ncr_001e_pr102_masked_batch_review.py) is deliberately disjoint from the original `scripts/ncr_001e_sprocket_wire_diagnostic.py` and does not reuse its original 21 generations as fresh runtime acceptance evidence. [Negative source-only tests](../tests/test_ncr_001e_pr102_masked_batch_review.py) execute without weight downloads. [Hosted reviewer workflow](../.github/workflows/ncr-001e-pr102-independent-batch.yml) separately executes real frozen root Instruct and uploads original partial JSONs on success or failure.

## Interpretation

Successful reviewer execution **would only add one source-identified, finite real-host masked/padded causal check to the Instruct lane**. It would not qualify learned tool use or memory persistence; the released Sprocket model card reports memory as weak. It would not independently admit the four-model #72 protocol: gradient and controlled optimizer-step checks remain blocked by the explicit training-admission requirement, and SmolLM2, Chat, void.1/Byte-EOS and matched-MLP controls have distinct unmerged dependencies.

**All PRs remain draft and unmerged. `train_authorized=false` remains unchanged; any failing source or causal test blocks this subgate instead of relaxing it.**

## Fresh original frozen-model result and source custody — 27 October 2026

**Actual original [GitHub Actions run 40590487839](https://github.com/techrote/neurounner/actions/runs/40590487839): BOTH jobs PASS**. Offline source-only reviewer passed **six negative/positive groups** before loading model weights; original full root Instruct frozen-model job ID `122493234132` succeeded separately. This was an original model execution, not readback of PR #102's prior JSON.

- Reviewer code during the original execution: `07eaac38a857a8d67285166944c91b7bc4543e9c`; original Actions merge SHA `724c05a508a2f53f7752198e7ddad1d14d6902c2`; reviewed PR #102 source fixed at `3592841802d85c442ca2521479b5d8795bee7233`, original hook `b95c010f9b2ba3cbd9f81317981316bb9c8acd19`.
- Source model independently re-downloaded and full SHA-256 verified: root Instruct original 1,002,207,984-byte `model.safetensors` SHA `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`; **different** Chat weight hash `551c6e26...`. Python 3.12.3, torch 2.14.1+cpu, 4 CPU threads, GitHub Actions runner `GitHub Actions 1000045500`, PID `2909`; maximum RSS **3,276,648 KiB**.
- Two independently source-serialized, **synthetic** prompts: **133 Instruct tokenizer units** containing full host/assistant control framing (45 assistant-supervised units), and a **31-unit** ordinary second conversation (8 assistant-supervised units).
- Frozen reference versus zero readout, left and right padding: **exact**, max drift `0`. Activated but **untrained** readout changed logits by `0.10165786743164062`, never by optimization.
- Both padded orientations' valid logits versus independent per-request forwards: **exact**. Replace *masked* padding positions with source tool-result ID **7**: **zero valid-position drift**, exact. Alter the unrelated second user's last real token: **zero change to first batch member's logits**, exact.
- **True batched masked-cache test:** full left-padded, publisher-structured `133`-position forward versus `129`-prefill + **4 sequential one-token cached** forwards, max absolute logit error **`3.218650817871094e-05`**, below **unchanged** protocol tolerance `5e-4` absolute and `5e-4` relative.
- Negative sentinel: deliberately *unmasking* the second example's many leading padding positions measurably changed its valid logits (max **`14.123714447021484`**). This is expected and proves the reviewer harness detects a false masking assumption.
- A→B→A request reset **exact**. Original backbone parameter hash identical before/after (`b73d854127c43266d123bdb266db63a15b4189db9bb3398be035463cbb4fac4d`). Frozen base **501,090,560** parameters, separate untrained NCA **658,904** parameters, gradient-free inference only, **optimizer steps zero**. No leaked original model hooks.

### Lossless actual original receipt archival

- [Original Actions artifact #12773497808](https://github.com/techrote/neurounner/actions/runs/40590487839/artifacts/12773497808), full ZIP SHA-256 `d0328b54347b952534ef1321953c5c94f6b66c48c791aa72e67bf2cf51d56b42` (2,975 bytes). The ZIP remains in Actions storage pending expiry; do **not** claim its bytes were committed.
- Its **unaltered original** actual-model JSON is **6,721 bytes**, SHA256 `1757dc6836b974d74d32acf6c45ccece4d1f69f980030b3bdffc8fad216f403e`. Its original exact source prefetch JSON is **874 bytes**, SHA256 `0272d25c42331841ce62ec26156b0805adb38f4a52e5258bac5c6272cbad538a`.
- Both complete JSON files are **losslessly preserved byte-for-byte** in [gzip/base64 original USTAR archive](../experiments/ncr-001/evidence-review/pr102-masked-batch/original-jsons.tar.gz.b64) (gzip SHA `ee01823ebccb19f7d417096db7b9819dc40d0b86ad690d4436dce9765de72460`), with [original model/hook/runner/ZIP provenance](../experiments/ncr-001/evidence-review/pr102-masked-batch/provenance.json).
- The separate [standard-library original-receipt auditor](../scripts/ncr_001e_pr102_masked_batch_audit.py) refuses source variant swaps, altered file/runner digests, incorrect masked-batch/cache numerical gates, corrupted source archive and false training/qualification claims. [Hosted archive-only CI 40591376322](https://github.com/techrote/neurounner/actions/runs/40591376322) **PASS**, seven positive/negative groups; **this is not a second actual-model execution**.

### Review decision and remaining blocks

**Result: PASS for this finite original frozen Instruct source-wire masked-batch/cache prepass only.** No claim of novel tool/memory competence: all source controls were synthetically host-provided; the publisher's negative memory-learning claims remain intact. **Not final independent #72 admission**, which requires the full frozen four-lane model checklist and the separately controlled gradient/update gate; no optimizer was run because `train_authorized=false`. Unmerged/preregistered model hooks, Chat sibling, SmolLM2, void.1/Byte-EOS and matched-MLP controls retain their independent gates. Neither source/code nor its original sealed holdouts have been altered. All stacked PRs remain draft/unmerged.
