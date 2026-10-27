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
