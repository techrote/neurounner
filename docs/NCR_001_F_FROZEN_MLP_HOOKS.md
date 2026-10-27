# NCR-001F — Frozen-model conventional MLP at exact depth taps (no training)

Parent [#67](https://github.com/techrote/neurounner/issues/67), matched-control issue [#73](https://github.com/techrote/neurounner/issues/73). This is a **draft stacked on [#87](https://github.com/techrote/neurounner/pull/87)**, which depends on core [#82](https://github.com/techrote/neurounner/pull/82) and preregistration [#77](https://github.com/techrote/neurounner/pull/77). It does not authorize, initiate or compare trained model adapters.

## What changed

- scripts/ncr_001f_frozen_hooks.py attaches the ordinary per-token MLP from #87 to the **same three post-decoder-block taps** as the NCA. It supports real Hugging Face decoder model.model.layers Tensor or tensor-first tuple outputs. No cellular locality, depth-state handoff, cross-token communication or time recurrence.
- Four **independent** original frozen model identities include repository, revision, precise safetensors path, actual weight SHA-256 reference, architecture/width, tap positions and adapter state identity. Sprocket Chat is **chat/model.safetensors** and Instruct is distinct **root model.safetensors**, never aliases. A caller-supplied hash only *attests* identity: real source runners must independently hash actual bytes.
- Hooks attach only inside a locked forward/generate, assert correct ordered tap visitation and always detach after exceptions. Frozen backbone weights stay untrainable and in evaluation mode. Checkpoints contain only the adapter, reject incorrect source/variant/architecture metadata, and do not store LM weights.
- tests/test_ncr_001f_frozen_hooks.py covers all four model geometries on synthetic decoder structures, no-op and generation parity, manually activated untrained residual, gradient-only frozen-base assertions, request reset, state reload, wrong-variant refusal and exception/concurrency cleanup.
- scripts/ncr_001f_real_smollm2.py separately downloads and hashes the **actual exact SmolLM2 original checkpoint** and three tokenizer/config files, then verifies zero-init 16/64/128 position logits, left/right masked batches, short generation, gradient-only readout (no optimizer), active *untrained* readout, true 48-token incremental KV cache, causal future-suffix stability, A→B→A request reset, different adapter-instance cold reload on the **same frozen base** and wrong checkpoint metadata refusal. It hashes all loaded original base tensor bytes before and after.

## Four within-model adapter identities

| Frozen original model | Weight path | Width | Post-block 0-based taps | MLP trainables | NCA trainables |
|---|---|---:|---|---:|---:|
| SmolLM2-135M Instruct | root model.safetensors | 576 | 9,19,29 | 296,576 | 297,048 |
| Sprocket-500M Chat | **chat/**model.safetensors | 1,280 | 7,16,25 | 658,432 | 658,904 |
| Sprocket-500M Instruct | root model.safetensors | 1,280 | 7,16,25 | 658,432 | 658,904 |
| void.1 Qwen3 | root model.safetensors | 640 | 7,15,23 | 329,472 | 329,944 |

Absolute difference is **472 trainable parameters** (all within preregistered ±5%). This is **parameter matching, not FLOP/time matching**. Cross-model raw token speed, CE and quality are not directly comparable.

Exact frozen source revisions: SmolLM2 **12fd25f77366fa6b3b4b768ec3050bf629380bac**; separate Sprocket Chat and Instruct **eb157d4ddddc4619e661d4a2426bb2dd000fc668**; void.1 **9c2be5087772867d45961abd11f6b35110617563**. SHA-256 values are in the source and original experiments/ncr-001/source-pins.json.

## Execution and original evidence preservation

The bounded GitHub workflow .github/workflows/ncr-001f-frozen-hooks.yml runs a four-lane synthetic hook-gate job followed by an original-model SmolLM2 4-vCPU CPU job capped at 30 minutes. It installs PyTorch 2.14.1 CPU, Transformers 4.57.6, huggingface_hub 0.36.2, and safetensors 0.8.0.

Run synthetic guards:

    python3 -m unittest discover -s tests -p 'test_ncr_001f_frozen_hooks.py' -v

Run pinned actual frozen SmolLM2 (after installing package pins and permitting original weight download):

    python3 scripts/ncr_001f_real_smollm2.py

The full machine-readable runtime/ncr-001f/real-smollm2-mlp.json stores actual model/checkpoint/file SHA and sizes, code/protocol digests, host identity, original full loaded tensor fingerprints before/after, numeric errors, peak RSS, exceptions/failures, all negative flags and zero optimizer steps. CI uploads the unmodified JSON on both success and failure. **Actions artifacts are transient**; after successful source-bound execution, promote the *exact original* receipt, run source commit and artifact ZIP digest under experiments/ncr-001/ with independent hash readback. A passing badge or derived prose summary does not constitute archival custody.

## Explicit limits and follow-ups

**train_authorized=false** and all independently accepted final #72 gates remain **blocked**. No optimizer constructed, training step, learned quality comparison, held-out data access, fixture modification, tool-memory grammar assertion or experimental admission. SmolLM2 is the **only original frozen model** actually checked by the new MLP source runner; Sprocket Chat, Sprocket Instruct and void.1 currently have **synthetic-layout MLP hooks only**. Same-process new-adapter reload is not a distinct-process restart. Do not confuse this with separate NCA source prepasses [#89](https://github.com/techrote/neurounner/pull/89), [#93](https://github.com/techrote/neurounner/pull/93), [#95](https://github.com/techrote/neurounner/pull/95), or BYTE-EOS-001 [#94](https://github.com/techrote/neurounner/pull/94).

Next: independent source/readback of the MLP receipt, original-weight MLP invocations for the remaining model lanes, exact dependency reconciliation and independent #72 final acceptance before considering any separately authorized comparative training.
