# NCR-001 void.1 2048-byte post-lease cache + generation qualification

**Draft diagnostic proposal only.** Parent #67, four-model #72, matched control #73, void integration #79 and BYTE-EOS #92. This is a strict descendant of draft #168, original head `4832ce0fe833dfd70098b7a6086c7276f4ad835e`. It imports no new variant checkpoint, tokenizer, fixture, optimizer or evaluation definition. **`train_authorized=false`**, and four-lane full #72 admission remains blocked.

## Distinct from earlier physical evidence

- [#128](https://github.com/techrote/neurounner/pull/128) ran 512/1024/2048 against the original *pre-lease* void NCA wrapper, with no matched MLP or corrected EOS helper. Its original successful evidence remains valid only for that source.
- [#167](https://github.com/techrote/neurounner/pull/167) and permanently archived [#168](https://github.com/techrote/neurounner/pull/168) execute the composed shared-decoder NCA+matched-MLP and separate cold adapter replay, but cover only **32/128** input units for active cached parity.
- This new runner `scripts/ncr_001_void_postlease_maxctx.py` uses the exact **post-lease source** in #168, calls the previously audited `ncr_001_void_eos_postlease.run()` (with source blob-lock), and performs **both** active original-model adapters at 512/1024/2048. A fresh actual frozen host execution is required; green historical badges cannot be reclassified.

## Frozen authority and checks

Pinned `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`, full Qwen3 model safetensors SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`; reviewed ByteTokenizer source SHA-256 `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`; Transformers 4.57.6. Corrected EOS helper is sourced read-only from #163 exact `9fbdee880697903aaf5b91cc29442c9e53479e50`. Composed decoder-root lock and NCA/MLP source blobs are checked against #168. Pinned four-model protocol and source-pins Git blob IDs remain unchanged.

The new synthetic-only physical plan requires:
1. Independent 512/1024/2048 ordinary ByteTokenizer unit source (mixed UTF-8 text) and raw frozen Qwen3 vs disabled vs zero adapter **exact** logit parity for each arm.
2. Fixed manually perturbed **untrained** NCA and gated-MLP readouts, nonzero finite effect, then `length−8` cached prefill plus **eight genuine one-unit Qwen3 KV-cache increments** at all three lengths for both adapters. Require original, unrelaxed `atol=rtol=0.0005`, and no per-token NCA field carryover.
3. Strict last-unit 2048-context future-suffix isolation: unchanged earlier logits **bit-for-bit**.
4. Real `GenerationMixin.generate()` for each adapter, disabled/zero/active untrained, cache on/off, explicit masks, EOS ID 258, fixed 512-unit prompt and four output units. Require same-cache zero-vs-disabled **exact** scores/IDs and cache-on/off scores within the fixed tolerance, identical IDs. Report generated units without competence or EOS stopping interpretation.
5. Frozen-backbone full parameter-byte fingerprint unchanged, no base gradients, no residual installed decoder hooks. Refuse source/pin/training-gate drift. The runnable model step has a strict 3600s watchdog; retain partial status and traceback on failure.

**Important limitation:** 2048 units here are an *inference/cache diagnostic*, **not** a 2048-unit supervised EOS target case. The separately preregistered void training sample unit cap is 1024, and the future EOS serializer's own no-truncation cap remains unchanged. The original #167 three synthetic causal-EOS forward losses and separate-process cold reload remain source-bound to their own receipts; they are not rerun by this script.

## Reproduction, custody and admission

Workflow: `.github/workflows/ncr-001-void-postlease-maxctx.yml`. Its source job runs offline no-model adversarial tests. Its physical job verifies the exact head, protocol source and independent EOS helper, downloads and hashes the frozen model/tokenizer, executes offline original Qwen3, and always uploads logs, partial `maxctx.json`, runner identity and negative results. The model is **not trained**. The workflow and tests are new and must be inspected at the *exact final head*.

The Actions artifact ZIP is only transient staging. Before numerical qualification may be promoted, independently download the original completed job logs/ZIPs, hash every member, archive them in Git, and run a separate byte/semantic auditor at the original source. **No independent scientific review, source-bound full #72 acceptance, complete tool/memory behavior acceptance, held-out access, comparative training, or merge is authorized by this PR.** Keep `train_authorized=false` and all draft dependency gates intact.
