# NCR-001 void.1 corrected-EOS + post-lease NCA/MLP original-host witness

**DRAFT / NO TRAINING / NO #72 ADMISSION.** Parent issues #67, #72, #73 and BYTE-EOS #92.

## Dependency and source boundaries

This branch is stacked only on PR #162's exact frozen head 4cf8c052c5ad91c384f2e15b68793a92c9a80cfe, which contains the shared decoder-root lock from #154, four NCA wrappers and matched conventional gated-MLP hooks. For future EOS labels, CI checks out independent draft PR #163 at immutable commit 9fbdee880697903aaf5b91cc29442c9e53479e50 as a **read-only second source tree**. It does not merge or cherry-pick #163–#166. The original #165 forward-loss receipt and #166 evidence archive are not overwritten.

The exact original model is dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563, weight SHA-256 ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3 (360,622,744 bytes), ByteTokenizer Python SHA-256 279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03. Source EOS ID 258, PAD 256, BOS 257. Transformers 4.57.6; eager float32 CPU; three post-decoder taps 7, 15, 23. This is one distinct frozen-model lane and must not be confused with either Sprocket variant or SmolLM2.

## New qualification checks

1. Refuse altered protocol/weight/source/variant identity, any training authorization, incorrect independent serializer revision, or online model execution; verify full original HF files by SHA-256.
2. Against original Qwen3, verify nonempty prompt, explicit attention mask, UTF-8 completion bytes, completion-only labels, terminal integer EOS ID 258 and supervised targets *after* one-position causal shift. Use precisely the three synthetic cases from PR #165, not sealed held-out fixtures.
3. Compare original Qwen3 native finite supervised loss with independently shifted cross-entropy, then verify exact zero-initialized and disabled-reference Qwen3 logits/loss for NCA and conventional MLP separately.
4. Confirm cross-adapter ownership refuses a disabled frozen-reference call during a competing instrumented forward. After **manual untrained** adapter readout perturbation, verify nonzero finite effect, true 32/128-byte-token KV-cache incremental parity (unchanged 0.0005 absolute/relative tolerance), A-B-A request isolation and causal future-suffix invariance.
5. Repeat untrained adapter loading in a **distinct Python interpreter**; demand byte-exact metadata and adapter-state hashes and identical logits. Fingerprint the original frozen backbone before and after both executions.

This probe does not exercise the full 2048-byte Qwen3 context, behavioral stopping/EOS competence, learning after an update or all #72 requirements. It does not claim that the adapted NCA or MLP improves the model. All other lane/admission decisions remain separate.

## Reproduce and preserve negative evidence

Workflow: .github/workflows/ncr-001-void-postlease-eos.yml. Source tests run without the model using tests/test_ncr_001_void_eos_postlease.py. The original host runs scripts/ncr_001_void_eos_postlease.py in producer and consumer phases under HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 after SHA-verified prefetch. A second checkout named byte-eos-authority must contain PR #163 at its exact pinned commit; NCR_EXPECTED_HEAD must match the *source branch HEAD*, not the synthetic GitHub merge commit.

CI preserves source/prefetch/producer/consumer full logs, partial JSON receipts, independent process IDs, original pinned hashes, untrained adapter binaries and failures as artifacts. **Artifacts are transient and not final Git custody.** Original ZIPs, decoded logs, runner details and exact source identity must be promoted without modification and checked by a distinct source-bound evidence auditor before drawing independent conclusions. A green CI badge by itself is insufficient.

No model comparison training, optimizer, backward, changed preregistration, opened held-out evaluation, historical TRAIN-002b rewrite, PR merge or scientific acceptance is authorized. The locked protocol continues to contain **train_authorized=false**. Leave this PR draft for distinct independent review. Any source/model/cache failure is a negative result rather than grounds to weaken the gate.
