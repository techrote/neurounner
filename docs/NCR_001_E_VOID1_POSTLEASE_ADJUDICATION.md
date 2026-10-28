# NCR-001E void.1 post-lease #72 adjudication

**DRAFT / source-runtime adjudication only / final gate blocked.** Parent #67 and #72, with matched-control #73, void integration #79 and BYTE-EOS #92 as supporting dependencies.

This decision supersedes neither historical TRAIN-002b nor the earlier pre-lease formal void.1 adjudication. It reconciles the **post-lease shared-decoder wrapper**, corrected future EOS serializer, separate-process cold replay, and the new 512/1024/2048 original frozen-Qwen3 cache/generation evidence.

## Source/runtime conclusion

**Decision:** `BLOCKED_REQUIRED_LEARNING_CHECKS__POSTLEASE_SOURCE_RUNTIME_PRETRAIN_PASS`.

The source/runtime pretraining subgate is supported by two independently preserved physical Actions runs:

- [40747035876](https://github.com/techrote/neurounner/actions/runs/40747035876): corrected causal-EOS forward loss, post-lease NCA + matched-MLP 32/128 cache, request/suffix isolation, shared-decoder reference refusal and separate-Python adapter cold replay. Original complete artifacts/logs are under `experiments/ncr-001/controls-evidence/void-postlease-20261028/`.
- [40755894558](https://github.com/techrote/neurounner/actions/runs/40755894558): post-lease 512/1024/2048 exact frozen-vs-zero parity, genuine incremental cache, strict 2048 future-suffix causality and real `GenerationMixin.generate()` cache on/off for disabled/zero/manual-untrained NCA and matched MLP. Original complete artifacts/logs are under `experiments/ncr-001/controls-evidence/void-postlease-maxctx-20261028/`.

Exact source remains `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`, 360,622,744-byte checkpoint SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`, reviewed ByteTokenizer source SHA-256 `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`, Qwen3 24×640, taps `[7,15,23]`, native context 2048. Corrected future EOS serializer remains pinned to #163 commit `9fbdee880697903aaf5b91cc29442c9e53479e50`; EOS ID remains 258.

## Protocol matrix

| Required #72 check | Post-lease decision | Evidence boundary |
|---|---|---|
| same_base_logits | **PASS** | Exact disabled-vs-zero frozen logits on corrected-EOS cases and 512/1024/2048 contexts |
| base_weight_unchanged_after_step | **BLOCKED_BY_TRAINING_POLICY** | Frozen fingerprints unchanged on forward-only runs, but no optimizer step was permitted or observed |
| causal_suffix_invariance | **PASS** | Exact post-lease suffix isolation including 2048-unit NCA witness |
| full_prefix_vs_incremental_cache | **PASS** | 32/128 + 512/1024/2048 cache and HF generate cache on/off, fixed tolerance 5e-4 |
| per_token_isolation | **PASS_SOURCE_RUNTIME** | Shared decoder-root lease, per-forward field creation and causal/cached sentinels |
| per_request_reset | **PASS** | Original A→B→A isolation |
| nonzero_adapter_gradient_after_warmup | **BLOCKED_BY_TRAINING_POLICY** | No post-lease backward/warmup; manual readout perturbation is not learned gradient evidence |
| finite_state_and_loss | **PARTIAL_BLOCKED_WARMUP_LOSS** | Finite states/logits and corrected causal-EOS frozen losses; no supervised adapter loss after warmup |
| cold_reload_state_dict | **PASS_PRETRAIN** | NCA and matched-MLP adapter replay in a distinct Python interpreter |

Exactly three protocol-required blockers therefore remain: post-step frozen-base invariance, post-warmup adapter gradient, and supervised finite state/loss after warmup. **The final per-model gate remains false.**

## BYTE-EOS boundary

The future serializer's corrected terminal EOS ID and original-host causal forward loss are source/runtime qualified for this stack. This is not evidence of learned stopping. In the 2048 follow-up, all 12 four-step generation records emitted `[101,110,32,109]`; none emitted EOS 258. The causal contribution of the historical TRAIN-002b literal-`<eos>` bug remains **unmeasured**.

## Matched-control boundary

The post-lease gated MLP is source/runtime qualified alongside the NCA for the same pinned Qwen3 host and cache paths, with 329,472 versus 329,944 trainable adapter parameters. No matched NCA/MLP optimizer run, quality comparison, held-out evaluation or effect-size claim has been performed.

## Admission boundary

Do **not** set `void1_conformance_pass=true`. Do **not** open the sealed evaluation fixtures or launch comparative NCA/MLP training. `experiments/ncr-001/protocol.json: train_authorized=false` remains authoritative.

This adjudication is deterministic and source-only. The CI regenerates the machine-readable report from committed original evidence, re-runs the underlying byte/semantic auditors, and refuses false PASS/training/EOS claims. Independent scientific review and stacked dependency reconciliation remain outstanding.

## Adjudication CI negative record

Two exact-head adjudication attempts failed **before the adjudicator executed** and are retained verbatim under `experiments/ncr-001/adjudication/`:

- [40762083896](https://github.com/techrote/neurounner/actions/runs/40762083896), job `123076382167`: shell redirection targeted `runtime/ncr-001/void1-postlease-adjudication.stdout.json` before the parent directory existed. No Python adjudication code ran.
- [40762357199](https://github.com/techrote/neurounner/actions/runs/40762357199), job `123077341445`: the attempted workflow repair accidentally committed a literal `\\n` inside the shell command, so the same pre-execution directory failure remained. No Python adjudication code ran.

The workflow-only correction creates `runtime/ncr-001` with a real shell newline before redirecting output. [Run 40762474121](https://github.com/techrote/neurounner/actions/runs/40762474121) then re-audited the original evidence, regenerated the committed decision exactly and passed all seven fail-closed adjudication tests. Neither failed attempt changed source evidence, tolerances, the machine-readable decision, model identities, or training policy.
