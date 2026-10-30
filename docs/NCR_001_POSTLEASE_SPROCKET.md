# NCR-001 — source-matched post-lease Sprocket Chat/Instruct original-host qualification

Parent epic [#67](https://github.com/techrote/neurounner/issues/67); causal/cache admission [#72](https://github.com/techrote/neurounner/issues/72); matched control [#73](https://github.com/techrote/neurounner/issues/73). This draft is stacked **only** on [PR #154](https://github.com/techrote/neurounner/pull/154) at exact base head `e961a454bc6125d41c0b9b3cb6589f6637e15b61`. It does not modify preregistration, fixture files, comparison thresholds, frozen weights or BYTE-EOS-001. `train_authorized=false` remains a hard source and workflow assertion.

## Reason for fresh physical evidence

Older real-model Sprocket Chat/Instruct cache, generation and matched-MLP receipts exercised **pre-lease source blobs**. They cannot establish that the newly composed #154 NCA and gated-MLP wrappers share decoder-root ownership correctly on an original frozen host. The new runner directly executes those wrappers together over the same frozen model instance, including deliberately rejected contaminated references.

## Exact separate checkpoint identities

| Lane | Pinned actual source and full-weight SHA-256 |
| --- | --- |
| Sprocket Chat | `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, `chat/model.safetensors`, `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97` |
| Sprocket Instruct | Same pinned publisher revision, **root** `model.safetensors`, `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864` |

Both original models have 26 Llama blocks, hidden width 1280 and distinct NCA/MLP parameter instances with post-block taps [7,16,25]. They must never be swapped despite matching config geometry.

The role-source authority is the exact independently frozen [PR #95](https://github.com/techrote/neurounner/pull/95) source commit `2271218d625e1462b2cf218445da975efe17fe32`. Its pinned README SHA-256 is `09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9`. Only the documented ordinary `<|user|>...<|end|><|assistant|>` framing is meaningful dialogue serialization here. Actual source control IDs `6`, `7`, `10`, `11` are tested solely for token transport; **tool argument, result and memory grammar/behaviour remain unqualified**.

## Bounded no-training execution

[`scripts/ncr_001_postlease_sprocket.py`](../scripts/ncr_001_postlease_sprocket.py) asserts exact checked-out PR head, immutable source pins, false training authorization and original model source byte hashes, then runs offline with original float32 PyTorch HF model weights. The Chat and root Instruct lanes are independent matrix jobs.

For **each** lane, the original-host producer exercises:

- Zero-initialized NCA and MLP no-op logits vs the same frozen original at lengths 16/128 and both left/right masked batch modes; four-unit native `HF.generate(use_cache=True)` parity.
- Cross-wrapper contention in both directions: disabled foreign baseline calls must be refused while NCA or MLP hooks own the original decoder.
- Manual **untrained** nonzero readout intervention, NCA and MLP 512/2048-unit full-prefill versus four-step cached incremental logits under the original strict `0.0005` absolute/relative tolerance.
- Strict future-suffix invariance, A→B→A request reset, and transport of recognized raw tool/memory control IDs as model inputs (not application-level semantics).
- Frozen parameter fingerprint before/after, no base gradients, separate adapter metadata and state files with SHA-256s.

A fresh consumer **Python interpreter** on the same host independently rechecks complete checkpoint/source bytes, model identity, tokenizer input hash and both adapter-file digests; it then demands byte-exact NCA/MLP output reload against the producer's 64-unit source-formatted input. Separate process/PID is checked, not presumed. This is not a cross-host determinism claim.

The seven source-contract adversarial tests reject a changed training flag, collapsed four-model lanes, swapped Chat/Instruct checkpoints, altered source revision, wrong geometry and changed taps. The existing ten cross-decoder lease synthetic tests are rerun. The workflow pins PyTorch `2.14.1+cpu`, Transformers `4.57.6` and huggingface_hub `0.36.2`, checking exact source/blob IDs before original physical execution.

## Evidence and refusal limits

Both original producer and consumer write partial JSON on **every** exit path (including exceptions), with run/attempt/head, runner/PID/platform/package versions, SHA-256s, CPU RSS, per-check outcomes and traceback. Original full logs and the untrained adapter checkpoints are preserved on success/failure through Actions artifacts.

**Actions artifacts are temporary staging, not immutable research evidence.** After completion, exact original receipts, negative outcomes, runner identity and byte hashes must be independently read back and promoted under `experiments/ncr-001/`; no success is asserted before evidence exists. The original frozen model only executes when the workflow actually reaches the corresponding model job.

This is **not final #72 approval**, even after passing physical source/cache and cold replay. The three still-unqualified required learning/update/loss checks (`base_weight_unchanged_after_step`, `nonzero_adapter_gradient_after_warmup`, `finite_state_and_loss`), external independent review and all other lane gates remain separate. No supervised backward, optimizer steps, held-out evaluation or comparative training are initiated.
