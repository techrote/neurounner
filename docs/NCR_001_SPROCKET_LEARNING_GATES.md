# NCR-001E: Sprocket NCA-only diagnostic learning gates

**Scope:** issue [#72](https://github.com/techrote/neurounner/issues/72), parent [#67](https://github.com/techrote/neurounner/issues/67). This draft is stacked on Sprocket explicit-mask custody PR #158 and the shared decoder-root lease/physical-host sequence #154–#157. It does **not** revise the source protocol, preregistered held-out fixtures, evaluation schedule, checkpoint pins, or frozen model integrations.

## Rationale

The frozen-host parity/cache/cold-reload checks do not establish the three separate learning obligations from #72: a finite supervised loss, post-warmup gradients into the NCA updater and input projection, and frozen backbone integrity after an adapter optimizer update. Zero-initialized NCA readout is an intentional gradient barrier for the first backward; the upstream updater should receive gradient **after** the readout is updated, not before.

This is a **one-step source-bound conformance diagnostic, not a trained model, training protocol admission, or a comparative NCA/MLP experiment**. Only the NCA is stepped once. No MLP is optimized, compared, evaluated or scored. No dataset/training/evaluation/held-out fixture is opened, and no adapter checkpoint is exported. The unchanged four-model protocol retains **train_authorized=false**.

## Actual original frozen hosts

Each matrix job independently downloads and verifies the complete weight file and pinned tokenizer/config/model-card files before the model is loaded offline:

| Lane | Source | Exact original weight file |
|---|---|---|
| Sprocket Chat | kandivault/sprocket-500m @ eb157d4ddddc4619e661d4a2426bb2dd000fc668 | chat/model.safetensors, SHA-256 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97 |
| Sprocket Instruct | same pinned publisher revision; independent weight identity | root model.safetensors, SHA-256 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864 |

Both use pinned ordinary-role formatter commit 2271218d625e1462b2cf218445da975efe17fe32 and only source-documented user/assistant framing. The diagnostic constructs a new, fixed, explicitly synthetic prompt/assistant completion in memory. Assistant-only labels mask the user prefix; the real terminal end-of-turn token **5** remains supervised and is attended. No padding is needed, and no attention mask is reconstructed by token equality (ID 5 could also be synthetic padding).

The actual test runs in float32 eager attention under pinned Transformers 4.57.6 and PyTorch 2.14.1 CPU. The script verifies the complete original loaded backbone before and after the single SGD update using all named parameter bytes, and checks exact disabled-source logits before/after.

## Gates and hard negatives

1. Initial zero-readout adapter logits must exactly equal the same frozen original model's disabled logits.
2. The synthetic assistant-only shifted cross-entropy loss must be finite, positive and differentiable; the first backward must reach readout but **not** upstream zero-initialization-barrier components.
3. Exactly **one NCA-only SGD conformance step** updates the readout; the optimizer does not receive frozen model or MLP parameters. No separate learning loop or subsequent optimizer step is permitted.
4. A second real-model forward/backward must show finite, nonzero gradients for NCA observe, perception, rule and readout weights. It must witness three distinct, ordered, finite field snapshots through taps [7,16,25].
5. Every frozen backbone parameter must remain nontrainable, with no gradient; its full before/after byte fingerprint and exact disabled-model outputs must remain unchanged.
6. Offline sentinels reject absent/zero/nonfinite upstream gradients, hidden trainable/frozen gradients, broken assistant EOS supervision, or fake scalar losses. Source and physical host CI results remain distinct.

The runner writes source/weight hashes, source-head SHA, full runner versions/identity, synthetic sample hashes, gradients, loss, update count, field hashes, frozen model hashes and partial failure tracebacks into an Actions artifact even on failure. **Actions artifacts remain provisional staging until independently read back and byte-for-byte promoted to the repository.** Do not count an uploaded ZIP or green badge as durable independent evidence.

## Admission boundary

An isolated diagnostic optimizer step is necessary for the specified #72 learning contract, **not** authorization to begin the preregistered 192-step comparisons. A physical PASS only advances the corresponding Sprocket lane's learning-gate evidence at the exact source head. It does not by itself certify role/tool/result/memory semantics, independent scientific review, the post-lease SmolLM2 or void.1 host, or final four-model acceptance. Both Sprocket variants remain individually **BLOCKED** for final #72 admission pending independent readback/review and additional protocol sign-off. Keep all related PRs draft and train_authorized=false.
