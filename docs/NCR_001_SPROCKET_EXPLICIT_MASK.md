# NCR-001E/F — Sprocket role-delimiter / padding attention-mask qualification

**Scope:** #67/#72/#73, separate frozen Sprocket Chat and root Instruct checkpoints. Stacked on original evidence-custody draft [#156](https://github.com/techrote/neurounner/pull/156), itself dependent on #155 and #154. No source history or raw physical evidence from those PRs is changed. This is a **new post-lease source-matched qualification witness**, not model training.

## Concrete unresolved ambiguity

The exact source-authoritative ordinary Sprocket role format is:

    <|user|>your message<|end|><|assistant|>

Both Chat and Instruct tokenizers have actual <|end|> marker ID **5** in pinned #95 source; existing #155 host config also assigns the tokenizer pad ID **5**. Consequently **content delimiter positions and synthetic padding positions can carry the same token ID**, but MUST have different attention-mask values: real <|end|> is visible (1), artificial left padding is invisible (0). Reconstructing a mask by comparing input IDs against the pad ID is wrong. Native HF.generate without an explicit attention mask had also emitted an ambiguity warning in the original archived #155 job logs; this is an unresolved observation, not proof of semantic corruption.

Historical #155 receipts test unmasked single-row four-token zero-init generation and separate masked-batch forward parity. They do **not** establish that active NCA and matched-MLP wrappers behave consistently when generating from ordinary-role **left-padded batches with explicit end markers and correct masks**, under cached and uncached decoding.

## New test, fixed before result inspection

The script [ncr_001_sprocket_explicit_mask.py](../scripts/ncr_001_sprocket_explicit_mask.py) loads the existing post-lease #155 NCA and MLP implementations with the #154 shared decoder lease, frozen original float32 host, and separate source-pinned Chat and Instruct weights. It checks the exact immutable protocol and existing publisher-role formatter/source revision (independent #95 exact commit 2271218d625e1462b2cf218445da975efe17fe32), not an invented tool/memory grammar.

For each variant, it constructs a one-turn and an alternating three-turn source-formatted prompt with different nonzero left-padding lengths. It compares padded token IDs against independently tokenized original strings and requires mask 1 on every real token including the ID-5 end markers, and mask 0 on synthetic ID-5 padding. An intentionally wrong ID-derived mask must differ.

It generates exactly three tokens for each of **NCA and matched gated MLP**, in three fixed no-training modes (disabled, enabled zero readout, manually perturbed untrained readout), and both cache modes. Each batch row must equal independent generation of its unpadded prompt with its correct mask. The cache modes and disabled/zero readout outputs must agree. Any discrepancy is **FAIL** and must be retained; protocol gates may not be changed to pass.

Before/after backbone fingerprints, gradient checks, exact source/checkpoint identity, runner/PID/package/CPU/RSS metadata, mask/token tensor digests and partial failure receipts are recorded. The [source-only negative tests](../tests/test_ncr_001_sprocket_explicit_mask.py) reject incorrectly masked genuine end markers, missing end markers, fake equal-length batches, lost explicit masks and training authorization.

The [exact-head Actions workflow](../.github/workflows/ncr-001-sprocket-explicit-mask.yml) checks the immutable source and publisher formatter, downloads and hashes each complete original frozen checkpoint, executes with the model network disabled, and uploads logs and JSON receipts even on failure. It is bound to this draft stack head. Historical #155 producer/consumer evidence remains unchanged.

## Frozen identities and remaining gates

| Lane | Actual original checkpoint SHA-256 |
| --- | --- |
| Chat subfolder model.safetensors | 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97 |
| Instruct root model.safetensors | 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864 |

Both remain pinned to kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668 with [7,16,25] taps.

Source-only CI passing would **not** establish physical model conformance. Successful physical checks would qualify explicit ordinary-role attention masking and generation consistency only, **not** tool-result/memory grammar, supervised finite loss, post-warmup gradients, frozen backbone after optimizer step or full #72 admission. Independent scientific review and artifact readback/promotion remain required.

**No comparative training, held-out probing, optimizer steps or PR merges. train_authorized=false remains mandatory.** Void.1 #92 EOS source/serializer qualification is a separate gate.
