# NCR-001E: post-lease source-structured Sprocket Instruct wire probe

**Status:** draft candidate; physical CI and external scientific review must be checked independently. This is a bounded **frozen-model serialization/causal transport** study, not model competence or #72 admission.

## Exact dependencies and source custody

- Parent #67; gate #72; matched control #73. Stack **#154 → #155 → #156 → #157 → #158 → #159 → #160 → this draft**, without merging or replacing any historical result.
- Source baseline is original #160 commit `d168fa41ae35536e893a5bc548a7b21965c45999`. The root **Sprocket Instruct** model is `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668` with **root** `model.safetensors` SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`; do not use sibling Chat `chat/model.safetensors` SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`.
- Ordinary-role tokenizer/card formatter source #95 is checked out at `2271218d625e1462b2cf218445da975efe17fe32`, and the **publisher-source-derived** tool/memory serializer from #102 at `3592841802d85c442ca2521479b5d8795bee7233`, blob `4e558ae530a1b79050e22bee587b3caf056e4563`. Publisher SFT source `kandivault-source/Sprocket-500M@a003767e70d766bc14b40c89792543ee8641c8bc`; this **does not cryptographically attest** the training-build origin of the HF weights.
- Four frozen-model protocol/model pins, causal tolerances and sealed evaluation fixtures unchanged. `train_authorized=false`, `full_72_conformance_pass=false`.

## Narrow, new question

Earlier source-bound PR #102/#113 examined publisher structured role/tool/memory serialization using the **older Instruct NCA hook**. Original PR #155–#160 instead examined **post-lease composed NCA + matched MLP wrappers** under ordinary roles and raw control-token transport. The two prove **different** facts and cannot be combined into one source-matched observation without another execution.

The new runner `scripts/ncr_001_postlease_sprocket_wire.py` imports the exact #102 source serializer read-only, verifies root-Instruct checkpoint bytes, constructs **12 synthetic host/assistant turns**, and records all prompt IDs, loss-mask ownership and full source hashes. It checks:

1. Source-authorized system/user/assistant, host tool-result, assistant tool-call, host/assistant memory-read and assistant memory-write positioning; correct **assistant-only SFT loss mask** versus the entirely separate **inference attention mask**.
2. Real in-context `<|end|>` token **5** distinguished from artificial left-padding token **5** via an explicit attention mask; naive `input_ids != 5` is a failing negative sentinel.
3. Original frozen reference vs disabled/zero-readout source logits for both composed NCA and gated-MLP wrappers, then **manually untrained** nonzero interventions with independent left-padded versus single-input logits.
4. Real incremental KV-cache versus full forward at assistant tool-call, host tool-result, assistant memory-write and host memory-read boundaries; original absolute and relative `0.0005` tolerance, without rewriting the protocol.
5. Raw six-unit nontrained greedy continuations recorded for diagnostic interpretation **without semantic scoring**, plus unchanged complete frozen-backbone parameter fingerprint and no backbone gradients.

No optimizer step, NCA-versus-MLP training, held-out fixture read, external tool invocation, stateful memory backend or release-quality assessment is permitted. Source test failures, host failures and partial JSON/logs must be retained without promoting them to passing evidence. Full original frozen weights are SHA-verified before offline loading, with runner identity, checkpoint/source hashes, raw IDs, raw generated text, numerical errors, negative controls and peak RSS in Actions staging.

## Historical negatives and admission boundary

Prior original #102 tool/memory output diagnostics reported **5/7** cases exhausting a 64-unit budget, **0/21** valid model-emitted structured tool/memory actions and injected memory labels appearing in output. These are *original-source diagnostics*, not newly measured results, and must not be relabelled as post-lease findings. Raw control-token transport, source-derived serialization, teacher-forced cache parity and synthetic host tool-result placement do not imply correct tool selection, arguments, execution, trust boundaries, memory retrieval or response correctness.

The new runner explicitly retains `model_generated_tool_memory_semantics_qualified=false` even on technical PASS. Physical CI success alone does not provide independent scientific review, historical publisher-build attestation, validated game/tool integration, SmolLM2 or void.1 admission, or authority for comparative training. BYTE-EOS-001 #92 is a distinct void.1 lane.

## Evidence and next steps

The GitHub Actions workflow `.github/workflows/ncr-001-postlease-sprocket-wire.yml` runs **11 standalone source/refusal tests**, the inherited shared-decoder lease tests and one original root-Instruct frozen-host job at the new exact PR head. The workflow uploads source logs and complete partial/positive original-host receipts with `if: always()`. Until Actions is checked, these are *planned* checks, not completed claims.

After successful execution, independently download original Actions ZIPs, inspect complete logs and raw receipts, preserve their original bytes (including any failures) under versioned Git custody with complete SHA-256/source/runner identities, and require a distinct scientific reviewer. Reviewers should establish whether any expected source mask or cache path fails, reconcile #102 source ownership to #154 shared lease, and explicitly leave tool/memory **semantic** capability unadmitted. Do not edit preregistration, enable `train_authorized`, merge this draft or run comparative NCA/MLP optimization.
