# NCR-001F — actual frozen Sprocket matched-MLP source qualification

**Issue:** #73, parent #67; dependency chain protocol PR #77 → core #82 → conventional controls #87 → frozen MLP hooks #96. This draft adds separate actual-host matched-MLP prepasses for **Sprocket Chat and Sprocket Instruct**. It does not supersede independent **NCA** evidence in #95/#97 or certify #72.

## Independent pinned authority

- MLP implementation: PR #96 commit f4f4b5becaebf17d89f11dc8db0952c31cbe451f. Conventional noncellular gated MLP: 658,432 trainable parameters at hidden width 1280 versus 658,904 NCA parameters (472 fewer; 0.0716%). Same three actual post-block taps [7,16,25], no spatial field or depth state.
- Sprocket Chat: kandivault/sprocket-500m revision eb157d4ddddc4619e661d4a2426bb2dd000fc668, **chat/model.safetensors**, exact complete SHA-256 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97.
- Sprocket Instruct: same upstream revision, **root model.safetensors**, exact complete SHA-256 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864. A Chat/Instruct checkpoint swap is an error.
- Independent ordinary-role serializer reviewed in #95, commit 2271218d625e1462b2cf218445da975efe17fe32, read-only checkout. Pinned README SHA-256 09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9; actual tokenizer user/end/assistant control IDs are verified. The source accepts only documented ordinary turns, refusing unsupported system/tool/memory frames.
- Frozen four-model protocol, tokenizers and source inventory are read-only. The train_authorized flag must remain false.

## Execution

Workflow: .github/workflows/ncr-001f-real-sprocket.yml. Two independent ubuntu-24.04 CPU runners with torch 2.14.1+cpu, Transformers 4.57.6, huggingface_hub 0.36.2, safetensors 0.8.0, four CPU threads, float32 eager attention.

1. Download and SHA-256-hash all actual bytes of each original 1,002,207,984-byte checkpoint, model config, README and exact variant tokenizer sources. Preserve prefetch partial/error receipt on failure.
2. Reassert exact #95 role-format source checkout and run HF offline. Serialize non-held-out synthetic user dialogue and check 16/64/128 source-token no-op logits, both directions of masked batch padding and short greedy parity.
3. Execute one actual-model loss/backward, verify finite readout gradient and entirely frozen original backbone. Construct **no** optimizer and take **zero** steps.
4. Manually activate one previously zero readout scalar, not a learned control. Verify nonzero logit effect, 48 true per-token incremental KV-cache versus full-sequence logits under preregistered 5e-4 tolerances, zero future suffix influence, exact request A→B→A reset.
5. Save/reload only the adapter into a newly constructed adapter instance on the **same process/model**; deliberately reject wrong lane, source file, checkpoint SHA and revision. Hash the complete loaded frozen backbone parameter bytes before/after.
6. Preserve complete partial or successful source-bound JSON for each phase in runtime/ncr-001f/sprocket-{variant}-prefetch.json and runtime/ncr-001f/sprocket-{variant}-mlp.json. Actions uploads even on failures; artifacts are **transient** pending lossless independent promotion into committed repository evidence.

Source-only tests reject source/variant/weight/tap/model-architecture and train-authorization substitution. All held-out prompts, fixture hashes, optimizer plans and original source evidence remain untouched.

## What this does not prove

A passed CI job is only a preliminary actual-host **MLP** hook/cache/gradient result. It does not qualify Instruct tool-call/result payload or memory-read/write semantics, full 2048-context behavior, separate-process MLP reload, comparative NCA/MLP learned quality, or aggregate #72 acceptance. PR #97 runs separately on **NCA** and cannot stand in for the MLP result.

No cross-checkpoint scores, unseen held-out examples, optimizer updates, draft merges or training admission are allowed. The protocol retains **train_authorized=false**.

## Follow-up custody

Keep draft. On real CI completion, independently download original JSON and artifact ZIP, verify lengths/hashes/runner/runtime/source identities and negative assertions, then commit exact original raw receipt bytes with provenance. Do not mark actual-host qualification PASS from a preliminary summary or claim success for any failed workflow. Independently advance void.1 matched MLP and all remaining #72/model serialization gates.
