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

## Verified real-model Jobs, complete source receipts and independent byte readback — 27 October 2026

The first complete original source-bound [Actions run 40519177555](https://github.com/techrote/neurounner/actions/runs/40519177555) **SUCCESS**: source/refusal job and both genuinely loaded pinned, distinct frozen Sprocket Chat and Instruct HF models passed. The source branch at launch was commit `5538f32f94459bdb8921c2898a44f45f4f5803ed`; Actions checked out its PR synthetic merge `05fefa7842ea79216a1d591d674a00ed92ba7f8a`. The real frozen backbone was 501,090,560 parameters each (complete 1,002,207,984-byte checkpoint each), MLP 658,432 versus NCA reference 658,904. Both used PyTorch 2.14.1+cpu, Transformers 4.57.6, Python 3.12.3, float32 eager attention, four exposed CPU threads, with independent runner CPU identities preserved.

| Source-bound **untrained** MLP observation | Chat subfolder | Root Instruct |
|---|---:|---:|
| Active 48-token full-vs-true incremental KV-cache max absolute error | **0.00003910064697265625** | **0.00018787384033203125** |
| Fixed protocol cache absolute/relative tolerance | 0.0005 / 0.0005 | 0.0005 / 0.0005 |
| Manual untrained readout max effect on logits | 0.018098831176757812 | 0.03246331214904785 |
| Source-role 16/64/128 zero-init and masked-batch logit drift | **0 exact** | **0 exact** |
| Future suffix drift; request A→B→A reset drift; same-process new adapter reload drift | **0 / 0 / 0 exact** | **0 / 0 / 0 exact** |
| Real-model readout-gradient L1 (one backward, **zero** optimizer steps) | 1849.1005859375 | 3612.0458984375 |
| Complete in-memory original frozen base before/after SHA equal | **yes** | **yes** |
| Host CPU | Intel Xeon Platinum 8370C | AMD EPYC 7763 |
| Peak RSS (KiB; process ru_maxrss) | 3,293,812 | 3,293,464 |

**Original artifact custody:** original [Chat artifact #12740114753](https://github.com/techrote/neurounner/actions/runs/40519177555/artifacts/12740114753) complete ZIP SHA-256 `f05b19116e942701bd045f26952eb5780f6005c458f7e3658ec308b2763a387a` (3,248 bytes); original [Instruct artifact #12739839951](https://github.com/techrote/neurounner/actions/runs/40519177555/artifacts/12739839951) complete ZIP SHA-256 `cf763fc93d9723f9eb08d011051652a1f3bdfdbc927c00dc303010f730e6a182` (3,370 bytes). Each contains its exact original partial/success source-prefetch and actual-model JSON. The four **unaltered uploaded original raw JSON bytes** (total 17,898 bytes) are stored in a deterministic compressed [originals bundle](../experiments/ncr-001/controls-evidence/real-sprocket-mlp/original-jsons.tar.gz.b64), gzip SHA-256 `0760c6daa8ef4d2934069a40a60d8975ed8ed9b3d2cec64af3ca4837872a703e` (3,095 bytes), accompanied by [individual SHA-256/size/source/runner/ZIP provenance](../experiments/ncr-001/controls-evidence/real-sprocket-mlp/provenance.json). No JSON receipt was reconstructed or normalized.

The [separate source-byte readback run 40520745880](https://github.com/techrote/neurounner/actions/runs/40520745880) **SUCCESS**: 10 standard-library tests and independent exact-source [readback script](../scripts/ncr_001f_sprocket_mlp_receipt_audit.py) passed. Mutation sentinels reject internally consistent-looking false success claims, wrong source variant, distorted KV-cache metrics, altered future causality/request isolation, false training admission, lost gradient/frozen-base assertions, changed role-token control IDs, stale Actions identity and raw archive byte tampering. This **checks custody and numeric claims from the original already-executed models**, rather than running a third actual LLM. No original job or original test reported a failure; intermediate superseded pushes that GitHub cancelled are not confused with failed scientific results.

**Admission remains CLOSED:** This proves a source-bound, original frozen Sprocket **MLP** prepass for both distinct variants. The Sprocket independent **NCA** 256-unit fresh-interpreter test in #97 is separate. Neither together nor separately approves tool-result/memory semantics, 2048-token long-context, full per-model #72 acceptance, matched trained quality, or any optimizer update. All four lanes retain `train_authorized=false`; no sealed held-out evaluation, no draft PR merges.
