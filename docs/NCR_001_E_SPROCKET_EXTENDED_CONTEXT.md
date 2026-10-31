# NCR-001E — pinned Sprocket extended-context frozen-host prepass

**Parent:** [NCR-001 #67](https://github.com/techrote/neurounner/issues/67), [conformance gate #72](https://github.com/techrote/neurounner/issues/72). **This review branch stacks on [draft #97](https://github.com/techrote/neurounner/pull/97) → [#95](https://github.com/techrote/neurounner/pull/95) → [#88](https://github.com/techrote/neurounner/pull/88) → [#77](https://github.com/techrote/neurounner/pull/77)**. It changes neither model-specific hook implementation nor the protocol or sealed fixtures.

## Narrow scientific question

Do the **actual frozen** Sprocket Chat subfolder and original root Instruct model, under the already qualified ordinary user/assistant source-token framing, preserve zero-residual parity, per-token depth-field causal/cache correctness and stable untrained NCA activation at **512, 1024 and the original 2048-context limit**? Do their four *recognized but semantically unqualified* tool/memory marker IDs survive a raw-ID full-vs-KV-cache transport test?

Previous actual independent Chat/Instruct [#95](https://github.com/techrote/neurounner/pull/95) checks reached 128 units with short 48-unit caching. [#97](https://github.com/techrote/neurounner/pull/97) independently passed a **256-unit**, 240-prefill+16-incremental, same-runner separate Python process restore. This branch adds a **fresh actual-model context probe**, not a reinterpretation of those archived receipts and **not** a new cross-process test.

## Source and runner contract

- Chat: `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, **chat/model.safetensors**, entire 1,002,207,984-byte source SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`, original NCA hook `83a56835a7476f17a17980d28dd1a7c14bb6f2f6`.
- Instruct: same source revision, **root model.safetensors**, entire 1,002,207,984-byte source SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`, separate original NCA hook `b95c010f9b2ba3cbd9f81317981316bb9c8acd19`.
- **Four distinct protocol keys** remain `smollm2`, `sprocket_chat`, `sprocket_instruct`, `void1`; `train_authorized=false` is checked before source prefetch and again on actual host.
- Ubuntu 24.04, 4 CPU threads, original Transformers 4.57.6, torch 2.14.1+cpu, huggingface_hub 0.36.2. Unchanged source model weights, immutable hooks and tokenizer/config/card exact hash checks; downloads precede an entirely offline execution.
- Every variant has its own CI matrix job with a **55-minute hard limit**. CPU/RSS, original pinned source SHA, context lengths, cache metric, runner identity, any traceback and all partial results are recorded as JSON and uploaded on *both* success and failure. The Actions artifact is staging, **not** Git archival; original JSON bytes, ZIP SHA and full failure history must be promoted independently before durable qualification.

## Probe procedure (not training)

1. Only use an artificially repeated, new engineering sentence in the publisher-documented ordinary role format, not TRAIN-003 or any held-out fixture. Predeclared test context sizes **512, 1024, 2048** are hard-coded; unexpected source context limits fail closed.
2. On **512 and 2048** units, check **exact** source-disabled vs zero-NCA adapter logits for every output position. A single NCA readout coefficient is **manually** set to 0.03 (untrained).
3. For **512, 1024, 2048** units separately, require a nonzero activated adapter effect, finite logits and fixed-protocol **5e-4 absolute / 5e-4 relative** agreement of full forward against one cached prefix forward plus **eight genuine 1-token cached forward calls**. A manually reactivated NCA is not a learned adapter.
4. At 512 units, mutate the future final token and require bitwise identical earlier logits; then perform independent A→B→A requests and require exact first/last A logits.
5. Require the *actual dedicated tokenizer IDs*, not ordinary textual substrings, for `<|tool_call|>`=6, `<|tool_result|>`=7, `<|memory_read|>`=10 and `<|memory_write|>`=11. Insert these four **raw token IDs** into a synthetic 64-unit sequence and verify full-vs-genuinely cached 60+4 logit transport. **This is deliberately NOT a well-formed tool call, tool result, memory read/write, or behavioural regression test.** Instruct's intended argument/recipient/continuation grammar has not been established by source, and Chat was not trained for these pathways.
6. Assert original weights remain frozen, no gradient accumulation, **zero optimizer steps**, no corpus evaluation, no training, no false release/conformance flag.

## Interpretation and blockers

A green run only establishes **source-bound real-host extended-context/cold-request feasibility for these two specific Sprocket checkpoints**. It does not prove arbitrary 2048-turn tool/memory continuation, complete left/right padding at full context, a per-token 2048-prefix scan (only the last 8 positions are independently decoded at each chosen length), a new-machine restart, or independent #72 final acceptance. The artificial control-token transport test says **nothing about semantic tool/memory quality or serialization grammar**.

Pending acceptance work: retain original Actions artifact ZIP digests/JSON SHA in Git after the job completes; independently audit/cache replay evidence; find and source-bind **actual** Instruct tool argument/result and memory serialization before qualification; audit #68/#69 and all model-specific final #72 checklists. SmolLM2 and void.1 are separate acceptance lanes; #73 gated MLP controls and BYTE-EOS-001 #92 remain separate dependencies. **Never merge draft PRs, train NCA/MLP, alter held-out fixtures, or change `train_authorized=false` without independent review and formal protocol admission.**
