# NCR-001A — persistent-depth neural cellular adapter preregistration

**State:** PROVISIONAL / NOT TRAINING-ADMITTED. Parent [#67](https://github.com/techrote/neurounner/issues/67); prerequisite [#68](https://github.com/techrote/neurounner/issues/68). Implement shared core only after source checks and protocol review. This page documents the exact current proposal; `protocol.json` is the machine-readable contract.

## Research question

Does an explicitly spatial, iteratively updated 8×8×4 neural field, *carried through transformer depth for each token*, improve a frozen language model more than an ordinary parameter-matched feed-forward residual adapter? Benefits are **hypotheses**, not prior evidence.

The comparison is **within each frozen model**: SmolLM2, Sprocket Chat, Sprocket Instruct, and void.1. Chat versus Instruct has differing fine-tune weights; void.1 has a different decoder family and byte-level tokenizer. None is an across-model causal comparison. A positive score obtained by merely adding extra trainable parameters does not establish locality, recurrence or useful routing.

## Sources, known information and hard uncertainty

| Lane | Required source | Current factual basis | Hard gate |
|---|---|---|---|
| SmolLM2-135M-Instruct | `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac` | TRAIN-001/003 proved CPU train/load/evaluate for this identity; Llama, 30 blocks, width 576, 9 query/3 KV heads, 8192 positions | Verify complete source identity + file hashes in a pinned preflight before training |
| Sprocket-500M **chat** | `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, `chat/model.safetensors` | **Pinned source inventory verified:** trainable `chat/model.safetensors` exists; 26 Llama blocks, hidden width 1280, 20 query/4 KV heads, 2048 context, 32k vocab. Publisher card omits the chat HF checkpoint, but the pinned repository has it. | Exact tokenizer/chat-template semantics and **checkpoint bytes** still require independent verification, followed by #72 model-forward and cache conformance. Do not use root instruct weights as chat. |

**Source inventory result:** [NCR-001A Actions 40401534069](https://github.com/techrote/neurounner/actions/runs/40401534069) confirmed the exact `chat/model.safetensors` file and pinned architecture. This is **metadata/file-presence evidence, not a downloaded weight SHA or numeric model-load qualification**. The chat GGUF remains a distinct behavioural reference; it is not a substitute for HF safetensors. SmolLM2 integration and Sprocket chat-template qualification can now proceed as separable tracks.

### Preflight commands

```bash
python3 scripts/ncr_001_protocol.py validate
python3 -m unittest discover -s tests -p "test_ncr_001_protocol.py" -v
python3 -m pip install "huggingface_hub>=0.36,<2"
python3 scripts/ncr_001_protocol.py preflight --out runtime/ncr-001/source-preflight.json
```

Only the last command contacts Hugging Face. It downloads metadata and small tokenizer/config files, **never weights**, and leaves training forbidden even when it succeeds. CI stores a temporary preflight receipt; promote reviewed source identities/digests into the repository before considering #68 complete.

## Additional model lanes — 2026-10-26 owner amendment

| Lane | Pin and adapter-specific identity | Architecture and taps | Qualification constraint |
|---|---|---|---|
| **Sprocket Instruct** | `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, **root** `model.safetensors` | Llama, 26×1280, **[7,16,25]** (verify fresh source inventory) | Distinct weights from Chat; preserve intended `<\|tool_call\|>` and `<\|memory_write\|>` control-token behavior, baseline negative results and exact instruct formatting |
| **void.1** | `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563` | **Qwen3**, 24×640, **[7,15,23]** (verify fresh source inventory); 259-token vocabulary | Use pinned `tokenization_byte.py` + `byte_vocab.json`; inspect remote code **without executing it in metadata preflight**, then explicitly review before HF loader; byte-aware generation budgets |

The prior [TRAIN-002b eight-lane study](../../docs/TRAIN_002B_VOID_EIGHT_LANE.md) already qualified void.1 CPU LoRA and documented its tokenizer trap; its exact checkpoint revision and source hash should be reused, not replaced by a newer unpinned version. Independent additions are integration issues **#78/#79**, bounded experiments **#80/#81**; #72, #73 and #76 apply to all four, conditional on per-model PASS.

Sprocket Chat remains a first-class **separate** lane. A user preference for Instruct does not retroactively relabel Chat evidence. Shared across lanes: only the NCA algorithm, field geometry and evaluation design, not weights, tokenizers, hidden-layer hooks or absolute-token score expectations.

## Exact NCA design candidate

- Position-specific field `F[B,T,4,8,8]`, 256 float32 variables per token, spatially **independent across T**. The shape is a conceptual layout; internal tensor ordering may differ with tests.
- Three depth taps **after** a transformer block, using **zero-based** completed-block indices: `floor(L/3)-1`, `floor(2L/3)-1`, `L-1`.
- SmolLM2's 30-block taps are **[9, 19, 29]**; both 26-block Sprocket variants use **[7, 16, 25]**; void.1's 24-block Qwen3 architecture uses **[7, 15, 23]**. Initial instruct/void numbers must be confirmed against exact revision in the hosted preflight before being accepted as source-proven.
- At the first tap, initialize the field from the current token's hidden representation and apply three learned local update iterations; later taps inject a new observation but **reuse/evolve the same field**, not overwrite it with a new field.
- A single learnable local rule scans 3×3 neighbours with **zero padding**, bounded/gated residual updates and weights shared across all sites and taps; there is no toroidal boundary by default.
- A learned field-to-hidden output projection is initialized to **exactly zero**, and is injected as an additive residual. Initial logits must therefore match a frozen, same-weights unmodified reference. After training begins, gradient-flow tests must account for the zero-init readout initially suppressing upstream NCA gradients until it moves.
- The base transformer, embedding and output head must remain frozen; no generic library code overrides or model source modifications.

The field is **persistent across depth, not time**. No cellular state may survive from token A to token B, from one forward call to another, or from one user's request to another. KV caches contain only normal causal transformer state; no hidden NCA cross-token memory. This distinction is essential: implementing persistent token-time memory would change the experimental hypothesis and must get its own issue.

**Routing:** The first stage is *dense modulation* of the hidden pathway, **not** conditional expert skipping. Reductions in computation cannot be claimed without an actually conditional execution path.

## Conformance before comparative training

For each backbone separately, #72 must demonstrate:

1. Exact source/version/variant and tokenizer/template identity; no surprise fallback.
2. Zero-init/disabled adapter vs same pinned HF model logits: `atol=rtol=1e-4` under the same precision and CPU backend.
3. Causality: alter a future suffix and prove previous-token outputs unchanged; compare full forward vs repeated prefixes and genuine incrementally cached decode. Initial max difference bounds `atol=rtol=5e-4`, adjusted only if measured and explicitly documented *before* training.
4. Per-token and per-request state isolation; no tensor/module-global field shared between forward calls or independent batch sequences; independent reset.
5. Frozen base weights unchanged after optimizer step and trainable NCA gradients demonstrably nonzero after the readout starts learning.
6. Deterministic state/checkpoint reload, finite activations/losses, bounded RSS and no unintended base gradient.

Training is prohibited for any model lacking its **own** conformance PASS. Passing conformance for any model must never waive failure on another; record four independently gated results. Sprocket Chat and Instruct have different weights even when their layer geometry matches. Void.1 must use its own Qwen3 hooks rather than Llama hooks.

## Control lanes and evaluation

Keep the **same model/weights, data split, sample order, optimizer policy, update count, and selected hidden tap locations** for all variants within a model.

1. Frozen no-adapter base.
2. Trainable parameter-matched gated/MLP residual adapter (≤5% parameter-count difference, or explicitly report a matching failure).
3. Depth-persistent NCA, three updates/tap (primary).
4. Persistent NCA, one update/tap.
5. Three-update NCA **reset** at each tap.
6. Three-update NCA with a fixed, seed-recorded neighbourhood permutation to distinguish local geometry from generic capacity.

The authoritative TRAIN-003 corpus contains **768 train / 128 family-disjoint held-out rows** and uses a fixed mechanical evaluator. Reuse it, but do not retune on the 128 held-out examples or use those examples to make Sprocket more instruction-following. Pre-register a separate original/appropriately licensed natural-language next-token holdout and immutable Sprocket behaviour probes; both **still require final hashes and approval**. Scores may not be compared across different tokenizers without careful normalization. **void.1** requires its custom pinned `tokenization_byte.py`/ByteTokenizer and `byte_vocab.json`; record tokenizer units, UTF-8 bytes, characters, budget exhaustion, EOS behavior, and bytes/sec separately. Earlier TRAIN-002b (issue #1) showed 768-unit generation exhaustion despite valid first JSON, so the proposed 256-unit training cap remains subject to actual-length preflight rather than truncating examples.

Record actual sequence and supervised tokens, optimizer steps, wall seconds, CPU model/thread count, peak RSS, checkpoint bytes, parameter counts, loss, forward/decode speed, raw outputs and scorer version. Baseline gains can be misleading when host hardware differs, evaluation selects only a single favourable seed, or the baseline adapter has significantly fewer trainable parameters.

## Frozen optimizer and cellular update values

Before any actual model training, the shared architecture and trainer defaults have been recorded in machine-readable `protocol.json` and enforced by negative mutation tests:

- Local state 8×8×4; hidden projection H→256 with affine LayerNorm and tanh; first-tap seed; at later depth taps, add 0.2×new observation to the previous field, apply tanh, then three recurrent update iterations.
- Every recurrent iteration: 3×3 zero-padded Conv2d 4→16, SiLU, pointwise Conv2d 16→8 split delta/gate, followed by **tanh(field + 0.25×sigmoid(gate)×tanh(delta))**. A zero-initialized 256→H bias-free residual readout keeps initial model logits unchanged.
- Shared NCA trainable parameter count is **514×H+984**. Its noncellular control gets a 256-unit bottleneck, LayerNorm, nonlinearity, gating and zero-initialized readout, with exact parameter count and compute budget verified in #73.
- **AdamW** adapter-only; learning rate **8e-4**; betas **(0.9,0.95)**; weight decay **0.01**; global gradient clip norm **1.0**; baseline and NCA receive identical in-model examples/order and update caps.
- Training per-example tentative ceilings: **256 ordinary tokenizer units** for SmolLM2 and both Sprocket variants, **1024 void.1 byte-tokenizer units**. Sample lengths must be preflight-checked; silently truncating ground-truth completions is prohibited.

These are **fixed candidate hyperparameters**, not evidence that they perform well. The experiment still has `train_authorized=false`: source-lane template/conformance, evaluation fixture hashes and the independent PR review remain separate gates.

### Original evaluation probes under review

- [Sprocket Chat](fixtures/sprocket-chat-probes.json) — 16 previously staged workshop/persona/coherence prompts.
- [Sprocket Instruct](fixtures/sprocket-instruct-probes.json) — 16 original role, tool, memory and repetition prompts. An authorised instruct formatter/tool harness must be qualified in #78; no hardcoded unreviewed special tokens.
- [void.1](fixtures/void-byte-probes.json) — 16 byte/Unicode/JSON/EOS/exhaustion probes, using pinned byte-tokenizer semantics only after #79 source review.
- A **disjoint natural-text holdout** with licensed/public-domain provenance and an original common behavioural regression fixture remain **unspecified** and deliberately block training. Predeclare their byte hashes without choosing data from held-out model responses.

All prompt fixture hashes must be read from the committed bytes, not re-created from a model generation. CI prints exact SHA-256 to support a later reviewed promotion.

## Provisional ceilings and decision policy

- Initial CPU smoke: **≤32 optimizer steps**, no held-out tuning.
- Next controlled primary lane: **192 steps/seed**, maximum **256 sequence tokens/example**, seeds **20261026, 20261027, 20261028**.
- Maximum peak RSS: **12 GiB**; provisional per-lane hard wall cap: **60 min SmolLM2/void.1**, **90 min each Sprocket variant**. These are *ceilings*, not permission to spend them before preflight.
- Tentative continuation rule: overall mean mechanical-score improvement of ≥0.03 absolute vs the matched MLP, nonnegative differences in ≥2/3 seeds, no natural-text CE deterioration >0.01 nats/token, and decode overhead ≤1.5× on reasonably matched hosts. Report effect uncertainty and sensitivity; this does not establish scientific significance with three seeds.
- **GO** means worth a larger controlled study, not production readiness. **NO-GO** means failed causal/source/numerical or cost gates or convincing nonbenefit. **INCONCLUSIVE** includes unverified source, missing heldout proof, or uncertain effects.

The listed optimizer LR/betas/weight decay and natural-language corpus are **not frozen**; the machine-readable protocol therefore **denies training**. The missing pieces must be finalized and reviewed without consulting the final held-out outputs. This is a deliberately transparent incomplete kickoff, not a claim that experimental preregistration is finished.

## Evidence, integration boundaries and next actions

- Reuse [TRAIN-001](../../docs/TRAIN_001_CPU_LORA.md), [TRAIN-003](../../docs/TRAIN_003_SEMANTIC_COMPILER.md), [TRAIN-004](../../docs/TRAIN_004_FULL_PARAMETER.md) and [RAG.md](../../RAG.md).
- Track experiment code, exact source hashes and preflight receipts under `experiments/ncr-001/`; durable datasets (if produced) use [filing policy](../../docs/DATASET_FILING_SYSTEM.md).
- Small metadata can live in Git; actual checkpoints and model binary blobs must use [model storage policy](../../docs/MODEL_STORAGE.md).
- Never edit fixed Village/Persona competitions or their immutable evidence; no Cybersand engine runtime integration, MoE execution, multi-token NCA recurrence or CUDA campaign in this epic.

**Next:** read back fresh exact-revision source metadata for Sprocket Instruct and void.1, verify the custom byte-tokenizer source identity and Sprocket Instruct control-token serialization, freeze missing fixture SHA-256s and optimizer controls, approve #68, then start synthetic #69 core tests. Full checkpoint bytes and cache/forward semantics are reserved for #70–#72 and #78–#79. No comparative training yet.
