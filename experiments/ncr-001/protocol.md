# NCR-001A — persistent-depth neural cellular adapter preregistration

**State:** PROVISIONAL / NOT TRAINING-ADMITTED. Parent [#67](https://github.com/techrote/neurounner/issues/67); prerequisite [#68](https://github.com/techrote/neurounner/issues/68). Implement shared core only after source checks and protocol review. This page documents the exact current proposal; \`protocol.json\` is the machine-readable contract.

## Research question

Does an explicitly spatial, iteratively updated 8×8×4 neural field, *carried through transformer depth for each token*, improve a frozen language model more than an ordinary parameter-matched feed-forward residual adapter? Benefits are **hypotheses**, not prior evidence.

The comparison is **within each frozen model**, not SmolLM2-versus-Sprocket superiority. A positive score obtained by merely adding extra trainable parameters does not establish locality, recurrence or useful routing.

## Sources, known information and hard uncertainty

| Lane | Required source | Current factual basis | Hard gate |
|---|---|---|---|
| SmolLM2-135M-Instruct | \`HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac\` | TRAIN-001/003 proved CPU train/load/evaluate for this identity; Llama, 30 blocks, width 576, 9 query/3 KV heads, 8192 positions | Verify complete source identity + file hashes in a pinned preflight before training |
| Sprocket-500M **chat** | \`kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668\`, proposed \`chat/model.safetensors\` | Publisher documents chat F16 and Q4 GGUF, 2048 context, 20 query/4 KV heads, 32k vocabulary. Only the **instruct** build is explicitly advertised as root HF safetensors in the model card. Whether \`chat/model.safetensors\` exists is **unverified**. | Check exact pinned repo tree, chat config/tokenizer and trainable checkpoint. If absent, mark **BLOCKED** and request separately authorised chat-HF export. Never use root instruct weights silently. |

The SHA above for Sprocket is an existing neurounner model reference, **not** proof that a trainable chat checkpoint exists at that revision. The reference GGUF is a useful behavioural benchmark, but its bits are not a differentiable safetensors checkpoint. If chat HF cannot be verified, work on the SmolLM2 lane may proceed independently after its own gates; Sprocket comparisons stay blocked.

### Preflight commands

\`\`\`bash
python3 scripts/ncr_001_protocol.py validate
python3 -m unittest discover -s tests -p "test_ncr_001_protocol.py" -v
python3 -m pip install "huggingface_hub>=0.36,<2"
python3 scripts/ncr_001_protocol.py preflight --out runtime/ncr-001/source-preflight.json
\`\`\`

Only the last command contacts Hugging Face. It downloads metadata and small tokenizer/config files, **never weights**, and leaves training forbidden even when it succeeds. CI stores a temporary preflight receipt; promote reviewed source identities/digests into the repository before considering #68 complete.

## Exact NCA design candidate

- Position-specific field \`F[B,T,4,8,8]\`, 256 float32 variables per token, spatially **independent across T**. The shape is a conceptual layout; internal tensor ordering may differ with tests.
- Three depth taps **after** a transformer block, using **zero-based** completed-block indices: \`floor(L/3)-1\`, \`floor(2L/3)-1\`, \`L-1\`.
- SmolLM2's 30-block taps therefore are **[9, 19, 29]**; Sprocket tap indices must be frozen from its actual pinned config, not guessed from parameter count.
- At the first tap, initialize the field from the current token's hidden representation and apply three learned local update iterations; later taps inject a new observation but **reuse/evolve the same field**, not overwrite it with a new field.
- A single learnable local rule scans 3×3 neighbours with **zero padding**, bounded/gated residual updates and weights shared across all sites and taps; there is no toroidal boundary by default.
- A learned field-to-hidden output projection is initialized to **exactly zero**, and is injected as an additive residual. Initial logits must therefore match a frozen, same-weights unmodified reference. After training begins, gradient-flow tests must account for the zero-init readout initially suppressing upstream NCA gradients until it moves.
- The base transformer, embedding and output head must remain frozen; no generic library code overrides or model source modifications.

The field is **persistent across depth, not time**. No cellular state may survive from token A to token B, from one forward call to another, or from one user's request to another. KV caches contain only normal causal transformer state; no hidden NCA cross-token memory. This distinction is essential: implementing persistent token-time memory would change the experimental hypothesis and must get its own issue.

**Routing:** The first stage is *dense modulation* of the hidden pathway, **not** conditional expert skipping. Reductions in computation cannot be claimed without an actually conditional execution path.

## Conformance before comparative training

For each backbone separately, #72 must demonstrate:

1. Exact source/version/variant and tokenizer/template identity; no surprise fallback.
2. Zero-init/disabled adapter vs same pinned HF model logits: \`atol=rtol=1e-4\` under the same precision and CPU backend.
3. Causality: alter a future suffix and prove previous-token outputs unchanged; compare full forward vs repeated prefixes and genuine incrementally cached decode. Initial max difference bounds \`atol=rtol=5e-4\`, adjusted only if measured and explicitly documented *before* training.
4. Per-token and per-request state isolation; no tensor/module-global field shared between forward calls or independent batch sequences; independent reset.
5. Frozen base weights unchanged after optimizer step and trainable NCA gradients demonstrably nonzero after the readout starts learning.
6. Deterministic state/checkpoint reload, finite activations/losses, bounded RSS and no unintended base gradient.

Training is prohibited for any model lacking its **own** conformance PASS. A passing SmolLM2 test must never waive a failed Sprocket test.

## Control lanes and evaluation

Keep the **same model/weights, data split, sample order, optimizer policy, update count, and selected hidden tap locations** for all variants within a model.

1. Frozen no-adapter base.
2. Trainable parameter-matched gated/MLP residual adapter (≤5% parameter-count difference, or explicitly report a matching failure).
3. Depth-persistent NCA, three updates/tap (primary).
4. Persistent NCA, one update/tap.
5. Three-update NCA **reset** at each tap.
6. Three-update NCA with a fixed, seed-recorded neighbourhood permutation to distinguish local geometry from generic capacity.

The authoritative TRAIN-003 corpus contains **768 train / 128 family-disjoint held-out rows** and uses a fixed mechanical evaluator. Reuse it, but do not retune on the 128 held-out examples or use those examples to make Sprocket more instruction-following. Pre-register a separate original/appropriately licensed natural-language next-token holdout and immutable Sprocket behaviour probes; both **still require final hashes and approval**. Scores may not be compared across different tokenizers without careful normalization.

Record actual sequence and supervised tokens, optimizer steps, wall seconds, CPU model/thread count, peak RSS, checkpoint bytes, parameter counts, loss, forward/decode speed, raw outputs and scorer version. Baseline gains can be misleading when host hardware differs, evaluation selects only a single favourable seed, or the baseline adapter has significantly fewer trainable parameters.

## Provisional ceilings and decision policy

- Initial CPU smoke: **≤32 optimizer steps**, no held-out tuning.
- Next controlled primary lane: **192 steps/seed**, maximum **256 sequence tokens/example**, seeds **20261026, 20261027, 20261028**.
- Maximum peak RSS: **12 GiB**; per-lane hard wall cap: **60 min SmolLM2**, **90 min Sprocket**. These are *ceilings*, not permission to spend them before preflight.
- Tentative continuation rule: overall mean mechanical-score improvement of ≥0.03 absolute vs the matched MLP, nonnegative differences in ≥2/3 seeds, no natural-text CE deterioration >0.01 nats/token, and decode overhead ≤1.5× on reasonably matched hosts. Report effect uncertainty and sensitivity; this does not establish scientific significance with three seeds.
- **GO** means worth a larger controlled study, not production readiness. **NO-GO** means failed causal/source/numerical or cost gates or convincing nonbenefit. **INCONCLUSIVE** includes unverified source, missing heldout proof, or uncertain effects.

The listed optimizer LR/betas/weight decay and natural-language corpus are **not frozen**; the machine-readable protocol therefore **denies training**. The missing pieces must be finalized and reviewed without consulting the final held-out outputs. This is a deliberately transparent incomplete kickoff, not a claim that experimental preregistration is finished.

## Evidence, integration boundaries and next actions

- Reuse [TRAIN-001](../../docs/TRAIN_001_CPU_LORA.md), [TRAIN-003](../../docs/TRAIN_003_SEMANTIC_COMPILER.md), [TRAIN-004](../../docs/TRAIN_004_FULL_PARAMETER.md) and [RAG.md](../../RAG.md).
- Track experiment code, exact source hashes and preflight receipts under \`experiments/ncr-001/\`; durable datasets (if produced) use [filing policy](../../docs/DATASET_FILING_SYSTEM.md).
- Small metadata can live in Git; actual checkpoints and model binary blobs must use [model storage policy](../../docs/MODEL_STORAGE.md).
- Never edit fixed Village/Persona competitions or their immutable evidence; no Cybersand engine runtime integration, MoE execution, multi-token NCA recurrence or CUDA campaign in this epic.

**Next:** review the read-only pinned HF preflight, resolve the Sprocket chat checkpoint question, freeze Sprocket tap list and fixture SHA-256s, lock optimizer controls, independently approve #68, then start synthetic #69 core tests. No expensive training yet.
