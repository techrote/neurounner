# NCR-001E — Sprocket real frozen adapter reload in a second Python process

**Parent:** [NCR-001 #67](https://github.com/techrote/neurounner/issues/67); [four-model conformance #72](https://github.com/techrote/neurounner/issues/72). **Dependency:** source-role review / frozen-host prepass [draft PR #95](https://github.com/techrote/neurounner/pull/95), based on archived-source [#88](https://github.com/techrote/neurounner/pull/88) → protocol [#77](https://github.com/techrote/neurounner/pull/77). This branch does not change the original model-specific hooks, preregistered fixtures, protocol, optimizer, or any model weights.

## Purpose and scope

The earlier #95 cold-adapter check constructed a *new adapter in the same interpreter against the same in-memory frozen model*. It did not prove a fresh runtime could independently reconstruct the original frozen model and restore the NCA. This bounded extension deliberately separates two **independent Python OS processes**, each loading the exact original full checkpoint and original pinned Chat/Instruct decoder hook implementation. It uses a **single Actions runner and local HF download cache**, so this is *not* a fresh machine, separate environment, container reboot, or final independent #72 acceptance.

There are two distinct sources:

| Field | Chat | Instruct |
|---|---|---|
| Hugging Face repo/revision | kandivault/sprocket-500m @ eb157d4ddddc4619e661d4a2426bb2dd000fc668 | same pinned upstream repository/revision |
| Required safetensors file | chat/model.safetensors | root model.safetensors |
| Actual expected complete weight SHA-256 | 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97 | 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864 |
| Original, immutable hook implementation | 83a56835a7476f17a17980d28dd1a7c14bb6f2f6 | b95c010f9b2ba3cbd9f81317981316bb9c8acd19 |
| Known ordinary message framing | source README user → end → assistant | same ordinary source framing; tool/memory grammar **not** approved |

## Actual process-bound witness

[Script](../scripts/ncr_001e_sprocket_cross_process.py) and [isolated workflow](../.github/workflows/ncr-001e-sprocket-cross-process.yml) do the following on each of the two genuine 500M HF checkpoints.

1. First re-download/re-hash every required source file, including 1,002,207,984 actual model weight bytes, on the hosted runner. Require the four-model protocol train_authorized=false, exact pinned sources, and original hook implementations.
2. **Producer process:** use the documented ordinary user/assistant role framing with no synthetic tool-call grammar, tokenize a fixed *new synthetic engineering sentence* (not any locked held-out fixture), and take 256 tokenizer units. Manually set precisely one *untrained* NCA readout scalar to 0.03, never call an optimizer or loss function. Prove this residual has an effect; compare full 256-position logits against a **240-token cached prefill + 16 actual incremental one-token forwards**, using the preregistered absolute/relative tolerance 0.0005. Check strict unchanged 255 earlier positions after mutating the final suffix, and A→B→A request reset.
3. Save the original NCA adapter through the **existing original implementation's save_adapter**, plus metadata, selected unmodified float32 reference logits and an exact identity/handoff manifest. The reference uses safetensors; the original adapter's own torch.save format is read only by the original load_adapter in weights-only mode.
4. **Consumer process:** launch a new Python interpreter, hash/reload the original full checkpoint/source files and construct a new frozen model/NCA instance, verify manifest and all handoff file SHA-256 digests, then read the exact original adapter with source identity checks. Replay the same active/disabled logit tail and separate A→B→A requests, requiring **exact float32 tensor equality** with the producer's persisted reference. Refuse matching-PID, changed variant, changed original hook commit, source/protocol drift, false training/semantic claims or corrupted adapter/metadata/reference bytes.
5. Always upload the complete available raw producer/consumer JSON, original adapter handoff, manifest and float32 reference logit safetensors, **including partial records on failed runs**. Capture runner identity, actual process IDs, package versions, checked source file hashes, memory high-water marks, actual numerical results, and explicit zero-training flags. The Actions artifact remains *temporary staging*. Independent byte-exact readback and preservation of the successful or failed **original source bytes** in the Git repository are required before claiming permanent evidence custody.

[Offline guard tests](../tests/test_ncr_001e_sprocket_cross_process.py) independently mutate source checkpoint identity, original implementation and protocol hashes, tool/memory/training claims, handoff fingerprints and cold-process identity. They do not execute or evaluate an LLM.

## Correct interpretation / admission barriers

- **Never** present token recognition as a validated Instruct tool-argument/result grammar or memory continuation protocol. Those semantics remain unsupported. Chat's missing tool/memory supervised examples must not be implied from Instruct's model card.
- 256 ordinary-role tokenizer units are more than the earlier 128-token zero-init prefix but below both models' 2048 maximum. This is a CPU cold-process/cache probe, **not** a complete long-session or all-context conformance proof.
- This is only a *manual, untrained* residual. There is no model-quality comparison, no gradients/optimizer steps, no training data, no sealed held-out access, and no learned adaptation.
- Any failed cold-source hash, manifest guard, float32 replay exactness, 256-unit causal/cache tolerance, resource bound or process separation is a **failure** to preserve, not a reason to relax a fixed gate.
- Neither Sprocket lane has a final independently admitted **#72 PASS** from this alone. SmolLM2 and void.1 have separate blocked acceptance dependencies. **train_authorized=false must remain unchanged.** All upstream and this draft PR remain unmerged until independently reviewed.

## Verified actual frozen source-run and original evidence custody — 27 October 2026

[Original Actions run 40515083104](https://github.com/techrote/neurounner/actions/runs/40515083104) **SUCCESS in both actual 500M model jobs**, bound to PR #97 original head a49a33632aa6f4be9df94b07e00e0e1ea58ee10f (GitHub Actions PR synthetic merge e6277a729613de21c260cd4f856d1ebc4578607a). This run used the exact original Chat and root Instruct hook revisions given above, genuine complete 1,002,207,984-byte original model files, reviewed 8,718-byte pinned README, frozen 501,090,560-parameter original model, 658,904 untrained NCA parameters, and recorded 4-CPU/Python 3.12.3/torch 2.14.1+cpu/Transformers 4.57.6 runner provenance.

| Actual source-bound 256-unit witness | Chat subfolder | Root Instruct |
|---|---:|---:|
| **240-prefill + 16 genuinely cached incremental positions** max absolute logit deviation | **3.814697265625e-05** | **4.4345855712890625e-05** |
| Fixed absolute/relative gate | 0.0005 / 0.0005 | 0.0005 / 0.0005 |
| Manually activated, **untrained** NCA effect on logits | 0.0184458494 | 0.0334434509 |
| Changed-future suffix drift; A→B→A reset drift | **0; 0** | **0; 0** |
| Fresh-process restored active/disabled logits and A→B→A replay max deviation | **0 exact** | **0 exact** |
| Producer / consumer Python OS PIDs | **2474 / 2489** | **2722 / 2738** |
| Original runner | GitHub Actions 1000045113 | GitHub Actions 1000045114 |
| Original full artifact ZIP ID | 12739296072 | 12740195021 |
| Exact original artifact ZIP SHA-256 | 6598b26dee5e774353a092e28265ee3cfd08f05673e3a38c17012da36caa5fb4 | 5677cdf4e8974f83d5c0c68188311659b04bf282c1dcda08c6f597e6185d2084 |

Six standalone manifest/source-negative checks also passed in each actual model CI job. All original numerical reports explicitly retain zero optimizer/training/held-out/evaluator claims. **No failure occurred in the original run**; future failed or partial stage receipts must be retained rather than overwritten.

### Durable original receipt archive and independent readback

The actual original GitHub Actions artifacts were downloaded and their complete ZIP byte SHA-256s computed. The **eight original JSONs** (each variant's producer, consumer, manifest, adapter metadata), *unaltered byte-for-byte*, are archived in [the source-JSON gzip/base64 bundle](../experiments/ncr-001/evidence-review/sprocket-cross-process/original-jsons.tar.gz.b64), with [individual byte lengths/SHA-256, source/runner identity, original ZIP digests and tensor-handoff digests](../experiments/ncr-001/evidence-review/sprocket-cross-process/provenance.json). The original compressed gzip member bytes have SHA-256 **73544c842468d9bf5c3eae8d8ba92a253716990be514e3bc969e834d8595f3bc** (2,919 bytes), and the eight JSON files are individually verified. The original full binary adapter and float32 reference tensors are retained in their Actions ZIP artifacts, not falsely described as copied into Git.

[Independent standard-library readback](../scripts/ncr_001e_sprocket_process_receipt_audit.py) validates compressed and individual original JSON hashes, two original pinned frozen checkpoint identities, protocol/tap/runner/original process identities, fixed numerical limits, exact cold restored output and absence of false training/tool/memory/final-#72 claims. Nine tamper and false-success [negative sentinels](../tests/test_ncr_001e_sprocket_receipt_readback.py) passed in separate **no-model** [Actions run 40516673958](https://github.com/techrote/neurounner/actions/runs/40516673958), independently of the two 1 GB frozen-host processes. It proves source receipt custody consistency; it is **not** a third real model execution or a standalone scientific admission.

### Remaining explicit barriers

These results close the narrower **different-Python-process, same-runner frozen Sprocket adapter restoration** gap for both separately pinned variants. They do **not** prove an independent machine restart, 2048-unit full context, long multi-turn tool/memory continuity, original Instruct's tool-call argument/result grammar, Chat tool ability, or all #72 acceptance gates. SmolLM2 and void.1 remain distinct source lanes. Model-specific complete #72 source/runtime/fixture reviews and the #73 matched MLP real-host Sprocket/void lanes remain outstanding. BYTE-EOS-001 is tracked separately by #92/#94. **Four-model training admission remains denied; train_authorized=false, zero optimizer steps, no sealed holdouts and no draft PR merges.**

