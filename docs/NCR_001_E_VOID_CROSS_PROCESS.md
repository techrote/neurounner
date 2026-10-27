# NCR-001E — exact pinned void.1 ByteTokenizer and frozen Qwen3 fresh-process replay

**Parent:** [NCR-001 #67](https://github.com/techrote/neurounner/issues/67); independent per-variant [conformance gate #72](https://github.com/techrote/neurounner/issues/72). This draft is deliberately **stacked on reviewer-owned PR #93**, not main. Its dynamic tokenizer approval is inherited only through the exact static source review [#90](https://github.com/techrote/neurounner/pull/90) and offline runtime [#91](https://github.com/techrote/neurounner/pull/91), with a repeat source/AST check prior to new execution.

## Scope and independent witness

Earlier [#93](https://github.com/techrote/neurounner/pull/93) ran the actual pinned frozen 90M Qwen3 model with mixed UTF-8 text, true ByteTokenizer and bounded cache/no-op tests. However, its 'cold' adapter load created a new NCA wrapper **in the same interpreter against the same original model instance**. This is insufficient to prove a restart where model and adapter are both reconstructed.

This branch introduces **two separate Python interpreter processes** on one ubuntu-24.04 GitHub-hosted runner:

- **Producer:** Independently downloads/checks source-pinned bytes, refuses unsupported tokenizer dynamic code before source hash/AST review, instantiates the exact original frozen Qwen3 hooks from commit `5fa3dc8c0de153098d1800f7b0c809819620aef4`, then runs a 256-unit **synthetic mixed UTF-8** source prompt. A manually modified, **untrained** NCA readout provides a nonzero effect; original no-op logits remain exact. Cache proof is 240-unit prefill plus 16 genuine individual incremental byte-tokenizer steps at unchanged `atol=rtol=0.0005`; strict future-suffix and A→B→A reset checks are required.
- **Handoff:** Saves only NCA adapter, metadata, original-source manifest and float32 reference logits. Every binary has an original byte SHA-256. Source-validated manifest binds original model revision, full frozen checkpoint hash, ByteTokenizer source hash, exact original hook commit, source/protocol digests, input token IDs and producer PID; all training, EOS-impact and final-gate claims remain false.
- **Consumer:** A completely new Python process **reloads a new original frozen Qwen3 instance**, rehashes full source bytes, reruns the reviewed dynamic tokenizer identity and mixed UTF-8 input check, then checks producer PID differs and all handoff hashes match **before** opening the adapter or reference tensors. Fresh adapter/metadata reload must reproduce the producer's active and disabled tail logits exactly and preserve A→B→A; independently repeat a true 240+16 cached tail comparison under unchanged protocol tolerance.

Original source: `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`, Qwen3, 24 blocks × 640, taps [7,15,23], 360,622,744-byte checkpoint SHA `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`, tokenizer Python SHA `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`. Actual PAD/BOS/EOS IDs are 256/257/258, whereas `"<eos>"` is five ordinary byte IDs [60,101,111,115,62]. The new run does **not** change historical TRAIN-002b or enable the future-only serializer from [#94](https://github.com/techrote/neurounner/pull/94).

## Source qualification and evidence policy

The source guard `scripts/ncr_001e_void_process_guard.py` has pure offline negative tests for pin/model substitution, reduced lane count, changed byte source, mistaken EOS text-vs-ID, fabricated training/held-out/final results, relaxed causal tolerances, same-PID replay and malformed/tampered handoff hashes. The frozen-model runner uses the exact original wrapper #85 and prior tokenizer AST guard; all failures and partial successful stages must be uploaded with runner, source, CPU and memory metadata.

The [actual-host Actions workflow](../.github/workflows/ncr-001e-void-cross-process.yml) uploads original producer/consumer JSON and handoff on **success or failure**. This is only **transient staging** until independently promoted with original JSON byte digests, original artifact ZIP byte SHA and no-loss preservation in this repository. Do not claim verified cold restart from unit tests or an uninspected green CI status.

The test intentionally does **not** establish a different physical runner/OS, arbitrary full 2048-byte context, unobserved Sprocket tool/memory role serialization, trained policy, NCA-vs-MLP effect, EOS-generation improvement, historical BYTE-EOS-001 causal contribution or independent four-model #72 acceptance.

**Current classification: prepared for actual pinned-host CI, not yet independently accepted.** No optimizer construction or steps, no sealed held-out prompts, no source fixture changes and no training: `train_authorized=false`. Keep all dependency PRs draft and unmerged pending separate review.
