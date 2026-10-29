# NCR-001J — Frozen Sprocket-500M Instruct NCA hook prepass

Issue [#78](https://github.com/techrote/neurounner/issues/78); parent [#67](https://github.com/techrote/neurounner/issues/67). This source-bound integration is stacked on common NCA PR #82; the four-model protocol PR #77 remains draft.

## Source identity and architecture

Use **root Instruct** weights from kandivault/sprocket-500m, exact revision eb157d4ddddc4619e661d4a2426bb2dd000fc668, model.safetensors. This is **not** chat/model.safetensors even though both variants have the same 26-block, 1280-width Llama architecture.

The full 1,002,207,984-byte root safetensors checkpoint has independently verified SHA-256:

    1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864

The independent Chat weight SHA differs. The adapter constructor requires the root weight digest as an explicit argument and the hosted loader hashes actual fetched bytes **before** constructing the adapter. The caller-supplied hash is not, by itself, cryptographic proof of the loaded base's identity; the source-bound loader and evidence are authority.

The model-specific NCA uses explicit per-forward state, zero-initialized readout, 8×8×4 local cellular fields after decoder blocks [7,16,25] and 658,904 trainable NCA parameters. The 501,090,560-parameter original Sprocket Instruct model is frozen; no weight updates were performed.

## Hosted result

The corrected [Actions run 40423957135](https://github.com/techrote/neurounner/actions/runs/40423957135) **passed**. Seven offline source-variant/frozen-toy-hook tests passed. On a four-thread AMD EPYC 7763 runner, using PyTorch 2.14.1+cpu, Transformers 4.57.6 and exact hashed root weights:

| Test | Measured result |
|---|---:|
| Zero-initialized logits vs same original Instruct checkpoint | Exact, maximum difference 0 |
| Padded two-request batch no-op | 0 |
| Greedy short generation | Identical, three separate decoder-forward resets |
| Readout gradient L1, no optimizer step | 3,930.003 |
| Manually nonzero untrained NCA max logit effect | 0.0113402 |
| Full vs incrementally cached logits, same perturbed NCA | 0.000103235 max absolute error (tolerance 0.0005) |
| Causal future-suffix invariance | 0 |
| Peak process RSS | 3,314,644 KiB |
| Actual model optimizer steps | 0 |

The complete [3,955-byte raw receipt](../experiments/ncr-001/sprocket-instruct-preflight/receipt.json) has SHA-256 cc7d677300b6b88e1b01a52a32fe9ae74ee111b9be58b3d9c006d0e346fc87f2, and [provenance](../experiments/ncr-001/sprocket-instruct-preflight/provenance.json) retains the source run, exact artifact ZIP digest, checkpoint and tokenizer hashes, plus the earlier superseded mislabelled attempt.

## Qualification limits

These tests use ordinary raw-text tokenization. **Sprocket Instruct tool-call, tool-result, memory-token and chat formatting semantics remain unqualified.** A full independently reviewed #72 admission needs long contexts, request isolation, cold reload and explicit role/control-token fixtures.

The modified readout was manually perturbed, not trained. No matched MLP control or held-out task evaluation was run. The protocol's training-admission flag remains false, and this PR is **not** authorisation for comparative training, production deployment, or a speedup claim.

The model-specific wrapper duplicates the proven SmolLM2 hook algorithm for now. Generalising and consolidating the framework is an optional later refactor after all four model-specific source-conformance surfaces are established, not a prerequisite for this bounded proof.
