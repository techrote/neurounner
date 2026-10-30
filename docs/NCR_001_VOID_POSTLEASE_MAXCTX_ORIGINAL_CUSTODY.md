# NCR-001 void.1 post-lease 2048-context original evidence custody

**DRAFT / no training / no final #72 admission.** Parent #67, #72, #73, #79 and BYTE-EOS-001 #92. This evidence branch is strictly stacked on physical qualification PR #169 at exact source head `1438f45efc0dab8ed669a814b4897fc7a839302f`.

## Original physical execution

Original [Actions run 40755894558](https://github.com/techrote/neurounner/actions/runs/40755894558) completed both jobs successfully. The source/negative job passed **8/8** tests. The physical job loaded the exact pinned `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563` Qwen3 checkpoint (360,622,744 bytes; SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`) and the reviewed ByteTokenizer Python source SHA-256 `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`. Corrected future EOS serialization source remained fixed at #163 commit `9fbdee880697903aaf5b91cc29442c9e53479e50`.

The physical receipt reports **32 checks** and status `PASS_VOID1_POSTLEASE_512_1024_2048_AND_GENERATE__72_BLOCKED`. Both the persistent NCA and the parameter-matched gated MLP preserved exact disabled-vs-zero logits at 512, 1024 and 2048 tokenizer units, demonstrated manually activated **untrained** nonzero residual effects, and passed genuine incremental KV-cache comparisons under the unchanged absolute/relative tolerance of **0.0005**.

| Original frozen-host cache max abs | 512 | 1024 | 2048 |
|---|---:|---:|---:|
| Persistent NCA | `1.3351440e-05` | `1.3351440e-05` | `8.5830688e-06` |
| Matched gated MLP | `9.5367432e-06` | `1.0490417e-05` | `7.6293945e-06` |

Both adapters also passed strict 2048-unit future-suffix causality. Real Hugging Face `GenerationMixin.generate()` was exercised in disabled, zero-readout and manually active-untrained modes with cache off/on. All four score steps for all six adapter/mode pairs stayed within the same fixed numeric tolerance and preserved identical nonfinite masks.

The full frozen-backbone tensor fingerprint remained exactly `7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904` before/after. Peak process RSS was **1,306,944 KiB** on four threads of an AMD EPYC 7763 host; measured model phase elapsed time was **245.39806 s**.

## Negative observations preserved

The diagnostic source string contained 4,256 UTF-8 bytes / ByteTokenizer units and was deliberately sliced to exactly 2,048 ordinary byte IDs before model execution. The tokenizer emitted its standard warning that the initially constructed 4,256-unit sequence exceeds the model's 2,048 maximum. This warning is retained in the original job log; it is not hidden or interpreted as a model failure because the receipt records the actual selected input length as 2,048.

All **12** four-step generation records produced IDs `[101, 110, 32, 109]`. **None emitted actual EOS ID 258.** No learned stopping behavior, historical BYTE-EOS effect size, behavioral competence, or adapter improvement is inferred.

No optimizer, backward pass, held-out fixture, comparative NCA/MLP training, or scientific review occurred. `train_authorized=false` and `full_72_conformance_pass=false` remain explicit in the original receipt.

## Permanent original-byte custody

The original Actions artifacts are preserved losslessly as base64-encoded ZIP byte strings under `experiments/ncr-001/controls-evidence/void-postlease-maxctx-20261028/`:

- Source artifact ID **12844994507** — original ZIP 497 bytes, SHA-256 `37babf3139f3d74deac90b64a8927db4bf697de9a70cbd5633798676ef79a427`; one member `source-tests.log`, SHA-256 `be45dd6567ebad0f711329df7e7e9441d272bfa04106bb00f21844db64dc89b7`.
- Qwen artifact ID **12845743165** — original ZIP 6,268 bytes, SHA-256 `eb209e2be9ab73615db23b60e5d75cfbe0bb14f9aa05ba81373a2d9231bb9026`; four original members: `maxctx.json`, `maxctx.log`, `prefetch.log`, and `runner.txt`, each independently hashed in the manifest.
- Complete decoded original source job **123054571007** and Qwen job **123054647348** logs are committed alongside the ZIP encodings.

`scripts/ncr_001_void_postlease_maxctx_original_audit.py` decodes and hashes the exact original ZIP bytes, validates every member, independently rechecks the source/model/runtime/no-training boundaries, all 32 physical checks, 2048 causality, GenerationMixin cache parity and the explicit **non-EOS** result. It then applies 15 semantic falsifications plus one ZIP-byte corruption and must reject all 16.

The dedicated exact-head workflow additionally re-downloads the **original** two Actions artifact ZIPs and both original job logs from GitHub and compares them against the Git archive. This is evidence custody/readback, not a new model run.

## Qualification boundary

This advances the post-lease void.1 source/runtime pretraining record from prior 32/128-unit evidence to an original frozen-host **2048-unit** cache/generation witness. It does **not** authorize training, prove learned EOS behavior, complete independent scientific review, or merge any stacked draft.

The separately required four-lane #72 decision, dependency reconciliation and any future comparative study remain blocked. Sprocket Chat, Sprocket Instruct and SmolLM2 retain their independent checkpoint identities and evidence chains. Historical TRAIN-002b results are unchanged.
