# NCR-001E — permanent original Actions custody for Sprocket full-position mask replay

This is a **read-only evidence preservation draft** stacked on original tokenizer/source check [PR #181](https://github.com/techrote/neurounner/pull/181) at exact source `40efcb5748894c6c7847c18b1ad442a455bd0aa3`, which is itself stacked on original post-lease frozen root Sprocket Instruct host ZIP archive [PR #162](https://github.com/techrote/neurounner/pull/162) and physical [PR #161](https://github.com/techrote/neurounner/pull/161).

## The question it answers

Can a byte-exact original real-tokenizer source replay of the original physical wire mask be independently reconstructed after GitHub Actions staging expires, including the negative/refusal tests and logs, without changing original model source or scientific gates?

This branch stores the exact original [Actions run 40782101613](https://github.com/techrote/neurounner/actions/runs/40782101613) source and tokenizer ZIP containers and complete unedited decoded CI job logs. The result is *source/serialization*, not a fresh frozen-model invocation or independent #72 scientific admission.

## Original evidence and negative results

Both exact-head original CI jobs succeeded on the PR #181 head:

- Job 123146082042 source/negative: **13/13 adversarial and positive tests passed**; also required unchanged original PR #162 Git archive with 24 physical original checks, including failed source run 40718490313. Complete original decoded job log committed.
- Job 123146150961 real pinned root Instruct tokenizer without weights: independently reconstructed all **123** token positions, **45** assistant-supervised positions and **78** host/other positions, plus the original whole-prompt, token-ID, supervision mask and explicit left-padding attention hashes. Complete original decoded job log and original receipt preserved.
- Original pinned renderer: PR #102 Git blob `4e558ae530a1b79050e22bee587b3caf056e4563`; original physical producer PR #161 source SHA `aadbfe823c9778066acac0a0869e0c559a533f05`; original physical run 40718706396 pinned **root Instruct** checkpoint (distinct Chat checkpoint untested here). Original model-generated output was an **incomplete six-token tool-call prefix** and remains negative.
- Original no-model runner: **GitHub Actions 1000046991**, Python 3.12.3, Linux, original run ID 40782101613; **no model weights loaded, optimizer steps, heldout fixtures, training or comparative scoring**.

## Exact original ZIP byte custody

Manifest: `experiments/ncr-001/controls-evidence/sprocket-wire-fullmask-20261028/manifest.json`.

Original GitHub source artifact **12857787221** (1,307 bytes; SHA256 `6502f7766e385a7825585dd21fed8129b3f1a095afb8569bd4a1dfabdc739279`) and tokenizer artifact **12856874121** (3,774 bytes; SHA256 `b941b681d61b0d1182eb3d71578f368aa1026e0628d05725668fe1e6295a50dd`) are preserved **byte-for-byte** in one deterministic TAR/XZ/base64 file. Original compressed XZ SHA256 `cc1010df896950d5debe0eec45052657cbdc14451c0f53cce5ee0b79aa48e856`. All four raw original ZIP members have individual SHA256/length pins. Original complete decoded GitHub source and tokenizer job logs have exact Git blob SHA pins. Original standalone receipt.json is also byte-identical to the original tokenizer artifact member.

Original Actions artifacts expire 27 November 2026; this immutable Git custody does not. The new stdlib auditor `scripts/ncr_001_sprocket_wire_fullmask_custody.py` reconstitutes both original ZIP containers, verifies all original bytes and four original members, complete job logs, runner, original result hashes and source identities, and refuses false scientific claims or lost negative results. Negative tests in `tests/test_ncr_001_sprocket_wire_fullmask_custody.py` alter ZIP bytes, complete logs, results, original-run identity, model/head, label hashes and training/review flags. The exact-head workflow additionally downloads both original ZIPs from GitHub and compares their bytes against Git, preserving failures as Actions staging.

## Scientific boundary

Source-only token-supervision transport is qualified narrowly; **no new physical frozen-model run** and no evidence of valid generated tool calls, persistent model memory, learned EOS stopping, NCA quality gains or independent scientific acceptance. The original physical failure, archived model receipt and byte-EOS legacy uncertainty are preserved, not reinterpreted.

Four independently pinned model lanes remain distinct. `train_authorized=false`, `full_72_conformance_pass=false`, no merge and no comparative NCA/MLP optimizer runs or held-out access. This evidence draft requires independent code/source review and upstream stacked PR reconciliation before any possible #72 decision.
