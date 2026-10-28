# NCR-001E — independent original Sprocket full-position publisher-wire mask replay

**Status:** new draft source/tokenizer-only evidence check, stacked on [PR #162](https://github.com/techrote/neurounner/pull/162) (original frozen-model producer [PR #161](https://github.com/techrote/neurounner/pull/161)). This is **not** a fresh frozen-model invocation, distinct scientific acceptance of #72, NCA–MLP comparison, tool/memory competence proof, or model training.

## Motivation: total counts and marker checks alone are insufficient

The original PR #161 12-turn source-structured Instruct witness used 123 tokenizer units and 45 assistant-labelled units. The original PR #162 receipt auditor checked control-ID ownership, 12 end-of-turn markers, host/model/host memory-read role types, and total label counts. Those checks cannot by themselves exclude a **compensating labelling error**: incorrectly supervise one ordinary host tool-result body token while unsupervising one ordinary assistant body token. Both the aggregate count and all special-control marker positions could remain unchanged. This is an audit coverage gap, **not a claim that such a failure occurred in the original receipt**.

This draft adds an independent, whole-position reconstruction of the exact assistant-only supervision vector, without relying on the publisher renderer's returned vector as its expected answer. For each frozen synthetic source turn, it checks the exact opening role ID, every tokenizer-encoded body position, closing real end ID 5, unsupervised header and mandatory unsupervised dangling assistant cue. Assistant body and end are labelled; system/user/tool/host-memory body and end are not. Tests explicitly reject compensating host/assistant body swaps and host-memory body leakage.

## Exact provenance and original evidence custody

- Parent PR #162 exact head: 4cf8c052c5ad91c384f2e15b68793a92c9a80cfe, stacked on original physical PR #161 head aadbfe823c9778066acac0a0869e0c559a533f05.
- Original physical [Actions run 40718706396](https://github.com/techrote/neurounner/actions/runs/40718706396), exact root Instruct weight SHA-256 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864. Distinct Chat weight source is **not** substituted. **No weights downloaded in this replay.**
- Original [failed source run 40718490313](https://github.com/techrote/neurounner/actions/runs/40718490313) (11 Python test-binding errors, no physical model execution) remains preserved in the original three-ZIP Git bundle.
- The original three Actions ZIP containers, all raw members, original whole-host receipt SHA-256 44f59fdf8eaa65ff6c4721b7f9c0c40986297442cf93382dce9201c83f33dc6e, complete successful/failed job logs and source-bound negative six-token incomplete tool-call prefix remain unchanged under experiments/ncr-001/controls-evidence/postlease-wire-20261028/.
- Original publisher renderer is read from immutable PR #102 commit 3592841802d85c442ca2521479b5d8795bee7233 and Git blob 4e558ae530a1b79050e22bee587b3caf056e4563. That publisher-derived source does **not** attest historical checkpoint-build provenance.
- The 12-turn synthetic fixture is extracted as a literal AST return from the unmodified original physical producer source and its SHA is checked against the archived physical receipt. No original fixture, prompt, label or evaluation suite is rewritten.
- Downloads only SHA-checked root Instruct README.md, config.json, tokenizer.json, tokenizer_config.json and special_tokens_map.json at revision eb157d4ddddc4619e661d4a2426bb2dd000fc668 with remote Python code disabled; after fetching, tokenizer replay is offline.

The independent full-mask reconstruction is checked against **both** the publisher renderer and the original actual-model-host prompt, int64 token-ID, assistant-label and explicit padded-attention-mask hashes. Any mismatch fails closed with a partial failure receipt.

## Validation and scientific limits

The focused no-model source mutation tests and original ZIP auditor run before the real-tokenizer replay. The dedicated exact-head GitHub workflow ncr-001-sprocket-wire-fullmask.yml archives source/negative logs, original archive readback, physical-receipt comparisons, runner identity and failures as **Actions staging artifacts**. Do not regard them as permanently filed evidence until the exact output and source hashes are promoted to Git.

The original physical model produced only a six-token incomplete tool-call prefix for disabled and manually perturbed adapters, never a complete valid tool action. No persistent memory, learned EOS, behavioral competence or comparative quality is established. Independent scientific review must still examine #161 original physical source/runtime and all four frozen-model #72 gates.

All NCR draft PRs remain draft. Do not merge or admit training on green source-only CI. The four distinct identities (smollm2, sprocket_chat, sprocket_instruct, void1), locked preregistration and train_authorized=false remain unchanged. No optimizer, held-out opening, model-weight change, future fixture modification, comparative NCA/MLP training or reinterpretation of historical BYTE-EOS outcomes is allowed here.

Authority: [#67](https://github.com/techrote/neurounner/issues/67), [#72](https://github.com/techrote/neurounner/issues/72), [#73](https://github.com/techrote/neurounner/issues/73), [#92](https://github.com/techrote/neurounner/issues/92), [#161](https://github.com/techrote/neurounner/pull/161), [#162](https://github.com/techrote/neurounner/pull/162).
