# NCR-001E — Sprocket post-warmup learning-diagnostic reconciliation

**Status: source-bound NCA-only diagnostic observed for two distinct Sprocket frozen hosts; complete scientific #72 still BLOCKED.** This is an independent *offline readback* extension stacked on [#160](https://github.com/techrote/neurounner/pull/160), itself on original physical [#159](https://github.com/techrote/neurounner/pull/159). No new model invocation, training, tokenizer execution, MLP optimizer, held-out evaluation or merge.

## Why this distinct reconciliation exists

The older formal Chat [#124](https://github.com/techrote/neurounner/pull/124) head `b2461a66241bc2fd0fbc389d7b3b2d10f729d2ab` and root-Instruct [#122](https://github.com/techrote/neurounner/pull/122) head `90f3006dd59b2a731db19064472d9f2215ac62d9` correctly described **three learning/update/loss checks as unobserved in their original pretraining cohort**. These are historical, non-merged, different branches. Do not rewrite their reports to pretend their old evidence contained later tests.

A later distinct real-host [Actions #40710171622](https://github.com/techrote/neurounner/actions/runs/40710171622), original exact source `df8479aae34550cba014d910a73e30e59cd493e2`, performed exactly one **NCA-only, in-memory SGD diagnostic** per variant on a fixed 19-unit *synthetic*, assistant-only source-framed sample, not preregistered training or evaluation. Both used fully byte-verified, **different** frozen weights. The separate [#160 Git-custody auditor](https://github.com/techrote/neurounner/pull/160) has verified the six original raw artifact members, three original physical/source jobs and the original failed formatter-hash readback job.

| Narrow observation | Chat (chat/ checkpoint) | Root Instruct (root checkpoint) |
| --- | --- | --- |
| Original weight SHA-256 | `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97` | `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864` |
| First synthetic supervised loss | 3.286933183670044 | 4.1148858070373535 |
| Loss after one NCA-only diagnostic step | 1.8742421865463257 | 2.7516422271728516 |
| Post-warmup nonzero observe/perception/rule/readout gradients | Original-host observed | Original-host observed |
| Full frozen-backbone fingerprint before/after | Identical within Chat | Identical within Instruct |
| Three ordered, distinct within-token depth fields | Original-host observed | Original-host observed |

The source-only [reconciliation script](../scripts/ncr_001_sprocket_postwarmup_reconcile.py) first verifies **eight committed Git blobs** (manifest, all three exact Base64 chunks, original source/Chat/Instruct job logs, original failed readback job log), then independently decodes the existing archived raw receipts and invokes the pinned source/runner/variant/gradient/frozen-backbone validator on each. Its companion [adversarial tests](../tests/test_ncr_001_sprocket_postwarmup_reconcile.py) reject changed decoded logs, archive whitespace, cross-variant weights, missing post-warmup gradients, false optimizer counts, backbone mutation, incorrect terminal EOS, held-out substitution, and false training/scientific claims.

## Protocol mapping and explicit NON-admission

- `base_weight_unchanged_after_step`: **observed in the bounded, synthetic, NCA-only diagnostic** for each own checkpoint.
- `nonzero_adapter_gradient_after_warmup`: **observed** on the synthetic supervised sample in each distinct checkpoint.
- `finite_state_and_loss`: **partial** synthetic positive finite loss and 3 distinct depth fields. It does **not** substitute for the complete multi-context stability and numerical-finiteness gate.
- Original #122/#124 machine-readable decisions remain unchanged historical reports. This descendant does **not** combine them into a final #72 per-model scientific adjudication and must not claim all checks PASS.
- No generated tool/argument/result or memory competence is shown; original root-Instruct incomplete six-token tool-call prefix remains negative evidence from separate #161.
- No #73 matched NCA/MLP optimization, learned quality metric, sealed held-out experiment, or model training. Historical void.1 EOS ID is 258; TRAIN-002b effect remains **UNMEASURED**.
- Three original #159 **whole ZIP containers** remain SHA-pinned but Actions-hosted (expire 27 Nov 2026), although all six original members and the complete decoded source/physical logs already have permanent Git custody. Full-ZIP container preservation remains a separate archival follow-up.

This is not a fresh physical-host run: source-based claims refer to #159; the new workflow executes **stdlib-only evidence and negative tests**. Original failed formatter-hash readback is preserved, not reclassified as model failure.

**Required next:** independent exact-source scientific #72 adjudication against all relevant conformance contexts and caches for each distinct variant, and complete stability/multi-fixture accounting; then scoped review of this reconciliation. Do not merge drafts or admit #74/#75/#80/#81 training without explicit protocol authorization. The [owner exception on #72](https://github.com/techrote/neurounner/issues/72#issuecomment-6483647556) accepts existing same-account COMMENTED scoped reviews for #161/#162/#181/#182 administratively; it is not a GitHub APPROVED or general scientific PASS and does not silently extend here.

`train_authorized=false` · `full_72_conformance_pass=false`.
