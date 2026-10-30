# NCR-001E — void.1 post-diagnostic evidence reconciliation (2026-10-28)

**Preliminary source-bound reconciliation, NOT independent scientific approval, not a final #72 PASS, and NOT an authorization for comparative training.** Parents: [#67](https://github.com/techrote/neurounner/issues/67), [#72](https://github.com/techrote/neurounner/issues/72), [#73](https://github.com/techrote/neurounner/issues/73), [#79](https://github.com/techrote/neurounner/issues/79), [BYTE-EOS #92](https://github.com/techrote/neurounner/issues/92).

## Source boundary and chronological evidence

The **historical** source-only post-lease #72 matrix in `experiments/ncr-001/adjudication/void1-postlease-pretrain.json` belongs to draft [#171](https://github.com/techrote/neurounner/pull/171) at `1738ac2b6bc6105d57d6a5b4c35f4beaad94ff86`. It correctly found pretraining source/cache/cold replay adequate while leaving three learning checks blocked. This matrix remains unchanged, including `optimizer_steps_observed=0` and the 12 original negative EOS-generation observations.

A **later and separate** original-host synthetic diagnostic ran on draft [#172](https://github.com/techrote/neurounner/pull/172) at exact head `2810e800ef890f0c99f9ea44adca29f879b5d0ea` in [Actions run 40765025332](https://github.com/techrote/neurounner/actions/runs/40765025332). Original weight SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3` and tokenizer Python SHA-256 `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03` are the exact pinned Qwen3/byte sources. The corrected future serializer supplies true EOS ID 258. The source tests passed 11/11; the real frozen host completed a *single* NCA-only in-memory SGD diagnostic on **one public synthetic example**, with 10 prompt units and 21 shifted supervised units. The original source and host ZIPs have SHA-256 `b153d177497f15227545bfe6b717a6430219a662f2b0b868e7a68b89682a90d0` and `8e9152042fb909f18b13589e6e21552c40f3104c4cc292cf9a643c600fbf08b5`, respectively.

[Draft #173](https://github.com/techrote/neurounner/pull/173) at `d6b730083bdf2d7d213e647f9770460bc1424e42` permanently committed complete, **unaltered** original ZIP bytes, original job logs, manifest, reporter and adversarial auditor. [Exact-head original re-download readback 40766551985](https://github.com/techrote/neurounner/actions/runs/40766551985) passed. Two earlier failed custody runs [40766098835](https://github.com/techrote/neurounner/actions/runs/40766098835) and [40766312502](https://github.com/techrote/neurounner/actions/runs/40766312502) remain archived; they were archive transcription failures, not model numerical failures.

## Newly evidenced, still unaccepted checks

| Previously blocked #171 required check | Original #172 diagnostic observation | Current disposition |
|---|---|---|
| `base_weight_unchanged_after_step` | Frozen Qwen3 complete tensor fingerprint `7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904` unchanged after one NCA-only SGD step; disabled logits bit-exact | Evidence present; distinct scientific review pending |
| `nonzero_adapter_gradient_after_warmup` | Zero readout has first-pass readout gradient but no upstream gradient; after one in-memory step, observe/perception/rule/input-norm/readout gradients all nonzero, with three distinct ordered depth fields | Evidence present; distinct scientific review pending |
| `finite_state_and_loss` | Native shifted byte/EOS synthetic loss finite: `3.065654993057251` before and `1.981663703918457` after the one diagnostic step; all ordered field states finite | Evidence present; distinct scientific review pending |

**Do not interpret the loss decrease as improved held-out accuracy, learned EOS emission, train readiness or a matched NCA/MLP comparison.** No MLP optimizer step, NCA/MLP comparative training schedule, trained artifact, held-out fixture, changed protocol threshold, or downstream independent scientific review is part of this evidence. The 12 max-context four-step generations **still emitted no true EOS ID 258**; the historical TRAIN-002b EOS defect's behavioral effect size remains **unmeasured**.

## Reproducibility and intentionally retained refusal states

`scripts/ncr_001_void_postdiagnostic_reconciliation.py` rebuilds the **previous** matrix from its original post-lease archives, checks it against the original committed JSON, invokes the established original-ZIP/member/job-log/source auditor from #173, executes its 14 falsified-receipt refusal cases, and independently reconciles the diagnostic's raw loss/gradient/field/source/runner evidence with the earlier matrix. A further **18 cross-source mutation tests** reject wrong checkpoint variant, changed policy, falsely approved old learning checks, missing negative EOS, altered pinned weights, scientific-review promotion, extra diagnostic steps, changed run, literal EOS bytes, faulty causal mask length, missing upstream gradients, unchanged readout, nonfinite loss, reused depth field, altered base, replaced run and omitted check.

The generated machine-readable addendum is `experiments/ncr-001/adjudication/void1-postdiagnostic-review-pending.json`, with `PRELIMINARY_DIAGNOSTIC_EVIDENCE__SCIENTIFIC_ADMISSION_BLOCKED`. It is *not* a replacement for #171, and its three candidate observation fields deliberately retain `DIAGNOSTIC_EVIDENCE_PRESENT__DISTINCT_SCIENTIFIC_REVIEW_PENDING`, not final PASS. The CI is source-only/stdlib-only and checks exact Git blob hashes, original evidence, source-head identity and the unchanged four-lane gate. It does **not** rerun the frozen model and may not be counted as an independent fresh-host qualification.

An independent qualified reviewer must evaluate the original code, physical logs and evidence interpretation, #171→#172→#173 stacked source dependencies, byte/EOS and CausalLM supervision semantics, complete four-model #72 and #73 obligations, and any unreconciled negative evidence before any gate decision. Maintain **`train_authorized=false`, `per_model_gate=false`, `full_72_conformance_pass=false`**. Keep stacked PRs draft/unmerged. No sealed holdouts, comparative NCA/MLP training, model promotion or fixture changes.

## Strict numeric evidence types — 30 October 2026 source-only hardening

A later independent adversarial check identified a Python type-confusion hole in the **derived cross-receipt validator**, not in the immutable physical receipt: `bool` is a subclass of `int`. Thus a forged `diagnostic_optimizer_steps=true`, positive `gradient_l1=true`, zero `gradient_l1=false`, tap index `true`, or tensor dimension `true` could previously satisfy direct numeric equality/positivity comparisons. A malformed string gradient could instead raise an unclassified `TypeError`. These are not legitimate physical measurements or structural identities.

The amended `check_inputs` now requires exact integer types for the single diagnostic step, true EOS ID 258, byte/shift counts and ordered tap metadata; finite non-boolean real measurements for both native losses and first/post-warmup gradients; five exact integer field dimensions; and 64-character lowercase hexadecimal field digests. Ten direct adversarial substitutions are asserted to fail with the explicit `REFUSED` disposition. The primary immutable #173 ZIP/log custody auditor, historical #171 matrix, physical #172 code and receipts, original negative generation observations and four-lane policy are unchanged.

This **does not** demonstrate an additional Qwen3 physical run, change any previously recorded numerical result, make the historical EOS effect measurable, approve science or authorize comparative training. Its test job is source-only and needs a successful literal-head workflow before claiming qualification. All existing #72/#73 admission and draft merge gates remain closed.

## Independent original Unicode/EOS tensor oracle — 30 October 2026

The original #172 frozen Qwen3 diagnostic records three SHA-256 fingerprints of
its *public synthetic-only* input-ID, supervised-label and attention-mask
`torch.long` tensors. The #173 primary custody validator already pins the
original complete ZIPs/logs and the input-ID and label fingerprints, but did
not independently recompute the attention-mask fingerprint. The #175
cross-receipt validator now performs a **separate standard-library byte
reconstruction** without importing Torch, the remote ByteTokenizer, a
checkpoint or any held-out material.

The exact immutable #172 source fixes prompt `"Q: π?\nA: "` (10 UTF-8
bytes) and completion `"🔧 café 日本語"` (20 UTF-8 bytes). The independent oracle
constructs input IDs as 10 prompt bytes + 20 completion bytes + real
integer **258**, labels as 10 `-100` positions + 20 completion bytes + real
EOS **258**, and a full 31-position ones attention mask. Hugging Face's
internal causal `labels[..., 1:]` shift therefore retains **21 supervised
targets**, including terminal EOS. It hashes contiguous signed
little-endian 64-bit integer bytes, matching the original Ubuntu x86_64
`torch.long` tensor serialization. Fixed hashes (SHA-256):

| Tensor | Independently reconstructed original digest |
|---|---|
| IDs | `ee6fa2a893301b858b7fdac7fe2b9190cd1fc786ddff1ffdf906da757488c09e` |
| Labels | `260dec3e7622aeada51e0483e005ad9d59cbae8f186d41432c381f2c35835241` |
| Attention mask | `88986881ba52ad36b430e41cd24497b9cf9a64f8e6a979290ebdee36b1b6f85c` |

Two additional Python tests compare all eight reconstructed fields with
the raw original archived host receipt and require **five independent
mutation refusals** (including changing the previously un-reconstructed
mask SHA, ID SHA, label SHA, UTF-8 source-byte count and bool-as-int byte
count). This closes a specifically scoped source-replay integrity gap; the
original physical receipt and #171 historical blocked decision remain
unchanged. It cannot establish new model causality, generated EOS
competence, historical TRAIN-002b correction impact or a full #72 PASS.
`train_authorized=false`, `per_model_gate=false`,
`full_72_conformance_pass=false`. Preserve the original failed archival
runs and prior source-only attempts in Actions history.
