# NCR-001E — independent PR #102 masked batch and cache reviewer prepass

Parent [#72](https://github.com/techrote/neurounner/issues/72);
research [#67](https://github.com/techrote/neurounner/issues/67);
reviewed [draft PR #102](https://github.com/techrote/neurounner/pull/102).

## Exact review boundary

- Fixed reviewed PR #102 head **`3592841802d85c442ca2521479b5d8795bee7233`**.
- Exact publisher-source commit `a003767e70d766bc14b40c89792543ee8641c8bc`, public SFT source SHA256 `e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69`. This is *source* identity, not cryptographic proof the public source generated the released training checkpoint.
- Original frozen adapter hook **`b95c010f9b2ba3cbd9f81317981316bb9c8acd19`** (root Instruct, taps 7,16,25).
- Exact **root** `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668` `model.safetensors`, 1,002,207,984 bytes, SHA256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. The Chat sibling SHA `551c6e26...` is forbidden.
- Four protocol keys remain `smollm2`, `sprocket_chat`, `sprocket_instruct`, `void1`. **`train_authorized=false`; no optimizer; no held-out fixture or comparative training.**

## Independent source/runtime questions

PR #102 demonstrates 108-position unpadded source dialogue cache parity, explicit host/action role masking, and 21 preserved synthetic generation negatives. The following source-structured **real frozen model** checks have not been independently demonstrated on that PR head:

1. Build the source-defined system/memory/user/assistant tool-call/host tool-result/assistant memory-write/assistant memory-read dialogue **without importing the reviewed serializer as authority**, then compare its actual tokenizer IDs and assistant-mask ownership to the exact PR #102 serializer. The second, unrelated ordinary dialogue is much shorter and never contains tool tokens.
2. Evaluate both dialogues together under **left and right** padding. Compare direct frozen-backbone and zero-NCA logits exactly, and activated-but-untrained NCA valid-position logits to each single-request result at fixed, unchanged absolute+relative `5e-4` tolerance. Record the numerical maximum deviation rather than just `allclose` status.
3. Change masked padding from token ID 1 to real host `<|tool_result|>` ID **7**. **Valid outputs must be bitwise unchanged**. Mutate the other user's real input while asserting cross-batch request independence. Include a *deliberate invalid-mask negative sentinel* proving that the harness detects unmasked padding.
4. On the left-padded batch, use the full attention masks and explicit position IDs for prefill and **four genuine one-token KV-cache forwards**. Compare both rows' final four positions to the full-sequence reference at source-prescribed 5e-4 absolute and relative gates; verify per-call state shape `[2,1,...]` and no remaining model hooks.
5. Independently hash the full original checkpoint and all source metadata, hash frozen tensors before/after the finite test, preserve runner, memory and any traceback on failure. **Zero optimizer steps**, no quality scores and no new held-out cases.

The new independent reviewer runner [`scripts/ncr_001e_pr102_masked_batch_review.py`](../scripts/ncr_001e_pr102_masked_batch_review.py) is deliberately disjoint from the original `scripts/ncr_001e_sprocket_wire_diagnostic.py` and does not reuse its original 21 generations as fresh runtime acceptance evidence. [Negative source-only tests](../tests/test_ncr_001e_pr102_masked_batch_review.py) execute without weight downloads. [Hosted reviewer workflow](../.github/workflows/ncr-001e-pr102-independent-batch.yml) separately executes real frozen root Instruct and uploads original partial JSONs on success or failure.

## Interpretation

Successful reviewer execution **would only add one source-identified, finite real-host masked/padded causal check to the Instruct lane**. It would not qualify learned tool use or memory persistence; the released Sprocket model card reports memory as weak. It would not independently admit the four-model #72 protocol: gradient and controlled optimizer-step checks remain blocked by the explicit training-admission requirement, and SmolLM2, Chat, void.1/Byte-EOS and matched-MLP controls have distinct unmerged dependencies.

**All PRs remain draft and unmerged. `train_authorized=false` remains unchanged; any failing source or causal test blocks this subgate instead of relaxing it.**

## Fresh original frozen-model result and source custody — 27 October 2026

**Actual original [GitHub Actions run 40590487839](https://github.com/techrote/neurounner/actions/runs/40590487839): BOTH jobs PASS**. Offline source-only reviewer passed **six negative/positive groups** before loading model weights; original full root Instruct frozen-model job ID `122493234132` succeeded separately. This was an original model execution, not readback of PR #102's prior JSON.

- Reviewer code during the original execution: `07eaac38a857a8d67285166944c91b7bc4543e9c`; original Actions merge SHA `724c05a508a2f53f7752198e7ddad1d14d6902c2`; reviewed PR #102 source fixed at `3592841802d85c442ca2521479b5d8795bee7233`, original hook `b95c010f9b2ba3cbd9f81317981316bb9c8acd19`.
- Source model independently re-downloaded and full SHA-256 verified: root Instruct original 1,002,207,984-byte `model.safetensors` SHA `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`; **different** Chat weight hash `551c6e26...`. Python 3.12.3, torch 2.14.1+cpu, 4 CPU threads, GitHub Actions runner `GitHub Actions 1000045500`, PID `2909`; maximum RSS **3,276,648 KiB**.
- Two independently source-serialized, **synthetic** prompts: **133 Instruct tokenizer units** containing full host/assistant control framing (45 assistant-supervised units), and a **31-unit** ordinary second conversation (8 assistant-supervised units).
- Frozen reference versus zero readout, left and right padding: **exact**, max drift `0`. Activated but **untrained** readout changed logits by `0.10165786743164062`, never by optimization.
- Both padded orientations' valid logits versus independent per-request forwards: **exact**. Replace *masked* padding positions with source tool-result ID **7**: **zero valid-position drift**, exact. Alter the unrelated second user's last real token: **zero change to first batch member's logits**, exact.
- **True batched masked-cache test:** full left-padded, publisher-structured `133`-position forward versus `129`-prefill + **4 sequential one-token cached** forwards, max absolute logit error **`3.218650817871094e-05`**, below **unchanged** protocol tolerance `5e-4` absolute and `5e-4` relative.
- Negative sentinel: deliberately *unmasking* the second example's many leading padding positions measurably changed its valid logits (max **`14.123714447021484`**). This is expected and proves the reviewer harness detects a false masking assumption.
- A→B→A request reset **exact**. Original backbone parameter hash identical before/after (`b73d854127c43266d123bdb266db63a15b4189db9bb3398be035463cbb4fac4d`). Frozen base **501,090,560** parameters, separate untrained NCA **658,904** parameters, gradient-free inference only, **optimizer steps zero**. No leaked original model hooks.

### Lossless actual original receipt archival

- [Original Actions artifact #12773497808](https://github.com/techrote/neurounner/actions/runs/40590487839/artifacts/12773497808), full ZIP SHA-256 `d0328b54347b952534ef1321953c5c94f6b66c48c791aa72e67bf2cf51d56b42` (2,975 bytes). The ZIP remains in Actions storage pending expiry; do **not** claim its bytes were committed.
- Its **unaltered original** actual-model JSON is **6,721 bytes**, SHA256 `1757dc6836b974d74d32acf6c45ccece4d1f69f980030b3bdffc8fad216f403e`. Its original exact source prefetch JSON is **874 bytes**, SHA256 `0272d25c42331841ce62ec26156b0805adb38f4a52e5258bac5c6272cbad538a`.
- Both complete JSON files are **losslessly preserved byte-for-byte** in [gzip/base64 original USTAR archive](../experiments/ncr-001/evidence-review/pr102-masked-batch/original-jsons.tar.gz.b64) (gzip SHA `ee01823ebccb19f7d417096db7b9819dc40d0b86ad690d4436dce9765de72460`), with [original model/hook/runner/ZIP provenance](../experiments/ncr-001/evidence-review/pr102-masked-batch/provenance.json).
- The separate [standard-library original-receipt auditor](../scripts/ncr_001e_pr102_masked_batch_audit.py) refuses source variant swaps, altered file/runner digests, incorrect masked-batch/cache numerical gates, corrupted source archive and false training/qualification claims. [Hosted archive-only CI 40591376322](https://github.com/techrote/neurounner/actions/runs/40591376322) **PASS**, seven positive/negative groups; **this is not a second actual-model execution**.

### Review decision and remaining blocks

**Result: PASS for this finite original frozen Instruct source-wire masked-batch/cache prepass only.** No claim of novel tool/memory competence: all source controls were synthetically host-provided; the publisher's negative memory-learning claims remain intact. **Not final independent #72 admission**, which requires the full frozen four-lane model checklist and the separately controlled gradient/update gate; no optimizer was run because `train_authorized=false`. Unmerged/preregistered model hooks, Chat sibling, SmolLM2, void.1/Byte-EOS and matched-MLP controls retain their independent gates. Neither source/code nor its original sealed holdouts have been altered. All stacked PRs remain draft/unmerged.

## Independent post-run source-custody repair — 27 October 2026

The existing original-source auditor, as committed on reviewer branch head
`31b052df673ec9fa382cf7a3d7587cdcbeb1707d`, verified the complete frozen
checkpoint and `config.json` hashes, but did **not** independently bind the
on-disk `experiments/ncr-001/source-pins.json` bytes or the original
`tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json`
digests/byte sizes from **both** original Actions JSON reports to the pinned
inventory. This was a **readback auditor coverage gap**, not evidence that the
model run or any original source bytes were incorrect.

[Commit `72bfd63`](https://github.com/techrote/neurounner/commit/72bfd63ba4562d9b693a8f2a734b4d7926054e98)
adds the absent independent five-source SHA-256 and byte-length checks,
an exact full `source-pins.json` SHA, and separate Chat-root checkpoint and
four-model inventory refusal. [Commit `9d62b0a`](https://github.com/techrote/neurounner/commit/9d62b0a34eb2e6f1e7432d393bce655a2b02ba84)
adds negative mutations for **every** original model/prefetch source record
and a changed on-disk source-pins inventory. Neither changes the original
read-only evidence or accepts a weakened cache tolerance.

A further historical adversarial reproducer explicitly checks out the
**prior auditor's exact immutable source** at commit
`31b052df673ec9fa382cf7a3d7587cdcbeb1707d` (Git blob
`2d1fb3eb6e3a30947d52b29d99d63be9fdaa936a`). It edits a
**temporary copy** of the source-pins JSON so the original Instruct tokenizer
SHA is false, while retaining every original Actions JSON and its checksum
unchanged. The old auditor **incorrectly accepts** the altered pin file; the
repaired auditor **rejects** it. This is a reviewer-controlled
*source-custody negative test*, not new model inference.

- [Commit `c03eac8`](https://github.com/techrote/neurounner/commit/c03eac8e43f48d71a12dc0e8f9152b1215bea1f5):
  exact old-auditor negative reproducer committed.
- [Commit `7c76b73`](https://github.com/techrote/neurounner/commit/7c76b738ff5e5a31fce5e633fc655df292a2b20f):
  CI checks out the exact historical reviewer source and refuses source
  revision/blob drift before the negative case is executed.
- [Independent archival Actions 40595754339](https://github.com/techrote/neurounner/actions/runs/40595754339):
  **PASS, 11 test groups**, including the actual historical
  original-accepted/repaired-rejected test. [Companion source
  Actions 40595754284](https://github.com/techrote/neurounner/actions/runs/40595754284):
  **PASS**; the source job ran and the 1 GB original-model execution was
  **correctly skipped** on reviewer-audit-only commits.
- Earlier [40595527851](https://github.com/techrote/neurounner/actions/runs/40595527851)
  passed the first five-source SHA/mutation gate before adding the exact
  historical negative proof.

These source-only, repeatable checks strengthen **independent evidence custody
for the 133/31-token Instruct masked-batch prepass**. They do not strengthen its
model quality or causal numerical evidence, are **not** a fresh model run and
do **not** satisfy all requirements for final independent issue #72 admission.
No source variant, evaluation fixture, frozen checkpoint, generation receipt,
original model hook, protocol or training flag was modified;
**`train_authorized=false` remains enforced.**

## Additional new source-bound HF generation/right-padding qualification — 27 October 2026

The prior finite masked-batch PASS (original [Actions 40590487839](https://github.com/techrote/neurounner/actions/runs/40590487839))
did **not** exercise the original HF `GenerationMixin.generate()` in both
cached and uncached modes, nor the harder right-padding case in which a shorter
sample's real incremental tokens are appended **after an attention-masked gap
in a shared KV cache**. A disjoint, bounded reviewer-owned test now covers
those previously untested call shapes, using the **same pinned Instruct and
publisher wire geometry**, without expanding or replacing any preregistered
held-out evaluation.

- [Original new frozen-source workflow **40598034739**](https://github.com/techrote/neurounner/actions/runs/40598034739)
  **PASS**: pure source test job `122518832102`, real original full
  1,002,207,984-byte root Instruct checkpoint job `122518920068`.
  Original source branch head `d727640e36ec9f7915aa208d69d9b78e4448f0c6`;
  original GH Actions merge commit `4470f3070abdee1a6b31d82b27663a8bfe0fdd41`.
- Fully reverified original weight SHA
  `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`,
  pinned model revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`,
  distinct Chat weight and all four original independent protocol keys. The
  original hook source, candidate serializer, publisher SFT source and
  original reviewer helper were checked out/hash-verified independently.
- Source inputs remained synthetic, **133 and 31 tokenizer units**. The
  right-padded batched cache prefills have **129 and 27 valid units** plus an
  attention-masked gap of 102 physical positions. Four genuine one-token
  forwards then append each sample's remaining units with separate,
  source-correct `position_ids`. The resulting valid-position logit
  maximum absolute deviations versus each independent original full forward
  were **4.920363426208496e-05** and **2.6702880859375e-05**.
  The protocol **5e-4 absolute/relative** limits remain unchanged.
- Replacing masked right-padding with actual host `<|tool_result|>` control ID
  **7** produced **exactly zero** change. Deliberately **unmasking** the same
  source gaps changed the second request's logits by **11.143782615661621**
  (required negative sentinel).
- Actual original `wrapper.generate()`, not a handwritten greedy loop,
  exercised disabled, zero-NCA and manually activated **untrained** NCA modes,
  each with `use_cache=true` and `use_cache=false`, for **four deterministic
  source units**. All six produced the same four tokenizer IDs
  `[8184, 746, 1066, 1525]`. Original frozen versus zero-readout
  processed scores were **bitwise exact** in both modes; cache-vs-uncached
  score maximum absolute deviations were at most
  **3.528594970703125e-05**, again under frozen 5e-4. Each score vector has
  31,999 finite logits and one expected suppressed EOS score; the
  independent comparison verifies the nonfinite masks agree exactly before
  checking finite values. Hook tracing confirms 4 completed, independent
  depth-3 forwards with per-token field reset.
- The manually altered **untrained** readout had a nonzero logit effect
  `0.03652691841125488`. The frozen base tensor SHA
  `b73d854127c43266d123bdb266db63a15b4189db9bb3398be035463cbb4fac4d`
  was unchanged before/after. A→B→A request reset was exact, no hooks leaked,
  no gradient/optimizer training occurred. Python 3.12.3, PyTorch 2.14.1+cpu,
  Transformers 4.57.6, four CPUs, runner
  `GitHub Actions 1000045611`, PID `3423`, maximum RSS 3,277,044 KiB.

### Newly durable original evidence and negative CI

The original [Actions artifact #12777685787](https://github.com/techrote/neurounner/actions/runs/40598034739/artifacts/12777685787)
was 5,754 bytes, ZIP SHA-256
`30bc7fa3231047561b5ab0142b59b7012e7e1ee4e72329198f7d778b65be9a1e`.
All **five original unmodified artifact members** (actual frozen-model
`pr113-rightpad-generate.json` 13,952 bytes, source-prefetch JSON 1,056
bytes, package inventory, source commit, source hashes) are [archived
losslessly](../experiments/ncr-001/evidence-review/pr113-rightpad-generation/)
under a deterministic USTAR/XZ Base64 container; original report SHA
`c50bc096161a2b0cfe40c19278adada101cb7314bff822064117ae347f232224`,
archived compressed XZ SHA
`3d2081f1494af69eb7cf8a9e90eea326ffc60ff1845483a4d02be34951728751`.
The **original ZIP bytes** are still in Actions retention, not Git; its
byte-identical *contents* are preserved and independently SHA-checked. A
[separate standard-library auditor](../scripts/ncr_001e_pr113_rightpad_receipt_audit.py)
and [tamper/missing-source/false-admission tests](../tests/test_ncr_001e_pr113_rightpad_receipt_audit.py)
re-read every original byte without executing a model.

Preserve all historical negative CI, rather than labelling it as model failure:

- [40597707675](https://github.com/techrote/neurounner/actions/runs/40597707675)
  and [40597891619](https://github.com/techrote/neurounner/actions/runs/40597891619)
  **FAILED before any model execution**: an invalid 41-character reviewer
  helper SHA was transcribed into the new CI guard. The exact original helper
  Git blob was never modified. Commit
  [`d727640`](https://github.com/techrote/neurounner/commit/d727640e36ec9f7915aa208d69d9b78e4448f0c6)
  corrected the expected 40-character blob, keeping strict validation.
- [40599710069](https://github.com/techrote/neurounner/actions/runs/40599710069)
  **FAILED only archival transport parsing**: the first auditor counted
  line breaks in a line-wrapped Base64 text envelope.
  [`1e28029`](https://github.com/techrote/neurounner/commit/1e280296c12ac921b9b0ab113192f5ee5be725d5)
  strips transport whitespace **before** enforcing the exact encoded length
  and original XZ/tar SHA. [Follow-up archival Actions
  40599803975](https://github.com/techrote/neurounner/actions/runs/40599803975)
  **PASS** with immutable original byte and negative-source checks. This is
  archive readback only, never a second real-model run.

The new dedicated source-matched [right-gap/generation workflow](../.github/workflows/ncr-001e-pr102-rightpad-generate.yml)
now distinguishes changes to the executable source from archive-only commits,
preventing repeated 1 GB checkpoint tests when only evidence or explanatory
documentation changes. **A green archive-only workflow is not new actual
frozen-model qualification.**

### Remaining scope and source-independent acceptance

This is an additional **finite PASS of original frozen Sprocket root Instruct
right-gapped cached continuation and actual HF generation path**, *not* a
quality or tool/memory benchmark. All source-structured tool-result/memory
turns were synthetic; the model was never asked to execute a tool, retain a
fact in storage, update weights or access sealed held-outs. The released
publisher SFT source remains descriptive code evidence, **not cryptographic
training-build attestation**. Final independent #72 source/runtime/optimizer
acceptance, full four-lane conformance, and matched #73 controls remain
blocked. The four distinct model identities, original fixtures and frozen
tolerances remain unchanged; `train_authorized=false`. **PR #113 and all
parent branches must stay draft/unmerged.**


## 29 October 2026 — exact committed representation immutability follow-up

The historical [original right-gapped/HF.generate physical run 40598034739](https://github.com/techrote/neurounner/actions/runs/40598034739) is unchanged. The PR #113 offline auditor previously normalized Base64 whitespace and parsed the JSON provenance before verifying values. Consequently, adding whitespace could change committed evidence text without changing its decoded member hashes or accepted semantic fields. The reviewed original archive and provenance Git blobs are now pinned separately: `3ca13858e4a878de259406a01f1a0baa1c97282d` and `e944ca2655729e60c587865a9168b34698031114`. Adversarial tests deliberately append an otherwise semantically harmless newline to each source and require refusal. The read-only workflow now checks the exact PR source HEAD, unchanged #102 base and four-lane protocol/source-pin Git blobs before auditing.

**Evidence boundary:** this hardening does not re-execute the frozen Sprocket Instruct model. The five original text members are committed losslessly, but the complete original 5,754-byte GitHub Actions ZIP (SHA-256 `30bc7fa3231047561b5ab0142b59b7012e7e1ee4e72329198f7d778b65be9a1e`) and complete original decoded source/model job logs are **not** yet Git-custodied. Original artifact #12777685787 expires 26 November 2026. Archive the exact original ZIP/logs in a separately scoped descendant without modifying these historical receipts. The prior source-only failed runs #40597707675 and #40597891619 remain negative evidence, not model failures. No training, heldouts, new source-model claims or final #72 admission; `train_authorized=false` and `full_72_conformance_pass=false`.
