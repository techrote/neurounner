# NCR-001E — Sprocket Chat right-gap and HF GenerationMixin reviewer

Parent issue: [#72](https://github.com/techrote/neurounner/issues/72).  
Reviewer draft: [#123](https://github.com/techrote/neurounner/pull/123).

This review fills a specific Sprocket **Chat** source/runtime gap without
training: previous evidence covered ordinary source formatting, no-op parity,
left/right padding, 48/256/2048 cache probes, suffix/request isolation and
fresh-interpreter adapter reload, but the original Chat `GenerationMixin`
path had not been compared with **both `use_cache=false` and
`use_cache=true`**.

## Exact source

- Repository/revision: `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`
- Exact distinct Chat checkpoint: `chat/model.safetensors`, 1,002,207,984
  bytes, SHA-256
  `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`
- Root Instruct sibling is independently different:
  `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`
- Original frozen Chat NCA hook commit:
  `83a56835a7476f17a17980d28dd1a7c14bb6f2f6`; taps `[7,16,25]`
- Ordinary source formatter Git blob:
  `39cf015735e74a1d1e08682b623404bc1e600b9b`
- Pinned model-card SHA-256:
  `09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9`

**Scope is ordinary source-documented user/assistant Chat roles only.**
The test explicitly refuses to inherit root Instruct tool-result/memory
semantics. Recognition of the same dedicated vocabulary tokens is not Chat
tool/memory competence.

## Original actual-model run

[GitHub Actions **40603226200**](https://github.com/techrote/neurounner/actions/runs/40603226200)
is the original actual-model source for this review:

- pure source/negative test job `122536255569`: **PASS**
- actual full Chat frozen-model job `122536314707`: **PASS**
- original reviewed branch head:
  `82c77846f3bbcb2cb98c7420be9cd14b2ab435f1`
- Actions merge checkout:
  `ef025d5e6899288005966a94551c6441211705bd`
- runner `GitHub Actions 1000045727`, PID 2752, 4 CPUs,
  Python 3.12.3, PyTorch 2.14.1+cpu, Transformers 4.57.6,
  peak RSS **3,293,208 KiB**
- **zero optimizer steps**, no backward pass and no held-out evaluation

### Right-padded masked KV gap

Two new synthetic ordinary-role prompts tokenize to **47** and **18** units.
Removing four true tail units leaves **43** and **14** valid prefill units.
The shorter request is padded to the shared 43-position cache width with
source pad ID 1 and a zero attention mask; four true incremental tokens are
then appended using per-request source positions.

With a manually activated **untrained** NCA readout:

| Check | Original max absolute error |
|---|---:|
| long Chat full vs right-gap cached tail | **2.8133392333984375e-05** |
| short Chat full vs right-gap cached tail | **2.7418136596679688e-05** |
| masked padding replaced by raw control ID 7 | **0, exact** |
| deliberately unmasked invalid padding | **7.743958473205566**, required difference |

Both genuine cache comparisons are below the unchanged #72
`5e-4` absolute **and** relative gates.

### Actual HF `generate()`: cache off vs cache on

The original frozen wrapper ran real Hugging Face `GenerationMixin.generate()`
for disabled, zero-readout and manually activated **untrained** NCA modes, each
with both `use_cache=false` and `use_cache=true`. The deterministic
four-unit cap produced `[32, 29, 7220, 277]` in all six cases.

Frozen-disabled versus zero-readout processed scores are **bitwise exact** in
both cache modes. Cache-on versus cache-off maximum finite-score error across
all 12 mode/step comparisons is
**2.002716064453125e-05**, below the fixed 5e-4 gate. Each processed score
contains 31,999 finite logits and the same one deliberately suppressed EOS
score; nonfinite masks are compared exactly before finite values.

Hook traces confirm three ordered depth taps for all four generation forwards:
the cached path uses the full prefill then three one-token decoder calls; the
no-cache path grows the source sequence each call. A→B→A request reset remains
exact.

The manual, untrained readout has a measurable max logit effect
`0.014346718788146973`; it is **not learning**. Frozen base tensor SHA
before and after is identical:
`232fdbdd370f72bdecf549952896970b75c56155b8c6b53387338ce38540813f`.

## Durable original evidence

[The durable evidence directory](../experiments/ncr-001/evidence-review/sprocket-chat-generate/)
preserves all five original Actions text members byte-for-byte inside a
deterministic USTAR/XZ archive:

- original model report: **13,417 bytes**, SHA-256
  `13a23dbe2ccb17bbbee0f168f1c518a85dc0b2b336a7700fa1320898e1b4149c`
- source prefetch report: 1,033 bytes, SHA-256
  `61d984b22326c1699f090f74a578c010c7839caef280c13f9e809d16c5200501`
- package inventory, source checkout and source-hash inventory also retained
- deterministic XZ SHA-256
  `bfbf783babfc12e12c09d86d2efa512c253c4b41f142b44e0ed5fdb445595ab3`
- original Actions artifact #12780115389, original ZIP SHA-256
  `b507c750a435880a290b4b3260a2de07cc20bf68f6ba17c68e1f502b89f7fff9`;
  the ZIP itself remains in Actions storage, while all five original member
  bytes are committed losslessly.

The independent
[`ncr_001e_sprocket_chat_generate_audit.py`](../scripts/ncr_001e_sprocket_chat_generate_audit.py)
rechecks original source/model/runner/ZIP identity, complete file hashes,
right-gap/cache results, six generation records, frozen-base hashes and
false-training/tool-semantic claims without executing the model.

### Preserved negative run

[Actions **40603167288**](https://github.com/techrote/neurounner/actions/runs/40603167288)
failed **before model execution** because one source-only unit test miscounted
the lengths of its own synthetic integer vectors by one. The runtime
`right_gap()` output was correct. Commit
`82c77846f3bbcb2cb98c7420be9cd14b2ab435f1` corrected only the expected
test arithmetic. The failure is preserved as evidence; no numerical model gate
was changed.

## Acceptance boundary

**PASS only the finite Sprocket Chat source/runtime right-gap and real HF
generate cache-on/cache-off pre-training subtest.**

This does not establish Chat tool/memory behavior, does not run gradients or an
optimizer, does not inspect sealed evaluation fixtures, and does not set the
four-model #72 gate. The protocol still has `train_authorized=false`.

A separate deterministic lane adjudication must combine this original run with
the already archived #95/#97/#100 source/cache/cold evidence before any
Sprocket Chat per-model status is changed.
