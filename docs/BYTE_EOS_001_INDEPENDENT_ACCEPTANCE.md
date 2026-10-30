# BYTE-EOS-001 — independent future serializer acceptance

Issue [#92](https://github.com/techrote/neurounner/issues/92), candidate
[draft PR #94](https://github.com/techrote/neurounner/pull/94), independent
review [draft PR #126](https://github.com/techrote/neurounner/pull/126).

## Decision

**PASS the reviewed future serializer source. Do not integrate it into training.**

Machine status:

`PASS_INDEPENDENT_FUTURE_EOS_SERIALIZER__VOID_TRAINING_INTEGRATION_STILL_BLOCKED`

The independent reviewer fixes PR #94 at
`b1c558f7d46f5d7c76b4e6336a37143886d16c08`, serializer Git blob
`724a5040d2fc9eb06ef0b17143d44daa36a4aa89`, and exact
`dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`
ByteTokenizer source SHA-256
`279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`.

The 1,024-unit bound is taken only from exact preregistration commit
`a1b41c358d215974ea9ef2748baa13c3872b0fbb`, not inferred from the older
stacked branch protocol.

## Original independent runtime

[Actions 40604675320](https://github.com/techrote/neurounner/actions/runs/40604675320)
passed both source-only and actual **tokenizer-only** jobs. No model weight was
downloaded.

The exact reviewed ByteTokenizer runtime at Transformers 4.57.6 confirmed:

- PAD=256, BOS=257, **EOS=258**, vocabulary 259;
- no automatic BOS/EOS insertion;
- literal `<eos>` encodes to printable byte IDs
  **[60,101,111,115,62]**, not EOS 258.

Seven new, non-heldout cases exactly matched a separately implemented UTF-8
byte oracle:

| case | sequence units | supervised units |
|---|---:|---:|
| ASCII | 22 | 16 |
| multilingual Unicode | 28 | 16 |
| literal `<eos>` content | 18 | 6 |
| empty completion | 7 | 1 |
| empty prompt | 16 | 16 |
| emoji boundary | 49 | 21 |
| exact 1,024-unit boundary | 1,024 | 1,024 |

Every target is exactly
`UTF8(prompt + completion) + [258]`; prompt labels are `-100`, completion
byte IDs and terminal EOS 258 are supervised. The 1,025-unit case is refused
rather than truncating EOS. Lone-surrogate Unicode is refused rather than
replacement-encoded.

Historical identity remains explicit: old literal suffix
`[60,101,111,115,62]`, corrected suffix `[258]`, four units saved.
**Historical TRAIN-002b generation impact remains UNMEASURED.**

## Durable original review evidence

The complete original independent-review Actions artifact is preserved
byte-for-byte as
[`original-actions-a1.zip.b64`](../experiments/ncr-001/byte-eos-001/independent-review/original-actions-a1.zip.b64).

- artifact #12779882453
- ZIP bytes: 2,396
- ZIP SHA-256:
  `9cde28bc5a927c262a0eda907f445491f4937c0ed7283a1ea535d2b5b323b1bf`
- original review JSON: 4,083 bytes, SHA-256
  `0400daf39c36e64707525856850e89720fd83742514520a9a27a019d1feae527`
- original source checkout and complete source-hash manifest preserved in the ZIP
- [source/runner/provenance](../experiments/ncr-001/byte-eos-001/independent-review/provenance.json)

The original candidate BYTE-EOS receipt remains unchanged:
`6c611bdc0f021f6d9e2fa39d91b7e141d937844f2b849f9ae437151254572922`.

## Independent archive audit history

The standard-library archive auditor rechecks the complete ZIP, member SHA,
candidate source bytes/Git blob, original BYTE-EOS receipt, source-pins bytes,
exact preregistration bytes, all oracle cases and all false training/history
claims.

- **40605056186 FAILED**: the positive original ZIP audit passed, but a
  deliberate mutation of the provenance preregistration **commit field** was
  not rejected. The auditor had bound the preregistration file SHA but omitted
  the redundant commit-field equality.
- commit `05f88eafbf8c04698e85a2689b058369ed9fee5a` added the missing fail-closed
  commit identity check.
- **40605116858 PASS**: exact preregistration commit and protocol SHA both
  fail closed; archive/tamper/false-integration tests pass. This is archive
  readback only, not another tokenizer or model execution.

## Boundary

This review accepts the **future serializer source only**.

It does **not**:

- modify or reinterpret historical TRAIN-002b;
- claim the serialization bug caused historical repetition/budget exhaustion;
- integrate the serializer into the void NCR training path;
- download or execute void model weights;
- authorize an optimizer/backward step;
- open sealed heldouts;
- satisfy final void #72;
- authorize NCA/MLP comparison.

Integration remains conditional on independent void #72 acceptance **and** an
explicit future training authorization. `train_authorized=false` remains.
