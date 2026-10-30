# NCR-001E — Four-model frozen-host prepass evidence readback

Issue [#72](https://github.com/techrote/neurounner/issues/72); parent [#67](https://github.com/techrote/neurounner/issues/67). This branch is a **separate source-receipt audit**, not a reproduction of the underlying physical model runs and **not** the independent final source/runtime qualification required by #72.

It branches from the locked protocol PR #77 and contains **byte-for-byte copies of the original JSON receipts and provenance** from four separate draft integration branches:

| Lane | Source PR | Original successful CI | Expected pinned checkpoint | Receipt |
|---|---|---|---|---|
| SmolLM2 135M | [#83](https://github.com/techrote/neurounner/pull/83) | [40422167066](https://github.com/techrote/neurounner/actions/runs/40422167066) | SmolLM2 revision 12fd25f… | [receipt](../experiments/ncr-001/evidence-review/smollm2/receipt.json) |
| Sprocket Chat | [#86](https://github.com/techrote/neurounner/pull/86) | [40426220940](https://github.com/techrote/neurounner/actions/runs/40426220940) | kandivault root revision eb157d… plus **chat/** model | [receipt](../experiments/ncr-001/evidence-review/sprocket_chat/receipt.json) |
| Sprocket Instruct | [#84](https://github.com/techrote/neurounner/pull/84) | [40423957135](https://github.com/techrote/neurounner/actions/runs/40423957135) | same repo revision, **root instruct** model | [receipt](../experiments/ncr-001/evidence-review/sprocket_instruct/receipt.json) |
| void.1 Qwen3 | [#85](https://github.com/techrote/neurounner/pull/85) | [40424675239](https://github.com/techrote/neurounner/actions/runs/40424675239) | dotlabs/void.1 revision 9c2be5… | [receipt](../experiments/ncr-001/evidence-review/void1/receipt.json) |

## Scope: independently check receipts, not reconstruct successful runs

The [standard-library readback](../scripts/ncr_001e_readback.py) validates each copied original receipt SHA-256 and byte size against its original Actions evidence manifest, validates model/variant/revision/tap/count/weight path against the predeclared protocol and hashed source inventory, and rejects confusing Sprocket Chat with its equally shaped but **distinct** Instruct checkpoint.

It checks that the source run downloaded actual full weight bytes matching the pinned digest; disabled/zero-init logits were exact; an intentionally **manually activated but untrained** NCA path changed logits yet stayed within predeclared full-vs-cached tolerance; future-suffix changes could not affect earlier logits; frozen-backbone gradients remained absent; and optimizer-step count was zero.

The [negative tests](../tests/test_ncr_001e_readback.py) independently mutate receipts, adjust custody hashes to look internally self-consistent, and establish that the auditor still rejects changed Chat weight identity, cache drift, causal future leakage, false training success, unexpectedly executed void tokenizer code and an illicit training flag.

**Pass classification is only source-integrity readback.** It is explicitly **not** independent full-model conformance, not a second physical forward/generate experiment, not a scientific result about learned NCA, and not a GO for comparative training. A clean green CI does not change the protocol flag.

## Missing separate qualification

- Fresh reviewer-owned actual-model invocation for all four checkpoint variants, with longer sequences, diverse causal masks and incrementally cached decode, cold adapter checkpoint reload, batched requests and explicit concurrency/race testing.
- Sprocket Chat role-template semantics and Sprocket Instruct tool/memory/control-token behavior must be source-bound.
- void.1 custom ByteTokenizer Python review, then UTF-8/Unicode/BOS/EOS/byte-unit roundtrip and actual tokenizer-generated text tests. Its earlier integration purposefully did **not** execute downloaded code.
- A correctly integrated conventional MLP control using identical model hooks, inputs and optimizer/token budget, then **separately authorised** finite experiments after independent acceptance. No trained adapters exist.

Each lane will remain marked as **PREPASS_SOURCE_READBACK_PASS_NOT_FINAL** or **BLOCKED_FINAL_QUALIFICATION** until these tests are performed. Any failed/cancelled or mismatched run must remain in custody, and a positive prepass must never be relabelled as a completed final gate.

## Reproduce readback without PyTorch or network

    python3 -m unittest discover -s tests -p "test_ncr_001e_readback.py" -v
    python3 scripts/ncr_001e_readback.py --out runtime/ncr-001/four-model-prepass-readback.json

The [CI workflow](../.github/workflows/ncr-001e-readback.yml) runs on ubuntu-24.04 in minutes, without network downloads apart from pinned checkout action, and uploads a *transient* JSON reviewer summary. Original four receipts/provenance are retained durably in Git.
