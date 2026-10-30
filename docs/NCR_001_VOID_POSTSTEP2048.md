# NCR-001E — one-step post-update void.1 native 2048 diagnostic

**DRAFT / separately authorized bounded diagnostic / no comparative training / no held-outs / no final #72 admission.**

Parent #67/#72, void source #79, BYTE-EOS #92, matched controls #73. This draft is stacked directly on the source-bound four-cohort reconciliation #216 at `d7739bcc8ada5663157ae45a20e07dd3cf144dc2`. It does not modify historical #171/#172/#173/#175/#210/#211 evidence or their interpretations.

## Purpose

Close the largest remaining joint void.1 qualification gap: previous physical evidence demonstrated **2048-context correctness before learning** and **short-context correctness after one public synthetic NCA-only update**, but never a single post-update adapter exercising the native 2048 context.

The new runner therefore:

1. loads exact `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563` Qwen3 and the independently reviewed ByteTokenizer source;
2. performs exactly the already-reviewed **one public synthetic NCA-only SGD diagnostic step** using true EOS ID 258 and 21 shifted supervised completion/EOS targets;
3. saves the temporary adapter only for the diagnostic;
4. starts a **second Python interpreter on the same Actions runner**, reloads the exact updated adapter and frozen Qwen3;
5. executes all 2,048 positions through genuine incremental KV-cache decoding and compares them to one full active 2,048-unit forward under the unchanged 5e-4 absolute/relative tolerance;
6. runs explicit **B=2** left- and right-padded 2,048-wide masked batches, comparing each valid row against an identical-shape B=1 call and against row-order-swapped B=2 execution;
7. mutates the final 64 source units and requires every earlier active-adapter logit to remain **bit exact**;
8. uses native `GenerationMixin.generate()` with a 2,044-unit prefix and exactly four generated units so the resulting sequence ends at the model's native 2,048-unit context limit, comparing cache-on/off generated IDs and score tensors;
9. verifies the frozen backbone tensor fingerprint after the long diagnostic;
10. deletes the temporary updated adapter **before** any Actions artifact upload.

## Locked source

- Qwen3 checkpoint: SHA-256 `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`, 360,622,744 bytes.
- ByteTokenizer code: SHA-256 `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`.
- EOS: integer **258**, pad 256.
- Model taps: post-block [7,15,23].
- Protocol Git blob `14ce5fbaa38f8806c8c8738f19c01b91385eebaf`; source-pins blob `59fff1aed872b5563aa6898171983973e0cdb02c`.
- Cached-logit gate: atol=rtol=5e-4.
- Existing one-step learning source, poststep cold-replay source and pretraining max-context source are required at their immutable Git blobs before execution.

The long input is a public synthetic multilingual UTF-8 string embedded directly in the source code; no sealed NCR fixture is read. The secondary masked-batch sequence is a different slice of that same public source.

## Scientific boundaries

This is **one additional diagnostic optimizer step**, not a comparative NCA/MLP experiment, learning campaign or held-out quality evaluation. It may support the source/runtime mechanics of #72, but it does not measure:

- NCA superiority over the matched gated MLP;
- multi-example or multi-seed learning;
- general language quality;
- reliable EOS stopping;
- the historical effect size of TRAIN-002b's literal-text EOS bug;
- independent fresh-host replication (producer and consumer are distinct Python interpreters on the same runner);
- actual matched training compute.

The frozen sibling model lanes are untouched. No SmolLM2 or Sprocket result transfers to void.1 or vice versa.

## Evidence policy

The physical workflow uploads all partial/success/failure source and host receipts, logs, runner identity and prefetch hashes with `if: always()`, but **never the temporary adapter weights**. A successful physical Actions artifact is temporary evidence only. A separate downstream custody PR must re-download and permanently commit the complete original source and host Actions ZIP containers and complete decoded job logs byte-for-byte, remove any write-enabled bootstrap after success, and install a read-only exact-head Git-blob/SHA/member auditor with adversarial mutations.

Only after permanent custody should a separate source-only adjudication PR update the #216 cross-cohort interpretation. Historical #171 statuses are not rewritten retroactively.

**Locked gates:** `train_authorized=false`; `void1_conformance_pass=false`; `full_72_conformance_pass=false`; `actual_compute_budget_qualified=false`.
