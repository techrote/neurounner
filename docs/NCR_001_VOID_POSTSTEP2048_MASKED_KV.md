# NCR-001E — post-update void.1 B=2 explicit-mask incremental-KV follow-up

**DRAFT / bounded physical diagnostic / NO comparative training / NO heldouts / NO final #72 admission.**

Parent #67/#72, void integration #79, BYTE-EOS #92. This branch is stacked directly on #219 exact source/adjudication head `ce4280f6f1cf2dbfb43eec973efa94bafde25496`. It does not modify any historical physical evidence, protocol, source pin, fixture, threshold or scientific registry.

## Purpose

#219 deliberately left one concrete mechanical cache/mask gap open:

> post-update **B=2 explicit masked incremental-KV** replay had not been physically measured.

Cohort E (#217) already demonstrated:
- one exact public synthetic NCA-only update;
- second-interpreter reload;
- all 2,048 B=1 incremental positions;
- B=2 left/right masked **full-forward** isolation;
- 64-unit future-suffix exactness;
- native 2,044+4 generation cache parity.

This follow-up tests only the missing masked cached-decoding composition.

## Physical design

The producer reuses the exact reviewed #217 one-step public diagnostic:
- 31 input units;
- 21 shifted supervised completion/EOS targets;
- integer EOS 258;
- one NCA-only SGD step;
- frozen Qwen3 unchanged;
- no MLP optimizer and no sealed fixture.

The consumer is a **new Python interpreter on the same Actions runner**. It verifies the producer receipt and temporary adapter hashes, reloads exact pinned `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`, then runs two synthetic B=2 masked cache layouts at width 2,048:

### Left padding

- row 0: 2,048 real byte-tokenizer units;
- row 1: 512 synthetic PAD=256 positions, 1,472 real prefix units, then **64 real continuation units**;
- explicit attention mask;
- explicit position IDs from `clamp(cumsum(mask)-1, min=0)`.

### Right-gap sentinel

- row 0: 2,048 real units;
- row 1: 1,472 real prefix units, **512 masked PAD=256 positions**, then 64 real continuation units.

This second form is intentionally a **synthetic mask-gap sentinel**, not a claim that a normal autoregressive publisher sequence contains a padded interior gap. It is useful precisely because the genuine incremental continuation must ignore cached masked keys while maintaining the real short-row position sequence.

For each side, the consumer:
1. computes a B=2 full forward and retains the final 64 real-token logits;
2. performs a genuine B=2 cache prefill through the first 1,984 batch positions;
3. executes **64 one-token B=2 incremental cache forwards** with explicit attention masks and position IDs;
4. compares B=2 full-vs-cached tail logits under the unchanged 5e-4 tolerance;
5. independently computes B=1 full+cached references for the long and short source rows;
6. compares both B=2 cached rows to their matching B=1 cached rows;
7. swaps the two batch rows, replays B=2 cached decoding, swaps outputs back and checks row-order invariance;
8. checks NCA per-forward state traces at incremental steps 0, 31 and 63 are finite with shape `[2,1,4,8,8]`;
9. verifies the frozen Qwen3 tensor fingerprint remains unchanged.

The temporary updated adapter is deleted before any Actions artifact upload.

## Frozen source authority

- Qwen3 checkpoint SHA-256: `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`;
- reviewed ByteTokenizer source SHA-256: `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`;
- EOS 258, PAD 256;
- post-block taps [7,15,23];
- protocol Git blob `14ce5fbaa38f8806c8c8738f19c01b91385eebaf`;
- source-pins Git blob `59fff1aed872b5563aa6898171983973e0cdb02c`;
- full-vs-cached atol=rtol=`5e-4`;
- producer source is the exact #217 poststep2048 implementation blob already reviewed by #219.

The long/short byte sequences are slices of the same public synthetic multilingual source embedded in the earlier #217 source. No preregistered or sealed NCR evaluation fixture is read.

## Evidence discipline

The source-only job must pass before any model execution. The physical job must preserve:
- exact model/tokenizer/pip/runtime/runner identity;
- producer and consumer JSON/logs;
- original public one-step learning receipt;
- source prefetch hashes and runner CPU inventory;
- failures and partial receipts.

The temporary updated adapter itself must never be uploaded or committed.

If the physical result is meaningful, it must be followed by:
1. a **separate evidence-custody PR** committing the complete original Actions source/host ZIP containers and both complete decoded job logs byte-for-byte;
2. deletion of any one-shot write-enabled archive workflow;
3. a final read-only literal Git-object/member/live-redownload audit with adversarial mutation tests;
4. a **separate source-only adjudication PR** updating #219's scientific boundary without rewriting historical #171 states.

A green physical or custody run is not automatically a final scientific #72 PASS.

## Boundaries

This probe does **not** measure:
- useful learned quality;
- multi-example/multi-seed effects;
- matched NCA-vs-MLP quality or actual training compute;
- reliable EOS stopping;
- independent fresh-host replay;
- any other frozen model lane.

It is an incremental mask/cache mechanics diagnostic only.

**Locked:** `train_authorized=false`; `void1_conformance_pass=false`; `full_72_conformance_pass=false`; `actual_compute_budget_qualified=false`.
