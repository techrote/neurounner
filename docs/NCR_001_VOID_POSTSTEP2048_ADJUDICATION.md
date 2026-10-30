# NCR-001E — independent source-bound adjudication after the post-update 2048 cohort

**SOURCE/RECEIPT SCIENTIFIC INTERPRETATION ONLY / NO MODEL / NO TRAIN / NO FINAL #72 ADMISSION.**

Parent #67/#72; void integration #79; BYTE-EOS #92; matched-compute #73. This draft is stacked on the verified permanent original custody PR #218 exact head `058ed306e9ea13395c08b8513265363da179fa4e`. It does not alter any historical original physical ZIP/log/receipt, protocol, source pin, fixture or the merged historical #171 decision.

## What is independently re-audited

The adjudicator re-executes the complete source-only reconstruction behind #216, which itself re-audits four distinct original frozen-Qwen3 physical cohorts:

- A — post-lease short/cache/cold, run 40747035876;
- B — pretraining 512/1024/2048 cache/suffix/generate, run 40755894558;
- C — original one-public-example NCA-only gradient/update diagnostic, run 40765025332;
- D — separate 31-unit post-update second-Python cold/final-four-KV diagnostic, run 41058648266.

It then independently runs #218's permanent raw-byte custodian for the new **E — post-update 2048** cohort, original physical run 41133450850, exact source head `520c4ae5b3152aff8d94760dab2e3103746f7334`, including both original Actions ZIP containers, both complete decoded job logs and every original ZIP member.

Historical #171 statuses are retained verbatim; later evidence does not retroactively rewrite those archival statuses.

## Cohort E: exact bounded physical result

The new original cohort is one exact pinned `dotlabs/void.1` Qwen3 physical execution:

- one public synthetic NCA-only diagnostic SGD step;
- 31 training input units, 21 shifted supervised completion/EOS targets;
- true terminal EOS ID 258;
- frozen-backbone SHA-256 fingerprint unchanged;
- second Python interpreter on the **same Actions runner** reloads the temporary updated adapter;
- **2,048/2,048 genuine one-token incremental positions** compared with the full updated-adapter 2,048-unit forward;
- maximum absolute all-position cache/logit difference **7.522106170654297e-05**, below unchanged 5e-4 absolute/relative protocol tolerance;
- explicit width-2,048 **B=2** left and right masked full forwards with 1,536 valid units on the short row and 512 pad units:
  - batch-vs-identical-single maximum abs **0.0**;
  - row-order-swap maximum abs **0.0**;
- final **64 source units** mutated: every earlier logit bit exact, maximum abs **0.0**;
- native `GenerationMixin` from a 2,044-unit prefix for exactly four generated units:
  - cache on/off IDs identical;
  - generated IDs **[105, 110, 115, 112]**;
  - cache-on/off score maximum abs **5.7220458984375e-06**;
  - real EOS 258 **not emitted**;
- effective Torch threads 4; consumer elapsed time 289.62789 s; peak RSS 1,610,688 KiB;
- temporary adapter deleted before original Actions artifact upload.

The original source and physical ZIPs/logs are under permanent Git custody in `experiments/ncr-001/controls-evidence/void-poststep2048-20261030/`, and exact-head read-only #218 CI re-downloaded all four complete original streams byte-identically.

## Nine required #72 checks after Cohort E

The table below is a **current source-bound observation map**, not a retroactive edit of #171 and not an independent final scientific PASS.

| Required check | Historical #171 status | Current bounded source-bound evidence |
|---|---|---|
| `same_base_logits` | PASS | A/B frozen/zero evidence plus C/E exact initial zero-readout mechanics |
| `base_weight_unchanged_after_step` | BLOCKED_BY_TRAINING_POLICY | C/D/E each source-bound one-public-example update mechanics preserve frozen Qwen3 |
| `causal_suffix_invariance` | PASS | A/B pre-step plus **E post-step 2048 strict suffix** |
| `full_prefix_vs_incremental_cache` | PASS | A/B pre-step, D four-tail post-step, **E all 2,048 post-step B=1 positions** |
| `per_token_isolation` | PASS_SOURCE_RUNTIME | A/B source/runtime; E cached B=1 state checkpoints and explicit B=2 left/right masked full-forward batch/single + row-order isolation |
| `per_request_reset` | PASS | A pre-step and D post-step A→B→A physical evidence; E does not replace/relabel D |
| `nonzero_adapter_gradient_after_warmup` | BLOCKED_BY_TRAINING_POLICY | C/D/E public one-step source-bound postwarmup gradient observations |
| `finite_state_and_loss` | PARTIAL_BLOCKED_WARMUP_LOSS | C/D/E finite public one-step loss/state plus E finite native 2048 active path |
| `cold_reload_state_dict` | PASS_PRETRAIN | A pre-step NCA/MLP, D short post-step NCA and E **post-step NCA second-interpreter native-2048** |

The strongest statement supported is:

> **All nine protocol check names now have bounded source-bound physical observations somewhere in the exact void.1 evidence graph, and Cohort E closes the previously largest joint post-update/native-2048 mechanical gap.**

That statement is deliberately weaker than `void1_conformance_pass=true`.

## Precise unresolved scientific boundaries

### 1. B=2 masked cached decoding is still not directly measured

Cohort E tests:
- B=1 all-position incremental KV at 2048;
- B=2 explicit left/right masked **full-forward** isolation.

It does **not** perform a B=2 left/right padded incremental-KV replay. Because #72's expanded acceptance language refers to cache correctness across context lengths, batch sizes and masks, the adjudicator records this explicitly instead of transferring B=1 cache success into B=2 cached success.

### 2. Same-runner second Python is not independent-host replication

The post-update adapter is genuinely reloaded in a new interpreter, but the producer and consumer run on the same Actions runner. A fresh independent host reproduction would strengthen robustness; it is not silently claimed here.

### 3. Learning effectiveness is outside this diagnostic

One public synthetic update proves bounded gradient/update/freeze mechanics, not learned usefulness. Multi-sample/multi-seed quality, generalization and matched NCA-vs-MLP effect remain future #81/#73 obligations. These are not retroactively inserted into the original nine mechanical checks.

### 4. EOS behavior remains negative/unknown

The new 2044+4 generated IDs do not contain EOS 258. Earlier 12 short generation records also did not emit EOS. Corrected target serialization is source/runtime qualified, but useful learned stopping is not. Historical TRAIN-002b outcome impact remains **UNMEASURED**.

### 5. Explicit final scientific admission remains separate

This same-account source-only adjudication can verify original evidence and identify what it supports. It does not invent a distinct independent human reviewer, does not call COMMENTED reviews GitHub APPROVED and does not mutate the repository-wide conformance/training registry.

## Fail-closed implementation

`scripts/ncr_001_void_poststep2048_adjudication.py`:

1. pins the actual committed protocol, source pins, #216 machine decision and #218 original manifest by Git tree mode/blob;
2. regenerates #216 exactly from original A/B/C/D evidence;
3. reruns #218's raw-byte custody auditor for E;
4. directly reads E's original producer, consumer and learning receipts from the permanently stored original physical Actions ZIP;
5. rejects shortened/relabelled cache coverage, bool-as-int optimizer counts, changed model/EOS/tolerances, false scientific promotion, altered frozen source, missing masked/suffix/generate evidence or rewritten #171 statuses;
6. emits a new machine-readable source-only decision without changing any historical source.

A read-only exact-head workflow must pass. A CI green result is not a fresh physical model execution.

**Locked:** `train_authorized=false`; `void1_conformance_pass=false`; `full_72_conformance_pass=false`; `actual_compute_budget_qualified=false`.
