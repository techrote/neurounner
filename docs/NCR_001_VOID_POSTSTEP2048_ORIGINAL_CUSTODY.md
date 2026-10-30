# NCR-001E — permanent original custody for the post-update void.1 2048 diagnostic

**EVIDENCE CUSTODY ONLY / NO MODEL / NO TRAIN / NOT SCIENTIFIC #72 ADMISSION.**

This branch is stacked directly on the original physical [PR #217](https://github.com/techrote/neurounner/pull/217) source head `520c4ae5b3152aff8d94760dab2e3103746f7334`, original [Actions run 41133450850](https://github.com/techrote/neurounner/actions/runs/41133450850). It preserves the exact original Actions artifact containers and complete decoded source/physical job logs from that one physical execution.

## Original source/run identity

- Repository: `techrote/neurounner`
- Exact physical source head: `520c4ae5b3152aff8d94760dab2e3103746f7334`
- Original Actions run: **41133450850**, attempt 1, pull_request event, SUCCESS
- Source job: **124329952468**, SUCCESS
- Physical frozen-Qwen3 job: **124330038397**, SUCCESS
- Frozen model: `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`
- Checkpoint SHA-256: `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`
- Reviewed ByteTokenizer source SHA-256: `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`
- Frozen-backbone tensor fingerprint: `7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904`
- True EOS ID 258, pad ID 256; effective Torch threads 4 after model load.

## Complete original byte streams under Git custody

| Original stream | Bytes | SHA-256 | Git blob |
|---|---:|---|---|
| Source Actions ZIP | 558 | `024ed9a0ac0d89338ef249ba79b3ebb4a20a7659c714fe2e557dd055379cc806` | `1bbace701bea59969854cbdf50fcb0444d2aba4c` |
| Physical host Actions ZIP | 10,002 | `db66de004d26a6c24539b08898f85df88b3be7631ba300655a82bd021de71145` | `e164d792be74857d6f7a2384cc316bd7631162a0` |
| Source complete decoded job log | 52,537 | `c9ebe268baca70c7f3a79eff29f74a07943dbd096bf50e97d07b4cb3814bd8ea` | `55faf7ca98b0529f4f713087bc0dcaa91bbd2d1f` |
| Physical complete decoded job log | 82,834 | `5cff4684863fc505a752d779b60a884e91327627b99dcea0e65da50d4f263d75` | `a062dbfed90d643cca0162ebe446daf64e6ee5f2` |

The immutable manifest is `experiments/ncr-001/controls-evidence/void-poststep2048-20261030/manifest.json`, Git blob `462a43cf3e56b5edefe9dcf14ffc6dae199a3158`. The source ZIP has its complete 10-test log. The physical ZIP has all eight original members: producer/consumer receipts and logs, exact prefetch hashes, original runner identity, one-step diagnostic receipt and source-loader diagnostic receipt. The temporary post-update adapter/checkpoint is **not present** in the Git archive.

The one-shot write-enabled bootstrap used the corrected redirect-safe GitHub/Azure Actions artifact downloader, successfully archived the originals in [run 41134433036](https://github.com/techrote/neurounner/actions/runs/41134433036), then its write-enabled workflow was deleted. The acquisition script remains only as readback/provenance code; subsequent verification is read-only.

## Original physical observation

Exactly one public synthetic NCA-only SGD diagnostic update occurred. That learning sample was **31 tokenizer units**, with **21 shifted supervised completion/EOS targets** and integer EOS **258**. It is not a corpus train, matched MLP update or learned-quality study.

After saving that temporary NCA state, a **new Python interpreter on the same original Actions runner** reloaded the exact frozen Qwen3 and updated NCA adapter. Original physical results preserved in the manifest:

- **2048/2048 genuine incremental KV positions** compared against the active full 2048-unit forward.
- Maximum absolute all-position cache/logit deviation: **7.522106170654297e-05**, inside unchanged protocol atol/rtol `5e-4`.
- Explicit **B=2** width-2048 masked batching:
  - left-padding valid-row batch vs identical B=1 maximum abs: **0.0**;
  - left row-order swap maximum abs: **0.0**;
  - right-padding batch vs identical B=1 maximum abs: **0.0**;
  - right row-order swap maximum abs: **0.0**.
- Final **64 source units** mutated: all earlier logits bit-exact; maximum abs **0.0**.
- Native `GenerationMixin`: 2,044-unit prefix + exactly 4 generated units, cache on/off generated IDs equal and score tensors within tolerance.
- Generated IDs: **[105, 110, 115, 112]**; real EOS 258 **not emitted**.
- Generation cache-on/off score maximum abs: **5.7220458984375e-06**.
- Consumer second-interpreter runtime: **289.62789 s**, peak RSS **1,610,688 KiB**.
- Frozen-backbone fingerprint unchanged after the complete long-context diagnostic.

The result closes the previously unmeasured *physical* combination of one updated NCA adapter with the native 2048 context, full all-position incremental KV, explicit masked B=2 forward isolation, future-suffix invariance and native generation cache parity. It does **not** itself authorize a scientific per-model PASS.

## Primary read-only auditor

`scripts/ncr_001_void_poststep2048_original_custody.py` checks:

1. literal committed regular-file Git tree modes and object SHA-1 for the manifest and every original ZIP/log;
2. exact whole-file byte counts and SHA-256 values;
3. exact ZIP CRC and member inventory/digests;
4. source run/head/job/artifact identity;
5. exact model/tokenizer/frozen-backbone source;
6. exact one-step public sample/EOS mechanics;
7. all-position 2048 cache and unchanged 5e-4 tolerance;
8. both masked batch directions;
9. strict future-suffix result;
10. native GenerationMixin cache parity and negative EOS observation;
11. no temporary adapter/checkpoint in evidence;
12. all training/science/compute/heldout gates remain false.

Adversarial tests mutate each original complete stream, delete streams, introduce symlinks, forge manifest/science fields, relabel a 31/tail cache as 2048, change mask/suffix/generation results, substitute variants and exploit Python bool-as-int semantics. A passing source-only auditor is **not** a fresh physical model run.

The final read-only workflow may additionally re-download the same two original Actions ZIPs and both complete decoded logs and require byte-for-byte equality with Git custody while the temporary Actions artifacts remain available.

## Scientific boundary

This custody branch makes no independent scientific judgment beyond preserving and checking the original observation. It does not rewrite historical #171 statuses, measure NCA-vs-MLP quality, open sealed heldouts, establish general EOS stopping, measure historical TRAIN-002b EOS impact, or establish independent fresh-host replication.

**Locked:** `train_authorized=false`; `void1_conformance_pass=false`; `full_72_conformance_pass=false`; `actual_compute_budget_qualified=false`. The next phase is a separate source-only independent adjudication stacked on this accepted custody head.
