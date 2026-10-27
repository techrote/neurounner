# NCR-001E — independently audited source/publisher Sprocket Instruct wire receipts

**Status:** Source-scoped structural-serialization and frozen-cache **prepass**, not full independent #72 PASS. Original held-out fixtures unopened; `train_authorized=false`; optimizer steps **0**; all dependency PRs draft/unmerged.

## Actual original pinned host, 27 October 2026

- [GitHub Actions 40567832272](https://github.com/techrote/neurounner/actions/runs/40567832272) **SUCCESS**. Publisher-source identity/negative-test job `122413740921` passed five offline tests. Original root Instruct physical frozen-checkpoint job `122413848927` **SUCCESS**.
- Original Actions PR source head `1e398b8a48825b89b695c850b906a1a9cf574532`, checked-out Actions merge SHA `e2d4acf5b252c53f9723146ed17e550c12a3c7ee`. Original Ubuntu runner `GitHub Actions 1000045291`, Linux 6.17.0-1022-azure, Python 3.12.3, torch 2.14.1+cpu, Transformers 4.57.6, PID 2729, 4 vCPU; peak RSS **3,293,680 KiB**.
- Exact full **root Instruct** `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668` checkpoint: `model.safetensors`, **1,002,207,984 bytes**, SHA256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`; original hook source commit `b95c010f9b2ba3cbd9f81317981316bb9c8acd19`.
- Public [publisher SFT wire](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/src/train/sft_data.py) Git blob `90bddf8976b9d6410fe7c5c1f084708e1cebab73` and its placement validator `97811aa4245ad3f05f66ce0c2d8aa8c353619267`. The source-only CI confirms the first file raw SHA256 `e5568c15989f3987fec07545640f2f271ffe967c27cbe9d9883c0dd7b5c4ec69`. This is descriptive publisher-code provenance, **not** cryptographic historical HF training-build attestation.

| Frozen-model source-wire measurement | Original result |
|---|---:|
| 120-unit synthetic typed host/assistant sequence, assistant mask units | 46 |
| Frozen source vs zero-init NCA maximum absolute logit difference | **0, exact** |
| Bare assistant tool-call boundary: full vs genuine cached tail | **3.075599670410156e-05** |
| Host-injected tool-result boundary: full vs cached tail | **2.7418136596679688e-05** |
| Assistant memory-write boundary: full vs cached tail | **4.744529724121094e-05** |
| Unchanged per-check absolute / relative causal tolerance | 5e-4 / 5e-4 |
| Strict future-suffix drift; independent A→B→A request reset | **0; 0** |
| Manually activated, **untrained** NCA readout nonzero effect | 0.06854724884033203 |
| Actual optimizer updates, trained model parameters | **0; none** |

The publisher's source distinguishes `<|tool_call|>` assistant-emitted control ID **6**, `<|tool_result|>` host-input ID **7**, `<|memory_write|>` assistant-emitted ID **11**, and `<|memory_read|>` ID **10**, which serves **both** masked host memory framing and supervised assistant request depending on position. This test uses those original IDs and a synthetic multi-turn context, not fabricated bare token transport.

## Durable original byte custody

- [Original Actions artifact #12762719757](https://github.com/techrote/neurounner/actions/runs/40567832272/artifacts/12762719757), original ZIP **2,273 bytes**, SHA256 `83c3a83c88c20da99bb4b9131c2a67528689f6a6adc7cd27bb2e681f3f6fc82a`. Full ZIP is retained at GitHub Actions pending expiry, **not** in Git.
- The original source report `sprocket-instruct-publisher-wire.json` is **6,816 unchanged bytes**, SHA256 `58f88ec403f8471e3ccbd76a9e665ef7f3dc7080969c3330117134ac381a29a2`, losslessly committed as [gzip/base64 original JSON](../experiments/ncr-001/evidence-review/sprocket-publisher-wire/original-instruct.json.gz.b64). Gzip: **2,093 bytes**, SHA256 `a744a82ae5dcd93d5dba67cd9aef0df9c4726ad0fcfab4ca6af5882efa0d08ac`.
- [Provenance](../experiments/ncr-001/evidence-review/sprocket-publisher-wire/provenance.json) binds exact original model source, publisher source, runner PID/platform/versions, JSON/ZIP hashes, fixed tolerance and all false training/semantic claims. The committed base64 was independently read back from GitHub and matched the original 2,792-character local archive fingerprint, then independent hosted SHA256 replay [40569426693](https://github.com/techrote/neurounner/actions/runs/40569426693) **SUCCESS**.
- [Independent standard-library auditor](../scripts/ncr_001e_sprocket_publisher_receipt_audit.py) and [eight positive/negative test groups](../tests/test_ncr_001e_sprocket_publisher_receipt_audit.py) refuse modified raw bytes, source-model swapping, altered publisher SHA, relaxed cache tolerance, fabricated tool/memory behaviour, disabled causal tests, false optimizer/heldout claims and reversed host-vs-model memory masks.
- **Preserved genuine archival failure:** first [readback run 40569318251](https://github.com/techrote/neurounner/actions/runs/40569318251) **FAILED** because the auditor's direct semantic validation did not catch a deliberately mutated *provenance* original-receipt SHA field; the separate archival bytes loader already correctly validated original JSON. Commit `c8fbf4f61b56fea4564c98d182611091edf2ed3b` extended direct receipt validation to bind both original JSON and gzip digest; follow-up [40569426693](https://github.com/techrote/neurounner/actions/runs/40569426693) passed. This is an archival mutation-test gap repaired in code, **not** a model/causal failure. Do not delete the earlier CI negative receipt.

## Scope and unresolved acceptance

All tool-call, tool-result, memory-write and memory-read strings are *synthetic* in source-permitted positions. The source shows what the frozen model was taught to receive/emit; a cached forward says **nothing** about whether it can choose a tool correctly, actually consume the tool result, remember a fact or produce an appropriately bounded action. The publisher reports an important *negative* memory-learning outcome. Do not equate this new structural wire proof with learned memory competence.

The Chat subfolder is not tested on this tool/memory variant and must not inherit these results. The full source-linked model training pipeline remains publicly described but is not attested byte-for-byte against the pinned checkpoint. #72 still needs independent final four-lane source/runtime/resource/frozen-optimizer review, and #73 matched MLP acceptance is separate. No actual execution of a tool or sealed behaviour fixtures occurred; no license to merge, train, tune or change `train_authorized=false`.
