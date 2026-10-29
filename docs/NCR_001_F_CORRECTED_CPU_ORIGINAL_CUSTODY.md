# NCR-001F — corrected adapter-only CPU original Actions custody

**Issue:** [#73](https://github.com/techrote/neurounner/issues/73). **Descendant source:** draft [#98](https://github.com/techrote/neurounner/pull/98) at immutable head `c8139cc36b1105f6bcafd585b766ff470e8c9e1d`. This is a source-only evidence-retention descendant; it does not modify the original run, the pinned protocol, either Sprocket checkpoint, the NCA/MLP implementation, training gates or held-out fixtures.

## Why these are distinct original bytes

The prior source-only CPU run [#40867677298](https://github.com/techrote/neurounner/actions/runs/40867677298) included guardrail validation inside its timed interval. That original ZIP/log is already preserved on upstream #98, including the NumPy warning and the failed auditor replay [#40869267829](https://github.com/techrote/neurounner/actions/runs/40869267829). **Do not overwrite/reinterpret that run.**

The corrected-source run [#40870764754](https://github.com/techrote/neurounner/actions/runs/40870764754), job `123436909436`, executed the actual PyTorch NCA and gated-MLP adapter classes with unchanged zero-readout/finite checks during warmup and after timing, but excluded the validation scans from the timing samples. It **did not load either frozen Sprocket model**. It ran on AMD EPYC 7763 / Ubuntu 24.04, Python 3.12.3, PyTorch `2.14.1+cpu`, four threads, fixed seed 20261029. The original nine tests were successful. The NumPy initialization warning remains visible and classified as an original warning rather than a model failure.

| Synthetic sequence units | NCA 3-tap median ms | MLP 3-tap median ms | Observed NCA / MLP ratio |
|---|---:|---:|---:|
| 1 | 1.323654 | 0.459195 | 2.882553 |
| 16 | 4.095753 | 1.010530 | 4.053074 |
| 64 | 6.803301 | 2.530500 | 2.688520 |

These results measure synthetic adapter-only CPU invocation with Python/module-step overhead. They do **not** establish actual frozen-model latency, adapter incremental RSS, matched-host compute/training-budget parity or useful NCA quality. Shared geometry measured once does not represent equivalent Chat/root-Instruct checkpoint behavior. The gated MLP and NCA remain different operations, not comparable trained cohorts.

## Immutable original custody inventory

- Original **whole Actions ZIP** artifact `12896225909`, 5,696 bytes, SHA-256 `fe1b04341ee5792ab316f7199fa2052163fb3fdc11241c5fe2e046012404f8c8`, Git blob `c5f449fcf05df8c576ef10d25f1bada44655f2a8`. It contains exactly seven unmodified GitHub-original members: `report.json`, `tests.log`, `runner.log`, `pip-freeze.txt`, `lscpu.txt`, `source-head.txt`, `uname.txt`. All seven lengths and SHA-256s are individually pinned in `original-corrected-cpu-provenance.json`.
- Original **complete GitHub-decoded job log** `123436909436` at Git blob `669c7c73322673aef51c6929cf3ac961150a7d7c`. This is a decoded GitHub log representation, not the compressed redirect transfer's original bytes.
- Provenance ledger Git blob `900b4053851d354862998d2dc73f0b92040cd838`. Distinct source paths and weight digests are declared for `chat/model.safetensors` and root `model.safetensors`; neither checkpoint was downloaded.
- Original artifact expiry `2026-11-28T12:08:29Z`. This draft preserves the source bytes in Git before expiry; archival durability still depends on retaining the Git branch/objects and satisfying review/merge policy.

The [temporary bootstrap workflow](../../.github/workflows/ncr001f-corrected-cpu-archival-bootstrap.yml) was used once to byte-download and write **only the original ZIP and decoded log** to this descendant branch. The final workflow must be read-only, assert the exact Git blobs/protocol pins and re-download/byte-compare the original archive/log while they remain available. The verifier and adversarial tests are stdlib-only and never download models.

## Explicit non-admission and further work

Historical six-token incomplete root-Instruct generated tool-call prefix remains negative evidence; publisher wire transport is not generated tool/memory competence. void.1 real EOS ID 258 remains distinct and historical TRAIN-002b impact **UNMEASURED**. SmolLM2, Chat, root Instruct and void.1 each retain independent #72 scientific gates.

`train_authorized=false`; `full_72_conformance_pass=false`; `actual_matched_frozen_host_budget_pass=false`; no optimizer steps, held-out access, comparative NCA/MLP learning or draft merges.

**Next scientific action:** obtain independent, source-matched full-host Chat/root-Instruct masked 2048-context true incremental cache and separate-process cold adapter evidence, then measured full-host matched compute budget for #73. The current archive neither substitutes for that work nor changes scientific admission.
