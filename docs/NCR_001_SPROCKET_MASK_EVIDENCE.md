# NCR-001E/F — independently archived original Sprocket explicit-mask witnesses

Parent epic [#67](https://github.com/techrote/neurounner/issues/67), source/cache admission [#72](https://github.com/techrote/neurounner/issues/72), matched-control integration [#73](https://github.com/techrote/neurounner/issues/73). **Readback-only**, stacked on [draft PR #157](https://github.com/techrote/neurounner/pull/157) at its original exact tested head `8275063544b90c83d2740f0d356288460b58ef6b`, above draft #156/#155/#154. No model download, original-source re-execution, optimization, evaluation, or changes to original evidence in this archival PR.

## Original source-bound physical execution

All three jobs in [run 40698955980](https://github.com/techrote/neurounner/actions/runs/40698955980) **passed** on 28 October 2026, attempt 1, original full frozen sources: six new offline source/attention-mask tests, ten existing cross-decoder lease tests, independently executed original **Sprocket Chat** and root **Sprocket Instruct** CPU jobs. PyTorch `2.14.1+cpu`, Transformers `4.57.6`, pinned original HF revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`, four CPU threads per job, 0 optimizer steps.

| Source-matched original physical observation | Chat (subfolder) | Instruct (root) |
| --- | --- | --- |
| Exact full safetensors SHA-256 | `551c6e26...cefe97` | `1a9fd106...20d864` |
| Original frozen base fingerprint | `828166c9...0e985ff` | `2b74a745...11d399` |
| Padded ordinary-role batch token dimensions | 2 × 26 | 2 × 26 |
| Artificial left-padding units in short row | 19 | 19 |
| Original role-terminator ID 5 still attended | Yes | Yes |
| All 21 host checks | PASS | PASS |
| Explicit-mask batch vs individual 3-token generation, NCA/MLP | Exact tokens | Exact tokens |
| Generation with vs without cache, all 3 modes | Exact tokens | Exact tokens |
| Disabled vs zero-readout generation | Exact tokens | Exact tokens |
| Peak source runner RSS | 3,293,176 KiB | 3,292,816 KiB |
| Original independent CPU runner | `runnervmtai0g` | `runnervm48cm3` |

The three modes are disabled frozen reference, enabled zero-readout, and **manually perturbed untrained** readout. Both model lanes emitted the *same selected three greedy tokens across all three modes*; this is **not** evidence of response quality, adapter learning or any measurable generation improvement. It does demonstrate identical batch-versus-single and cache-on-versus-off output under these exact explicit masks. The original normal model-generated warning `torch_dtype is deprecated` is preserved in both unmodified host logs.

The genuinely visible `<|end|>` role delimiter and synthetic padding share token ID `5`; equality-to-pad masking is invalid. The tokenizer-generated mask and explicit mask argument are necessary to remove this ambiguity. The tested 2×26 batch is **not a new 2048-token full-cache comparison**: those earlier distinct physical checks belong to #155/#156 and retain their own source hashes.

## Byte-exact archive and independent auditor

The original Actions artifacts were independently downloaded and SHA-256 verified before decoding and re-bundling. This review retains all **six unmodified original members**, including both raw JSON original frozen-host receipts, both complete original host stdout logs and the two original source test logs. They are stored byte-for-byte in [original-receipts.tar.gz.b64](../experiments/ncr-001/controls-evidence/sprocket-explicit-mask-20261028/original-receipts.tar.gz.b64); [manifest.json](../experiments/ncr-001/controls-evidence/sprocket-explicit-mask-20261028/manifest.json) records each original member's byte length and SHA-256, whole archive SHA-256, original Actions ZIP artifact IDs/digests, exact head/runner identity, original file hashes and negative outcomes.

| Artifact | Original Actions ID | Original ZIP SHA-256 |
| --- | --- | --- |
| Original source/negative logs | 12820198139 | `d750b5a9...faaadfe9e` |
| Original Chat host | 12821505646 | `3065c5d4...7070d63` |
| Original Instruct host | 12821176345 | `c400f4bd...684d0b0` |

Bundled original gzip SHA-256: `719c95e11020c9c904f203ccad1d07494af2fa252c0b6f4abd1ab9fadcd84247`. The original ZIP binary files remain in their original Actions staging artifacts; the **actual raw ZIP members are now durable in Git** and can be compared with those originals using the committed hashes.

The stdlib-only [evidence auditor](../scripts/ncr_001_sprocket_mask_evidence_audit.py) decompresses all six original bytes, validates member SHA/size and exact source/weight/runner/check identities, checks every reported token pair **rather than trusting Boolean PASS flags alone**, verifies no false training/tool/semantic admission, and tests adversarial receipt mutations that must be rejected. The [archive-only CI](../.github/workflows/ncr-001-sprocket-mask-evidence-audit.yml) reruns this auditor at the exact new PR head, with no model download or optimizer.

## Retained unsuccessful attempts

The two initial source-branch Actions executions [40698646280](https://github.com/techrote/neurounner/actions/runs/40698646280) and [40698793257](https://github.com/techrote/neurounner/actions/runs/40698793257) **failed at workflow YAML parsing** because the initial job step name included an unquoted colon. They did **not** run model jobs. The correction was committed at the original tested head `8275063544b90c83d2740f0d356288460b58ef6b` without changing scientific metrics. Original error links and failure classifications remain in the manifest; no non-existent raw job logs are claimed.

## Gates explicitly still closed

**This is not final #72 acceptance.** Supervised finite loss, post-warmup NCA gradient paths, post-optimizer unchanged frozen base, independent scientific reviewer approval and source-authoritative tool/result/memory argument grammar are still unqualified. No comparison of Sprocket Chat vs Instruct task capability is authorized, and no held-out datasets or evaluation suite were touched. SmolLM2 and void.1 remain independent lanes; BYTE-EOS-001 #92 remains separate. All PRs remain draft. **`train_authorized=false` remains unchanged.**


## Archived complete decoded Actions job logs and independent readback CI

All three **complete decoded GitHub Actions job logs** for the original successful frozen-host run, and the initial failed and corrected archival audit job logs, are now stored unabridged under [actions-decoded](../experiments/ncr-001/controls-evidence/sprocket-explicit-mask-20261028/actions-decoded/). The complete original job logs include source download SHA verification, runner and package installation, and artifact custody in addition to the six original raw ZIP members. Each log is its own tracked Git blob; no altered pass/fail verdict was substituted.

The first archival-only [CI #40700438530](https://github.com/techrote/neurounner/actions/runs/40700438530) **failed** before executing its auditor because a new Python negative-test fixture used the reserved keyword `pass` as a named argument. It has an exact decoded failure log committed. Commit [`3688d85db0ebf9cc3ef51cafcb1132c45d4fdfdb`](https://github.com/techrote/neurounner/commit/3688d85db0ebf9cc3ef51cafcb1132c45d4fdfdb) corrected only the fixture syntax; the original six raw physical files, archive SHA, protocol, gates and source-matched original execution were not changed.

The corrected, exact-head [CI #40700518582](https://github.com/techrote/neurounner/actions/runs/40700518582) **PASS**: all six original members hash-verified and audited; 21/21 actual frozen-host mask/generation checks per checkpoint; **13 adversarial receipt-corruption/false-admission mutations rejected**; original source tests 6/6 and shared-decoder lease regressions 10/10 read back. Its full decoded success log is also committed. This is **independent offline evidence readback**, not an independent scientific reviewer judgment, optimizer-based conformance, or a new model run.
