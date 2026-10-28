# BYTE-EOS-001 — independent byte custody for original frozen Qwen3 forward-only EOS labels

Draft PR #166 stacks on draft PR #165, whose original **exact-head**
GitHub Actions run [40738222796](https://github.com/techrote/neurounner/actions/runs/40738222796)
executed both no-model source guards and the **original pinned** void.1 Qwen3/ByteTokenizer.
This branch does not rerun the frozen model or alter the source under test.

## Physical source and measured original results

Original Git source head: 06b858a70d7019398cdc7a3f205ae24423f57c67.
Source job 122992443448: **30/30 offline tests passed**.
Original host job 122992543245: **passed** on GitHub Actions 1000046597,
with torch 2.14.1+cpu, Transformers 4.57.6. All 6 original pinned model /
ByteTokenizer/config source files were verified by SHA-256 before execution.

| Synthetic fixed example | Causal supervised byte units | Original Qwen3 forward CE | Independent shifted CE |
| --- | ---: | ---: | ---: |
| One-byte context, empty completion (EOS target only) | 1 | 8.765803337097168 | 8.765803337097168 |
| Multibyte UTF-8 completion | 21 | 3.0656538009643555 | 3.0656538009643555 |
| Literal text "<eos>" plus real EOS 258 | 6 | 4.957742691040039 | 4.957742691040039 |

All absolute loss discrepancies are **0**. Both original frozen backbone
fingerprints were exactly
d711be05faf9064801eba19e91e2f58ff9d40ad1c066fea640542762d98423bb;
no frozen gradients, optimizer or backward were performed.

## Permanent raw custody / independent audit

- The exact original GitHub Actions artifact **12838158155** ZIP, SHA-256
  23f0c5bbca8cfacf9050656ed3a972ca4d716ce0c9db04c40a7bcb524c03f770,
  is preserved byte-for-byte in
  experiments/ncr-001/byte-eos-001/frozen-forward-evidence/original-actions-zip.xz.base64
  (lossless LZMA compression and base64 only). The archive contains all
  **seven** original member files including complete source hashes, package
  freeze, static ByteTokenizer audit and original numerical JSON receipt.
- Both **complete original decoded job logs** are committed separately, with
  Git blob identities and runner/run IDs recorded in the manifest.
- The standalone offline auditor verifies original ZIP and member SHA-256,
  original model/checkpoint/tok source, full logs, exact causal target IDs
  and masks, native/independent forward loss, frozen backbone before/after,
  and all no-training/no-admission flags. Adversarial tests inject false
  training acceptance, weight drift, loss corruption, wrong source identities,
  altered ZIP and modified logs. Exact-head source-only CI reruns the
  evidence audit. This is archival reproducibility, *not* a new physical
  model run or distinct scientific approval.
- Original GitHub Actions artifacts remain available under their configured
  retention for further live byte readback; Git custody avoids treating
  Actions retention as permanent storage. Failed/superseded original runs in
  parent #163/#164 remain intact and are neither deleted nor recast here.

## Boundaries

No change to TRAIN-002b or its historical EOS mismatch, no empirical EOS
generation propensity, no NCA/MLP post-lease cache or gradient testing, no
comparative training, no sealed heldout, no change to preregistered protocol.
This forward-only result is **not final #72 conformance PASS**. Independent
scientific review and the four separately qualified model lanes remain
outstanding. All parents and this PR stay draft, and **train_authorized=false**.

## Archived source-bound audit execution (28 October 2026)

The initial offline archive-audit Actions [run 40739674307](https://github.com/techrote/neurounner/actions/runs/40739674307) **failed** after exact source checkout when the bounded XZ decoder had only 32MiB of memory allowance for the original preset-9 LZMA dictionary. This was a custody-harness limitation, **not** a physical-model or source/numerical failure. The complete original failed job log is retained in Git.

A strict change to the offline decoder cap, **32MiB to 128MiB**, preserved the original compressed ZIP and every member byte, the original physical source/test head and all scientific thresholds. At source commit c1307934facddcb0974a540905f45d12399bd713, [run 40739776401](https://github.com/techrote/neurounner/actions/runs/40739776401) **passed** the original-archive offline audit and **7/7 adversarial/audit tests**, rejecting false training admission and source/ZIP/log/measurement mutations. Its complete successful job log is also in Git. The final manifest and offline auditor now cross-check both historical CI log Git identities. Subsequent exact-head audit CI must also pass to accept the expanded archival index.

No physical Qwen3 was rerun, no optimizer/heldouts were involved, and neither archival CI outcome constitutes distinct independent scientific reviewer approval.
