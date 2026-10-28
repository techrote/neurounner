# BYTE-EOS-001 — original causal-shift failure and exact source readback

**Issue:** [#92](https://github.com/techrote/neurounner/issues/92) (NCR #67/#72)
**Implementation candidate:** [draft #163](https://github.com/techrote/neurounner/pull/163) exact head `9fbdee880697903aaf5b91cc29442c9e53479e50`
**Parent:** [draft #94](https://github.com/techrote/neurounner/pull/94) exact head `b1c558f7d46f5d7c76b4e6336a37143886d16c08`
**Classification:** source-only label construction, not executed void.1 Qwen3 conformance or training admission.

## What changed; what did not

The original future-only BYTE-EOS-001 serializer addressed the historical mismatch
between literal `"<eos>"` (five UTF-8 byte tokens) and terminal integer
`258` (one dedicated EOS token). Its output had one further causal-loss blind spot:
with an empty prompt, the model's shifted labels `labels[...,1:]` drop the
first completion byte, or drop EOS entirely if the completion is empty.
A serializer that checks only the unshifted final label can misreport
supervision. The new candidate requires at least one valid UTF-8 prompt
byte and verifies the count **after the standard decoder-only causal shift**.
It does not silently insert a BOS token, change budgets, modify sealed data,
replace actual tokenizer source, or authorize training.

The corrected candidate tests seven new adversarial cases plus nine original
BYTE-EOS label cases. Both independent runs passed the **same 16** source tests.
These are source-only synthetic tokenizer tests. The original #94 pinned
ByteTokenizer runtime receipt remains separately preserved, **not re-executed
as part of this source-only change**.

## Original source/test custody

The manifest under
[`experiments/ncr-001/byte-eos-001/causal-shift-evidence`](../experiments/ncr-001/byte-eos-001/causal-shift-evidence/manifest.json)
is the machine-readable evidence index.

| Original run | Source checkout | Result | Runner / ZIP |
|---|---|---|---|
| [40728812489](https://github.com/techrote/neurounner/actions/runs/40728812489) | `1d301e31d16132cb56a1cf43cb7fff5a32ca73dc` | 16/16 PASS | GitHub Actions 1000046563; `7df1511725b79cd38ac3a1ea713b6d2191eeb5fd8fde474f19e75cfdda1690e3` |
| [40728952524](https://github.com/techrote/neurounner/actions/runs/40728952524) | `9fbdee880697903aaf5b91cc29442c9e53479e50` | 16/16 PASS | GitHub Actions 1000046564; `4a6cb4b5c32fc336bee68a61bce5c7de186e6aa18ec0ef96dfade70fbe68504e` |

Both **original ZIP containers** are archived byte-for-byte as a deterministic
compressed TAR/XZ encoded in RFC4648 Base64. Reconstruct using
`base64 -d original-actions-zipset.tar.xz.base64 > zipset.tar.xz`, then
`tar -xJf zipset.tar.xz`. The decoded TAR/XZ has SHA-256
`874b2278252dd0615c703366231506e71809621bf9b53c4616fc9bff3deb3fb6`,
and contains exactly the two original Actions ZIPs listed above. Each
ZIP has its original `receipt.json` and `test.log` with independent SHA-256
identities recorded in the manifest. Both complete original decoded
job logs are also committed separately, including runner identities,
executed checks, all-positive results and archival output.

**Original negative retained:** run 40728812489 correctly checked out
`1d301...`, but its receipt mistakenly used GitHub's `GITHUB_SHA`
(`858bea...`), the synthetic pull-request merge commit, as
`head_sha`. The original log, wrong-head receipt and run identity
are preserved. This is an **evidence provenance error**, not an original
scientific test failure. Run 40728952524 corrected the receipt by recording
`git rev-parse HEAD` separately from `GITHUB_SHA`, and matched exact
checkout head `9fbdee...`. Do not silently rewrite the earlier receipt.

The independent offline auditor
[`scripts/byte_eos_001_causal_evidence_audit.py`](../scripts/byte_eos_001_causal_evidence_audit.py)
checks fixed source PR/parent, negative qualification flags, both complete
Git job-log byte identities, both raw original ZIP digests and member
digests, exact runner/source receipts, and the historical provenance failure.
Adversarial tests falsify identities, original bytes and false admissions.
The dedicated Actions workflow also downloads both original ZIP containers
again and compares their **actual bytes** against the recovered Git archive.

## Scientific and protocol boundaries

The following remain **NOT ADMITTED**: actual pinned frozen Qwen3
post-integration learning/cache/cold replay, byte tokenizer supervision on
the real model, independent #72 source/runtime and scientific acceptance,
full four-model conformance, future optimizer steps or any NCA-vs-MLP
comparison. Historical TRAIN-002b source, weights, outputs, evaluator and
heldouts remain unchanged. No generated tool or memory semantics are
qualified here. Four checkpoint identities remain separate and
`experiments/ncr-001/protocol.json: train_authorized=false`.

The archive is durable **on this unmerged review branch**; permanent
availability on main depends on the normal independent review and authorized
integration sequence. No source-only PASS can waive independent model gates.


## Independent readback qualification checkpoint — 28 October 2026

The first original evidence-workflow run
[40730382950](https://github.com/techrote/neurounner/actions/runs/40730382950)
**FAILED** at the adversarial test harness only: the actual auditor correctly
rejected `optimizer_steps=1` using "Nonzero optimizer steps in source receipt",
but one test had expected only "False scientific". No scientific threshold,
model observation, actual original ZIP or protocol was changed. Its
[complete decoded failed job log](../experiments/ncr-001/byte-eos-001/causal-shift-evidence/readback-run-40730382950-failed-job.log)
is permanently retained before the assertion text was broadened.

The follow-up original successful run
[40730479431](https://github.com/techrote/neurounner/actions/runs/40730479431)
at exact candidate checkout
`613fb1b3dd74c55e732cf5e1f855ea30a7617c59`
**PASSED** the offline original ZIP/member/log audit,
**8/8** adversarial evidence tests and **both fresh GitHub Actions
download comparisons byte-for-byte**. Its
[complete decoded successful job log](../experiments/ncr-001/byte-eos-001/causal-shift-evidence/readback-run-40730479431-success-job.log)
and [original source-bound receipt](../experiments/ncr-001/byte-eos-001/causal-shift-evidence/readback-run-40730479431-original-receipt.json)
are permanently retained. The original readback artifact was
[ID 12835221091](https://github.com/techrote/neurounner/actions/runs/40730479431/artifacts/12835221091);
its complete container SHA-256 was
`3794ffdf9b8c2d70fa543d525e7c04486f6d7a19fa3cb5db36af75454a2ffdad`.
The archived original receipt has SHA-256
`e0c9ff5d9f6b198379152d05e2d709a4afb8095f793a7ae56e17cc388d7a74ea`.
This remains **evidence/serializer qualification only**, with no Qwen3
model execution and no training admission.

Any subsequent commits to this review branch require their own fresh
exact-head CI. Earlier successful runs do not prove an unexecuted later head.
