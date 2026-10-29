# NCR-001E — permanent whole-ZIP custody for original Sprocket NCA-only diagnostics

**Scope:** original GitHub Actions evidence preservation, not fresh model execution or scientific training approval. This *separate descendant draft* branches from [#160](https://github.com/techrote/neurounner/pull/160) exact head `d168fa41ae35536e893a5bc548a7b21965c45999`; original physical [#159](https://github.com/techrote/neurounner/pull/159) head `df8479aae34550cba014d910a73e30e59cd493e2` and [run #40710171622](https://github.com/techrote/neurounner/actions/runs/40710171622) are unchanged. All four frozen-model protocol sources and the six historical raw ZIP members / original job logs in #160 remain untouched.

## Original complete compressed containers

Three whole original ZIPs were downloaded *independently* from Actions (original run 40710171622) and byte-verified before promotion as lossless exact Base64 of their **complete ZIP containers**, retaining original per-entry compressed bytes, ZIP headers, filenames, CRCs and archive metadata. These are not reconstructed ZIPs and are distinct from #160's earlier re-encoded six-member JSON archive.

| Original lane | Artifact ID | ZIP bytes | Original ZIP SHA-256 | New committed ASCII Git blob |
|---|---:|---:|---|---|
| Source | 12826525345 | 1,024 | `c66706d3e541925638e901d37232f204c480f0016aba393023fa925c67a61242` | `518344afea1956d356bd9a661ad1ded5dde8184b` |
| Chat subfolder | 12826246889 | 2,955 | `e040e1ce982a1c0f0c5e99e014cada04cc41b334f1755d60cf37fd0bb909a05b` | `df27a222657606813d8d20ba95bcc293bd8231cb` |
| Root Instruct | 12825942600 | 3,008 | `9656ff3f8019f497cff62d816276f44bdc33f149c07072357bd5cfd9c90d0f93` | `cd5781917412f755e79d7938f96c4a104643055d` |

The three files `original-actions-{source,chat,instruct}.zip.b64` under `experiments/ncr-001/controls-evidence/sprocket-learning-20261028/` are the permanent original-byte stores. Decode with `base64.b64decode(file_bytes.strip(),validate=True)`. The new standard-library auditor `scripts/ncr_001_sprocket_original_zip_custody.py` checks all three **encoded** Git blobs, decodes the originals, verifies three whole-ZIP SHA-256s/lengths, exact six member hashes/lengths and the existing #160 manifest/source identity. Nine offline positive/adversarial tests refuse whitespace-only encoded archive edits, truncation, changed bytes, swapped variants, altered artifact IDs, forged scientific/training admission and ZIP modifications even if the member-level interpretation could survive.

The *read-only* exact-head workflow `.github/workflows/ncr-001-sprocket-complete-zip-custody.yml` additionally checks immutable original decoded source/Chat/Instruct job logs **and original failed formatter-hash readback log**, exact four-model protocol/pins, parent head, and re-downloads all three original Actions ZIPs to independently byte-compare each against the now permanently committed archive. It persists no checkout credentials and has no bot push step.

## Negative evidence and scientific gates

Original [#160](https://github.com/techrote/neurounner/pull/160) already retains the six original raw member bytes, original physical/source job logs, and the initial failed formatter-hash audit job. Those originals remain unchanged. The separate [#194](https://github.com/techrote/neurounner/pull/194) source-only reconciliation covers the *narrow* Chat/root-Instruct one-step synthetic NCA-gradient evidence without granting scientific acceptance.

**Not established:** full 2048-context masked-batch/cache and fresh-process cold-adapter conformance, publisher-generated tool/memory competence (original root-Instruct six-token tool-call prefix is incomplete), matched learned NCA-vs-MLP quality, complete independent four-model #72 review or #73 permission to train. These ZIPs contain no new model results; void.1 EOS ID remains 258 with TRAIN-002b historical impact UNMEASURED. No held-outs opened, optimizer steps executed here, training or merges authorized. `train_authorized=false`, `full_72_conformance_pass=false`.

**Review/CI:** a draft archival PR requires its own scoped review and exact-head CI; owner-accepted same-account COMMENTED administrative exception for #161/#162/#181/#182 does not automatically apply to this PR.
