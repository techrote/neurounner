# NCR-001E — original void.1 poststep cold/KV diagnostic custody (30 October 2026)

**Scope: permanent original evidence bytes and no-model source verification only. Draft, not scientific #72 admission.**

This is the source-only preservation of original physical [PR #210](https://github.com/techrote/neurounner/pull/210), immutable source head `f78d302ab6062c3739315c3e04317a9a97b97191`. Its original [Actions #41058648266](https://github.com/techrote/neurounner/actions/runs/41058648266), attempt 1, completed source and actual frozen void.1/Qwen3 jobs. This archival work does not rerun a model or optimizer, modify the #172 synthetic source, inspect heldouts, or restore/distribute the temporary post-step adapter.

## Originals and immutable custody

| Original stream | ID | Whole original bytes | SHA-256 |
|---|---:|---:|---|
| Source ZIP | Artifact 12976819289 | 369 | 3313913ce5dd5e28f4f89f777f3e9d99271c894ee301e495b311a9b3adca992b |
| Host ZIP | Artifact 12977852817 | 6,770 | 75df0cc9733d4873516880ebe2ce449cc5be4d79da03166ee408798dad9ce64b |
| Source decoded original Actions job log | Job 124072474714 | 22,795 | 121779b202b1b505deb81ca9db4082a117b2424401e4162e224529e582d66df5 |
| Host decoded original Actions job log | Job 124072552187 | 53,480 | 8b3c6a826ebc16d22a264e8b9ebd039331d76da6995da4f4a61e4b60e2582e0a |

The four original streams are committed byte-for-byte under `experiments/ncr-001/controls-evidence/void-poststep-cold-20261030/`. Their independently pinned Git blob SHA-1 values, full byte SHA-256 and member inventories are in `manifest.json`; the manifest itself has immutable Git blob `3b56c4b3e05da323e65cc3a187e5d2d320a7a007`. Source ZIP has one `tests.log` member and the physical ZIP seven source/consumer/producer/diagnostic receipt or log members; **no adapter/checkpoint binary** is archived. Original source, runner and Qwen3 pin provenance remain separate from this archival head.

The primary standard-library auditor `scripts/ncr_001_void_poststep_original_custody.py` fails closed on Git tree identity, manifest and all four original raw stream blobs; also checks CRC/member digests, exact source run and job IDs, original EOS/shifted-label identities, exact post-step active-logit equality, two distinct producer/consumer PIDs, A–B–A reset, actual final-four cached logits and no false training/scientific admission. The accompanying adversarial tests deliberately corrupt **every** original ZIP/log, remove original streams, introduce symlinks, forge original metadata, rehash a substituted decoded log, and attempt to promote training/review fields.

The read-only Actions job checks the literal current PR HEAD and unmodified four-model protocol/pins; runs the offline auditor/tests and then independently asks GitHub Actions for the **same original run, jobs, two ZIPs and two fully decoded job logs**, requiring byte-for-byte equality with permanent Git. It has `contents: read`, `actions: read`, no push and no model runtime. The original Actions artifacts may expire 29 November 2026; live byte re-download is a *supplemental* historical recheck, never a substitute for committed raw evidence. Preserve failures and incomplete source runs separately.

## Interpretation boundaries

Original physical source only: one synthetic 31-unit UTF-8 void.1 context, **integer EOS 258**, 21 correctly shifted supervised units, exactly **one in-memory NCA-only SGD diagnostic step**, then separate Python cold reload of that post-step adapter, bit-exact logits and A–B–A isolation. Four genuine final-token KV comparisons had maximum absolute deviation `9.5367431640625e-06` against the original `5e-4` gate; effective Torch threads 4; source/host checks succeeded. This is **not** a 2,048-unit post-step test, MLP comparison, long-term learned checkpoint, source-independent host replication, useful generated EOS behavior, or completed four-model #72 scientific qualification.

Keep the earlier 12/12 short void.1 generations with no EOS 258 and the Sprocket root-Instruct incomplete six-token generated tool-call prefix as negative findings. Correcting void.1's historic literal EOS target to ID 258 has **UNMEASURED** impact on original TRAIN-002b. The owner's same-account `COMMENTED` review acceptance is administrative and is not GitHub `APPROVED`, a second identity or scientific admission. Source/custody technical review and explicit source merge authorization are separate gates.

**Locked:** `train_authorized=false`, `void1_conformance_pass=false`, `full_72_conformance_pass=false`, `actual_compute_budget_qualified=false`; no comparative NCA/MLP training, heldouts, protocol/fixture changes or draft merges. Final CI result must be observed at the exact committed head before describing the new auditor as verified.
