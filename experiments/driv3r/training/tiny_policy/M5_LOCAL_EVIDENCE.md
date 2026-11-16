# M5 topology-diverse tiny-head experiment — local evidence, HOLD

2026-11-16. Research-only synthetic 96×96 CyberSand-like action host. No native/Workbench/FINAL/game changes; no pretrained-backbone or LoRA updates. This is a local experiment, **not** an Actions-qualified checkpoint or a promotion.

## Scope and protocol

Protocol `tiny-policy-m5-topology-v1`, legal host actions exactly `UP DOWN LEFT RIGHT STOP`, 64 decisions of a 96-action host budget, STOP never suppressed. Generated local 13×13 logical-node DFS mazes (3 native pixels per edge) with seeded braids and protected gates. TRAIN seeds 21000–21035 (36), DEV 22000–22007 (8), new-world EVAL 23000–23011 (12). Prior M4 OOD comparison uses original 11040–11051 (12) on **known research topologies, not blind final**. Three families on identical geometries. Every split's seed set is disjoint.

Training uses locally generated legal-action examples from oracle/random/wall/novel explorer states, a 7×7 blocked/visited/protected local observation and 162 features total. Labels use locally visible two-step novelty and family-specific imperfections, never global reachable-cell targets. Actual supervised CPU AdamW updates: **240 per head, 720 total** across independently trained linear (815 parameters), MLP (20,549) and convolutional (53,605) heads. TRAIN 24,300, DEV 5,400, diagnostic EVAL 3,492 synthetic examples; no native corpus admission claimed.

TRAIN identity SHA-256: `2728c8ae225cc122dd39102d0a80f7866a31922fd3f6b39868aa038f60ea6f4c`.
DEV: `a89237606e77667bcbed8d965a6954127a1c63c927db886592b6e0d4a0a70e04`.
EVAL: `94317eaacac7d4eefd2147ce0c04fb49215522f88b359b3ec7f27d52acc8b157`.

## Measured original-host 64-action closed loops

Policy | New maze mean unique cells /36 | Known M4 OOD mean /36 | M4 branches mean | M4 early STOP
---|---:|---:|---:|---:
Novel deterministic | 51.333 | **52.500** | 0.833 | 0
Local synthetic oracle | 48.389 | 41.861 | 0.333 | 0
Random legal | 12.056 | 15.056 | 0.972 | 0
Wall deterministic | 8.250 | 42.667 | 0.667 | 0
M3 linear frozen checkpoint | 46.000 | 33.694 | 0.111 | 0
M3 MLP frozen checkpoint | 48.972 | 36.722 | 0.306 | 0
M3 convolutional frozen checkpoint | 49.000 | 33.778 | 0.000 | 0
**M5 linear trained** | 49.083 | 38.639 | 0.306 | 0
**M5 MLP trained** | 48.361 | **43.417** | 0.528 | 0
**M5 convolutional trained** | **50.083** | 38.806 | 0.444 | 0

M5 MLP improves against its M3 ancestor by 6.695 cells on prior M4 OOD (+18.2%) and yields three distinct family-conditioned trajectories in 10/12 M4 worlds (M3 MLP: 6/12). However, novel deterministic still wins by 9.083 cells, and M5 MLP does worse than M3 MLP on the new maze group. This is **not** clean universal generalization or artistically meaningful promotion. Novelty coverage alone strongly favours straight routes. No neural candidate promoted.

Held-out synthetic oracle action-agreement diagnostics: linear 88.918%, MLP 94.702%, conv 89.490%.

## Independent qualification, provenance and limits

- **720** paired original 96×96 PNG/JSON trajectories; **46,045** model/control legal choices independently recomputed from original checkpoints and observed local states; **720/720** full state/legal/action/PNG replays pass.
- Full SHA manifest **1,449/1,449** content members; seven local positive/adversarial unit tests pass.
- Two independent full optimizer+rollout executions produced **1,449/1,449 byte-identical** evidence members.
- Immutable original M3 checkpoint SHA-256 retained: linear `a5d1ec8f5f956c05968f7b5b538952e68d89b861d337a48559e0de759fc32435`; MLP `1b2c9cea0b99fffde0b3ebd39942835540880d93a6e70ce1518ff39c508ceeb5`; conv `b44a97d0a6af84010280bf83e22c481afa15546d8b1575ef4c855d5bb6661d6d`.
- M5 local checkpoint SHA-256: linear `8382459268e258025ef5e0324ad2aceea385a8ae084f0fac8d5f793f4a90db8d`; MLP `55f0ea469278086bef2c91b533a8451e1938d1739d596dd3f0ed8542a7267940`; conv `39d0c2eb067eb0a12cb454f859de8f6434294704f7e3494af1999d25b484f636`.
- Executed local M5 source SHA-256: `m5_topology_train.py=c7fb7683f44c5248b78b2d353f598fe81a8d9e138dc3d32ab9f8a94ecb901680`, `audit_m5_topology.py=bf48897130698af32250a2a8a452dc937c1ab5c021861318622182cd3f31874f`, `test_m5_topology.py=878d2f544b3b25b32ca2a700bf811b113ff011ce1cc53b2370e4880b588b37cb`.
- Complete source/checkpoints and 1,449 original evidence files in the local conversation artifact `tiny-policy-m5-topology-diverse-evidence.zip`; archive SHA-256 `340b9df45223d1f5f1c9d4a36b549cb4d754c4a34a806b90f0ad6a1f47f14aea`. **The local artifact is not yet an immutable GitHub Actions artifact and the M5 .py source is not yet committed.** These explicit provenance gaps block promotion. Checkpoints are intentionally excluded from Git history.
- The local M5 harness used an M3-reconstructed `core.py` behavioural counterpart; the repository's own core blob is `e323f6ba333c13785f1348f55bacdb7182ef6bfe`, not byte-identical to the local reconstructed copy. Hosted qualification must use exact repository core and verify original receipts against it. No native-authority claim.

## Next experiment / required gate

Commit the executed three M5 scripts after transferring exact source bytes, run source-matched hosted validation against repository `core.py`, store checkpoints in immutable Actions artifacts, then introduce *blind new topology families* and explicit artistic/branching metrics instead of tuning on M4 repeatedly. No M1/SMOL optimizer re-run justified solely by this evidence. Keep PR Draft/HOLD, unmerged.

## M5 source-publication qualification — 16 November 2026

The previous provenance limitation that M5 Python scripts were absent from Git has now been **resolved at the exact blob level**. The three files are committed under this lane on PR #634, with remote blob SHA-1 matching the already locally executed contents byte-for-byte:

| M5 file | Exact original Git blob |
|---|---|
| `m5_topology_train.py` | `3fe56a89f9f43962b016b5737660314d16584703` |
| `audit_m5_topology.py` | `48039d155cf0150b098633881947401c80a9b6e3` |
| `test_m5_topology.py` | `97d7ba49f9c5f7fc817fdef3539ca534b4f8dc87` |

An additional independent **local rerun from these exact source bytes** regenerated M3 checkpoints using the original M3 source and verified all three expected original M3 SHA-256 digests. It then reran the complete M5 training, matched **1,449/1,449 manifest members byte-for-byte** against the original M5 archive and replayed/recomputed **720/720 trajectories and 46,045/46,045 action selections**. Seven M5 tests passed. Measured local full regeneration: M3 19.14 s, M5 28.30 s and M5 audit 15.12 s on the current local CPU. This does **not** assert timing or reproducibility on a hosted GitHub runner.

The dedicated `[M5]` job in `.github/workflows/driv3r-tiny-policy-pretrained-cpu.yml` is separately gated and fail-closed: regenerate and source-verify M3 checkpoints, run actual bounded M5 AdamW optimization and original PNG receipt production, independently replay model/control choices, then upload checkpoint/receipt artifacts (retention 21 days). This is CPU research only and cannot silently rerun the separate M1/M2 pretrained jobs.

**Qualification status until that exact-HEAD hosted job succeeds:** local source-byte identity PASS; local 1,449-file reproducibility PASS; hosted source/run/artifact pending. No native gameplay, production integration, FINAL use, or model promotion.

### Hosted M5 checkpoint-identity caveat and repaired CPU lineage

The first hosted attempt, [run #43819296015](https://github.com/techrote/cybersand/actions/runs/43819296015), genuinely regenerated all three M3 checkpoints but **failed closed** when their hashes were compared with the earlier local M3 reference: linear matched, MLP and convolutional differed. It is a failed qualification, not a successful M5 optimizer run. The generated divergent checkpoints are preserved in the failure artifact.

The repaired workflow **does not silently accept/relabel these as the original M3 weights**. It records a separate `m3_reference_status.json`, then M5 can explicitly use `--m3-rebuilt-from-source` only when the M3 source-run's `report.json` confirms the synthetic protocol, zero illegal choices and exact checkpoint hash identities. The M5 result retains the actual M3 SHA-256 lineage and producer-report hash. Its independent audit fails if this lineage is inconsistent. Without the flag, the exact original M3 SHA-256s remain mandatory.

Locally exercised the new explicit source-rebuild path: **9/9 M5 contract/adversarial tests**, another real M5 optimization, and **720/720 original receipt/PNG/model decisions fully audited**. On that local CPU, the source rebuild happened to reproduce the original M3 hashes and all three original M5 checkpoint hashes; that does not imply the GitHub runner must do so. The subsequent hosted job is qualified or rejected independently under its actual SHA/source identity.

No weights/checkpoints are committed to Git. No native host or production files are modified, and no model is promoted.
