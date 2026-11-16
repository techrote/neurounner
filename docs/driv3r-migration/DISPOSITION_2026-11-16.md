# CyberSand → Neurounner research relocation disposition (16 November 2026)

**Owner directive:** Eliminate model-only pull-request proliferation in CyberSand. Maintain model/SLM science in Neurounner, preserve the game-facing CyberSand integration boundary, and do not merge without owner acceptance.

## Verified source preservation

- Target: https://github.com/techrote/neurounner/pull/252 — single **Draft/HOLD**, unmerged, source-consolidation PR.
- **1,208** source files transferred with Git-blob SHA checks at both ends. Four successful one-shot CI runs:
  - Primary 553 files: https://github.com/techrote/neurounner/actions/runs/43948648049
  - Auxiliary painter and fleet 506 files: https://github.com/techrote/neurounner/actions/runs/43949091251
  - Historical Model Zoo recovery 8 files: https://github.com/techrote/neurounner/actions/runs/43950043336
  - Stage D offline science and pinned shared-contract fixture 141 files: https://github.com/techrote/neurounner/actions/runs/43951430244
- Source-PR/head/files/sha custody: MANIFEST.json, AUXILIARY_MANIFEST.json, HISTORICAL_MANIFEST.json, STAGE_D_MANIFEST.json in this directory.
- The original PR comments, Git source commits, Actions run links, discussions and scientific cautions remain on CyberSand. Externally stored checkpoints, CI artifact ZIPs and output galleries are **not copied** by this code migration. Artifact availability and retention must be handled separately.
- Source-transfer qualification consisted of exact Git blob checks and Python compileall. This is **not a reproduction** of old model-training runs, a claim of performance improvement, or native Windows acceptance.

## Administrative closures — 44 CyberSand PRs, all unmerged

Each closed PR received a migration pointer and an explicit non-acceptance notice. The following original PR numbers were closed as misplaced model research:

**Model Zoo:** #632, #642, #645, #646, #648, #649, #650, #652, #653, #654, #655, #656, #657, #658, #659, #660.

**Research training, evaluation, ablations and architecture:** #616, #621, #622, #623, #626, #629, #630, #633, #634, #635, #636, #637, #638, #639, #640, #641, #643, #644, #647.

**Painter model experiments and W0 model-side support:** #573, #574, #575, #576, #614, #615.

**Stage-D synthetic/offline experiments:** #596, #598, #600.

## Deliberately retained in CyberSand

The following were **not** closed because they own or qualify native game integration, controls, Workbench-facing interfaces, or gameplay boundary contracts: #567, #569, #580, #594, #607, #608, #610, #611, #613, #627. In particular, #594 remains canonical authority for the DRIV3R observation/action contract; Neurounner only carries a pinned historical test fixture. These are not blanket endorsements or merge approvals.

There are no new per-experiment research PRs required in either repository. All 14 previously CyberSand-directed hourly research workers were paused, retargeted in their saved instructions to **Neurounner #252 only**, then re-enabled. The already-Neurounner DREAMGRAFT worker was left alone. The updated worker prompts forbid further CyberSand writes, automatic merges, Ready promotions, and per-experiment PR creation.

**Remaining owner gate:** Neurounner #252 stays Draft/HOLD. Do not merge, mark Ready or claim scientific / hardware acceptance without independent qualification and explicit owner approval.
