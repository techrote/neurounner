# DRIV3R D-2: procedural discovery (independent, offline)

This lane is a **synthetic-only preparation** for CyberSand [#570](https://github.com/techrote/cybersand/issues/570#issuecomment-6877408001). It must not write native World cells, emulate a native receipt, run a resident model, request paid resources, or claim production results. All changes are confined to \`experiments/driv3r/stage_d/procedural/**\`. [D-COMMON-001 #594](https://github.com/techrote/cybersand/pull/594) has an **owner-accepted immutable M1 source pin** (approved 14 November 2026: commit \`8fa90de6be209ed0b7bd125039e6ec8b332fb1a6\`, schema SHA-256 \`ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d\`).

## Research families

| Family | Original geometry | Routine algorithm | Recovery hypothesis | Material effect |
| --- | --- | --- | --- | --- |
| Weathering Worm | Frozen, filled Minkowski/n-flake polygon; **do not chase the newly eroded edge** | Original-perimeter-biased connected walker; branch-frontier selection | Brush-up backtrack or relocation to retained original-rock frontier | Persistent **erase** |
| Branching Growth | H-tree / branching skeleton and authorized volume | Raster-tree traversal from a fixed root with branch-frontier bookkeeping | Retraction to cursor-history ancestor versus logical branch relocation | Persistent **add**, compare contained vs bounded overshoot |
| Guided Coverage | Full Hilbert, closed Moore, and serpentine curves (not Stage B's \`HILBERT_MOTIF\`) | Incremental curve cursor, explicit visited center/footprint and next-segment inspection | Low/high scalar hysteresis, seek a *future unvisited four-step segment*, pause or record trap | Persistent **add** |

**Reversal semantics are not interchangeable.** \`BACKTRACK_CURSOR\` follows a previously traversed cursor chain with brush lifted; \`RELOCATE_TO_BRANCH\` is a discontinuous controller relocation with brush lifted (not a physical stroke or proof of transport). Neither changes native material. \`UNDO_MATERIAL_EDIT\` is **unavailable in the offline simulator**: only a future integration owner may issue a host-authorized native-history operation with source-fenced receipt, full original material hash/readback and bounds/partial-prefix safety.

## Records and custody

Keep an immutable, versioned **recipe** (generator, seed, World dimensions, permitted effect, curve/walker, guidance, thresholds, brush, recovery, budget), the original generated geometry and scalar guidance, candidate IDs and full offered catalogs, selected actions, exact visited/swept footprints, trajectory, failures and terminal reason. Save original before/midpoint/after **synthetic material-raster PNGs**, never call them framebuffer screenshots. Source and input digests, artifact SHA-256/byte lengths and a replay verifier must travel with every episode. Every absent native/model identity is explicitly \`null\` and \`native_replay_verified=false\`. Do not promote this exploratory corpus as native painter examples or training labels.

Metrics should be based on **actual executed synthetic changes**, not estimated tool plans: before/after material delta, connected traversal, branch coverage, retraces, original-boundary offset, retained negative space, trapped recoveries, viable reentries, uninterrupted steps per intervention, and total failures. Compare identical initial geometry, guidance, seed, budget and legal host candidate set; after first consequential divergence each arm is independently re-observed. Save failed/ugly trajectories rather than filtering them.

## Local preparation

Run the standalone Python module with stdlib only: \`python3 experiments/driv3r/stage_d/procedural/lab.py --out /tmp/driv3r-d2\`; test with \`python3 -m unittest discover -s experiments/driv3r/stage_d/procedural -p 'test_*.py' -v\`. This is a small **development** cohort, not held-out confirmation. Generated images/results belong in temporary artifact storage, not committed to Git history.

**Workflow preparation only:** \`ci/experiment-matrix.yml.disabled\` is a deliberately inert template under this lane. A future integration owner may, after *explicit budget, source and dispatch authorization*, review and migrate it into \`.github/workflows/\`; this lane does not create a runnable workflow or dispatch anything.

## Future integration/approval gates

1. **M1 ACCEPTED** exactly at D-COMMON commit `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6` / schema SHA-256 `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`; M2 offline reconciliation authorised. M3-A [#601](https://github.com/techrote/cybersand/pull/601) proved a separate **single-cycle** real-native logical brush-up backtrack only. Neither M1 nor M3-A authorises D-2 native integration, real Undo, long-horizon models, formal A/B dispatch or merging.
2. Host-owned World permissions, swept cells, Stop/timeout/cancellation, observation fence, readback, undo receipt and independent native replay qualification. No fake \`native\` field is permitted.
3. Opt-in 2/4/8-runner throughput measurement before considering a paired 4 steering × 3 seeds × 2 recovery = **24 episodes** model/native cohort. Two-tier screening on 8–12 identical saved observations, with raw distributions, option orders/paraphrases, latency, explicit information ablations and structurally distinct held-out confirmation.
4. Separate authorization for every native/model/GPU/API/training dispatch. Draft/HOLD, no merge, Ready or production-default changes.

## Synthetic control and replay-hardening checkpoint

- Weathering Worm now records a **synthetic offered-candidate catalog and selected point** at each step. `walker_policy=boundary_biased` retains the frozen-original-contour preference; `random_legal` is a seeded, all-other-inputs-matched negative control. Neither is a host/native legal catalog or a model decision.
- `study.py --recipe <development-recipe.json> --out <local.json>` independently simulates both policies from the same original World and scalar field, reports first **choice** divergence separately from first **synthetic material** divergence, and scores observed erosion, retained rock, offset error, reversals and traps. The supplied CLI requires a single recipe object, such as one case's `recipe` extracted from `corpus/development-recipes.json`. It does **not** estimate neural incremental value or assert either policy wins.
- Guided Coverage now explicitly logs every **virtual, brush-up seek to an unvisited future point**, its origin/destination and unchanged material hashes. A prior branch would record only a no-op backtrack and then silently jump; a one-step actual backtrack now occurs when available, followed by a distinct synthetic seek. The seek is **not** a verified native transport or a legitimate use of `RELOCATE_TO_BRANCH` (which remains restricted to visited anchors).
- Offline replay independently reconstructs **all emitted files byte-for-byte**, including guidance, raster before/mid/after, visited geometry and previews. Rewriting an artifact *and* its manifest hash is insufficient to forge a passing replay; source identity is still a hash, **not cryptographic signing or independent native evidence**.
- The development corpus now has **eight** cases: the six original cases plus a matched random-legal weathering control and a serpentine recovery variant. All remain DEVELOPMENT; preserve failure records and do not use them as held-out model validation.
- Cheap local commands: `python3 -m unittest discover -s experiments/driv3r/stage_d/procedural -p 'test_*.py' -v` and `python3 experiments/driv3r/stage_d/procedural/runner.py --out /tmp/d2-corpus`. The test command includes new negative-control, tamper and gap-accounting assertions. Do not claim exact-branch qualification unless both commands actually run.

## Read-only structural audit (14 Nov 2026)

After generating the **DEVELOPMENT-only** corpus with `runner.py --out /tmp/d2-corpus`, run `python3 experiments/driv3r/stage_d/procedural/audit.py --corpus /tmp/d2-corpus --out /tmp/d2-morphology-audit.json`. This second consumer **independently decodes original synthetic material PNGs**, verifies artifact hashes, reconstructs every synthetic STEP material mutation from recorded admitted footprints and compares it with the material image. It scores four-neighbour component counts, changed-material area, zero-change strokes, recovery exits, virtual brush-up seeks and subsequent deterministic steps; forged event/material records cannot pass simply by rehashing their manifests. Its tests are in `test_audit.py`.

The audit is **not** an independent regeneration of the original geometry or a native World receipt. Continue to use `lab.replay` for deterministic reconstruction, and do not treat file hashes as authenticated evidence. Results and their limitations are recorded in `corpus/development-morphology-findings.md`. The first eight-case study found four disconnected components in the Moore result versus one in Hilbert and serpentine, despite all three reporting 160 painted points. This is an actionable **material-continuity / artwork-quality** failure that naive coverage counts miss; the gaps were produced by unverified synthetic brush-up seeks. Any physical reentry path and artifact utility will require separate offline route qualification followed by owner-gated native verification.
