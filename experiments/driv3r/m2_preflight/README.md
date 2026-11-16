# DRIV3R — M2 integration preparation snapshot

**Status: OWNER-AUTHORISED M2 OFFLINE PREPARATION ONLY.** Not M3/native integration, not M4 model evaluations, not adoption or merge. Source is a **test-only composite** of three independent worker subtrees on the approved immutable M1 contract. The original worker branches and Draft/HOLD PRs stay independently owned; this snapshot is not a replacement for their contributions.

## Frozen shared reference

- Canonical issue: [#570](https://github.com/techrote/cybersand/issues/570)
- Original qualified Stage A–C research: [Draft PR #580](https://github.com/techrote/cybersand/pull/580) (evidence preserved, not modified)
- **M1 approved commit:** `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6`
- **M1 schema SHA-256:** `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`
- [M1 acceptance comment](https://github.com/techrote/cybersand/issues/570#issuecomment-6881686675)
- This branch is **based directly on the frozen M1 commit**, not a mutated source contract. Future source commits must not pretend to be that accepted M1 identity.

## Worker snapshots included for offline validation

| Workstream | Original immutable source snapshot following merge of M1 | Lane-owned paths | Scope |
| --- | --- | --- | --- |
| [D-1A #595](https://github.com/techrote/cybersand/pull/595) | `c82578a83cf7d21aea9c38119280d63911639d7c` | `experiments/driv3r/stage_d/direct/**` | 36-step selected-wall/routing fixtures, geometric and random controls, simulated divergence |
| [D-1B #596](https://github.com/techrote/cybersand/pull/596) | `08b4b67c4bc1310c1cb22512e82c1592b4d50a27` | `experiments/driv3r/stage_d/guidance/**` | Scalar airbrush authoring, PNG source custody and shadow KEEP/UNDO/STOP, A1/A2/A3 information views |
| [D-2 #598](https://github.com/techrote/cybersand/pull/598) | `0e8c3654149dd3898deb5e69303b679f8c32ba7f` | `experiments/driv3r/stage_d/procedural/**` | Weathering Worm, Branching Growth, Guided Coverage, persistent cursor rewind, synthetic corpus and negative-control study |

The three worker branches were reconciled with M1 through **non-forced fast-forward two-parent commits** without changing their lane-specific file bytes. Their PR diffs relative to frozen M1 remain lane-exclusive and inherited malformed CI YAML was repaired by taking the M1 version.

This snapshot combines their tree entries only for **static checking and fully offline synthetic unit tests**. No worker branch was merged into main, marked Ready, or used for a native run. The D-2 `ci/experiment-matrix.yml.disabled` remains inert and outside `.github/workflows/`.

## M2 bounded qualification

[Standalone offline-only Actions workflow](../../../.github/workflows/driv3r-d-m2-offline.yml) runs four independent jobs (`fail-fast: false`): accepted M1 schema/67 contract tests, D-1A direct tests, D-1B guidance tests and shadow CLI, and D-2 procedural tests and deterministic replay of DEVELOPMENT-only corpus recipes. It confirms the approved schema SHA and M1 ancestry and executes no native Godot process, no model backend, no API, no GPU, no synthetic-as-native promotion and no held-out outcome campaign.

The worker PRs' normal repository checks *do not automatically prove* the new lane tests ran; inspect the exact M2 workflow logs. A failing test is an engineering finding to repair at the **owning source lane**, followed by an exact-head re-snapshot and new qualification.

## Known interface differences requiring deliberate adapters (not silently merged)

- **D-1A** emits its own offline `fixture/observation/candidate` records. The M1 source-bound `observation` and `action` require actual native World epochs and receipts before any real edit. Candidate-set parity needs the M1 `legal_candidate_set_sha256` identity, separate from model-visible ordering and score annotations. Existing pure-python simulated effect divergence is **not a native consequence**.
- **D-1B** uses `positive_scalar/v1` as its internal baseline and `signal_class="geometry"` for signed-distance/flow in its standalone synthetic information helper; M1's mandatory scalar is represented by the guide's top-level `u16le` values, while optional `signal_class` must be `geometry_metric`. An explicit source-pinned mapping is needed; never silently relabel or interpret a channel. The A1/A2/A3 ablation outputs must retain a model-view boundary excluding unused channels while audit provenance retains their exact bytes/hashes.
- **D-2** has independent, source-hashed synthetic raster recipes, a `canvas=[w,h]` recipe field and cursor-only backtrack/branch relocation semantics; its output is NOT a D-COMMON native event journal and cannot be registered as real native receipt or history Undo. Actual RRT-like exploration, visual inference and graphical framebuffer evidence remain unproven.

**Recommended adapter preparation (M3 proposal only):** one central, source-pinned read-only adaptor per lane with explicit mappings, refusal paths for unregistered semantics, numeric scaling, origin/canvas/trace bounds, and authoritative native World pre/post fences. Keep original raw events unmodified and bind projections to their source hashes. Host-controlled material authorization, Stop, swept footprint, actual Undo and independently replayed readbacks require a separate native integration/qualification approval.

## Gate boundaries

- **M1** accepted/frozen. Do not alter its source or schema hash.
- **M2** this branch: cheap unit tests, syntax/policy, combined static compatibility and a clear owner handoff only.
- **M3** needs Owner's **separate** approval for native adapters/execution, chosen exact source, budgets, Stop/undo/cancel testing and GPU/visual-backend prerequisites.
- **M4** needs separate Owner approval for matched A1/A2/A3 model/native episodes, initial same-legal-candidate pairs, independent diverged Worlds, first consequential material/cursor/procedural change and recovery leverage. No trained model or paid API/GPU costs in M2.
