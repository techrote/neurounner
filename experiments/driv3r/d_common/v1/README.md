# D-COMMON-001 — DRIV3R research interface v1

**Status: FROZEN CANDIDATE / OWNER REVIEW — not approved for cross-branch integration.** This is an immutable *versioned proposal*, not an assertion that the five records are already wired into the running Workbench. Owner approval, exact schema SHA-256 and a conformance result are required to mark the milestone **ACCEPTED**.

**Canonical schema:** [interface.schema.json](interface.schema.json)  
**Updated design decision:** The Owner selected a mandatory positive scalar baseline plus **optional independently typed guidance channels** for A1/A2/A3 parallel exploration. See [guidance semantics](guidance-encoding-review.md) and [matched-comparison protocol](matched-comparisons.md). **M1 remains pending final exact-SHA acceptance.**
**Related research:** [CyberSand #570](https://github.com/techrote/cybersand/issues/570), [D roadmap](https://github.com/techrote/cybersand/issues/570#issuecomment-6877408001), [active Stage A–C PR #580](https://github.com/techrote/cybersand/pull/580). This branch was forked from PR #580 head `6067ea31adb8f3dfc76867676fc504078f5e9526` without modifying the qualified source or the #568 painter lanes.

## Scope and authority

The five discriminated JSON records all carry `protocol="cybersand.driv3r.d-common/1"`, `type`, unique `record_id`, `episode_id` and monotonic `sequence`. Each emitted record is immutable. Changes require a new record with a new ID; breaking changes require a new contract major version and explicit coordination. **World dimensions are explicit (`width_cells`, `height_cells`, in 1..65,536)**, rather than forcing a 512×512 World. Integer coordinates and rectangles are dynamically checked against actual World/focus/canvas dimensions; 512×512 remains the primary native test fixture. A single guidance tile and observation crop retain separate allocation/work limits so flexible dimensions cannot demand unbounded memory.

The contract is **not** the preexisting internal `experiments.driv3r.contracts.PROTOCOL`. It is a proposed **D-stage interchange/evidence layer** between independent components, with a future adapter to the already-qualified strict Stage A–C controller. Keep existing receipt hashes and original frozen reports unchanged. No production/default provider change is allowed from contract agreement alone.

### 1. Observation — `type="observation"`

A host-originated bounded read, explicitly tied to `world.id/generation/revision/tick/width_cells/height_cells`, material and full-World hashes. Includes original objective and active subgoal, cursor/target, authoritative material crop descriptor, optional guidance record ID, local host-computed geometry metrics, path-history hash, decision/time/material budgets, previous action and native receipt references, and separately typed image witnesses.

**Host invariants:** material crop bytes must agree with the native recorded state at the declared world fence; bytes/hash/coordinate frame must match; model text and image analysis never override focus, permissions or native writes. If the World changes between decision and execution, the host must re-observe and reject or replan. Geometry metrics must define their units in the experiment manifest, and computed geometry is not a substitute for the raw native receipt. Every native action requires a fresh post-action observation.

### 2. Guidance — `type="guidance"`

User-authored or procedurally generated **soft positive preference (the mandatory v1 baseline)**, *not* a collision mask and never native mutation authority. The field declares **`canvas_size=[width,height]`** and an independently bounded tile/frame inside it. Raster origin/frame uses world-aligned integer XY (origin top-left, +X right, +Y down). The mandatory numeric interoperability baseline is fixed-point **unsigned little-endian 16-bit**, one value per cell in row-major order, 0 = neutral, 65535 = maximally preferred. Blob byte count **MUST equal 2 × width × height**; the field SHA-256 is the authority. PNG/speckles are merely an optional visualization and cannot replace numeric values in numeric-guidance arms.

Seeded gradient-airbrush rendering records radius, centre density, hardness, falloff/gradient length and deterministic seed. Imported PNG conversions must be documented and source-pinned before creating the scalar field. Same scalar values can be rendered multiple ways without changing the guidance objective. There are two versioned thresholds: `exit_below` and `reenter_at_or_above`; **exit must be strictly lower than reentry** to create hysteresis. Crossing a preference threshold triggers a recovery *proposal*, not an automatic material write. The model can prefer to leave the field if all legal candidates are otherwise worse; hard exclusion stays the native host's job.

**Optional typed channels are supported without replacing the mandatory scalar.** `optional_channels` contains 0–8 independently named channels. Each states its `signal_class` (**preference** or **geometry_metric**), versioned `semantic_id`, `encoding`, `units`, exact rational `scale`, sign/zero meanings, optional reference boundary, `canvas_size`, tile `frame`, SHA-256/byte-bound values and source `producer` (kind, ID, revision, original source digest, parameters digest and optional seed). A channel never grants native authority. The canonical built-ins are:
- `soft_attraction/v1` and `soft_avoidance/v1`: independent `u16le` positive nonnegative preference, stored integer /65535. They are not hard masks.
- `signed_preference/v1`: signed `i16le` relative preference, stored integer /32767, negative discourages and positive attracts.
- `signed_distance/v1`: signed `i16le` distance in cell units from a **named boundary reference** (negative inside, positive outside); this is geometric information, not inherently desirable/undesirable.
- `flow_vector_xy/v1`: two signed `i16le` components, X then Y, divided by 32767, world +X right and +Y down, zero means no direction.
The exact canonical names, units, scale and sign semantics are enforced by the cross-record checker. `experimental.*` names can be stored but require **explicit consumer registration** before being counted as used.

Each observation includes `guidance_channel_handling`: `used_channel_ids`, `ignored_channels` (every ignored signal's ID, semantic version, SHA and reason), and `unsupported_policy` (`reject_unsupported` or `ignore_with_audit`). Every declared channel must be accounted for exactly once; an unsupported or ablated channel is **never silently dropped**. Source-hashed conversion is explicit. Channel bytes are retained for native-evidence custody; a scalar field's random speckle gaps are not treated as holes in the numeric preference data.

**Representation comparison arms (not three incompatible contracts):** A1 = mandatory positive core with no extras; A2 = core plus independently authored attraction/avoidance/steering channels; A3 = core plus separated geometric signed-distance/flow channels. These are optional experimental input *configurations* in one v1 interface. See [guidance meaning and alternatives](guidance-encoding-review.md) and the [paired comparisons specification](matched-comparisons.md). No experiment is dispatched by this contract change.

### 3. Action — `type="action"`

A **host-generated candidate/selected/committed** transition with observation ID, exact precondition World fence, candidate-set SHA-256, candidate ID, actor/policy, bounded path, brush state, predicted effects and exact native operation hash/receipt where applicable. A decision model may only choose an offered candidate ID; it never emits unrestricted coordinates, material IDs or permission changes. Predicted effects are non-authoritative. A path with multiple cells must be checked along its **entire swept footprint**; no endpoint-only permission checks.

`brush_state="up"` means logical travel without material mutation. `add` and `erase` mean host-authorised operations, confirmed only by actual native receipt/readback. A `stop` or `no_paint_plan` action must carry no mutation. Keep proposal and result as different immutable event records; do not change a candidate in place once the model has been queried. **A committed material edit must include `world_postcondition`, the source-fenced native receipt ID and `native_receipt_artifact` with actual JSON bytes, size and SHA-256.** `native_receipt_ref` alone is not proof of execution. Failed/partial native work must not be promoted to successful execution because a candidate was legal.

### 4. Reversal — `type="reversal"`

Exactly three behaviours:
- `BACKTRACK_CURSOR`: traverse a verified prior cursor path with brush lifted, **preserving all previously applied material edits**.
- `RELOCATE_TO_BRANCH`: logical return to an existing saved branch anchor with brush lifted, **preserving material edits**. This is not falsely recorded as a continuous physically travelled/painted line.
- `UNDO_MATERIAL_EDIT`: request a **real, bounded, owner-authorised native Undo**, and independently verify the actual result.

For successfully applied cursor-only reversals, **before and after native material hashes must be identical**, no native material changes may be recorded, `brush_lifted=true`, `preserve_material_edits=true`, and **both native Undo receipt reference and artifact must be null**. For applied native Undo, `brush_lifted=true`, `preserve_material_edits=false`, a positive verified material delta and actual hashed JSON `native_undo_receipt_artifact` plus the authoritative native history/undo receipt ID and readback are mandatory. A cancelled/refused reversal changes neither native materials nor logical cursor. A navigation correction without Undo is not labelled `UNDO_MATERIAL_EDIT`. Existing Workbench history capacity and Stop/revision fences remain authoritative.

### 5. Provenance — `type="provenance"`

Links the experiment ID/arm/split, algorithm name/version/seed/recipe, guidance, **original ordered event IDs**, outcome, metrics, replay result and artifacts. An explicit `evidence_class` differentiates **synthetic** (small record: experiment/arm/split/algorithm/outcome/metrics, never native replay) from **native** (mandatory exact scenario/source commit/tree/base, schema SHA-256, optional model revision/weights/backend, native World bytes, hash/byte-bound native receipt artifacts, final World identity and complete provenance). This allows early algorithm experiments to remain lightweight without weakening actual native qualification.

Artifact `path` is evidence-relative, never absolute or parent-traversing; content must match recorded size/SHA-256. **Native claims cannot pass the semantic checker without an explicit evidence root**; true native acceptance additionally requires external independent source-matched Godot execution, material readback and replay. Metadata-only synthetic fixtures do not establish those facts. Model results, native results, subjective visual evaluation, material PNGs and true framebuffer captures must remain distinguishable. Evaluation outcomes and all failures are retained, including refusals, partial material prefixes and timeouts. `replay="pass"` must mean independently executed native replay, never regenerated JSON alone. A synthetic fixture uses clearly fake SHA placeholders and **must never count as empirical acceptance**.

## Cross-record rules (mandatory beyond JSON Schema)

1. A referenced ID must exist in the episode journal and resolve to the expected type. Unique IDs are never reused with different content.
2. Source and candidate-set hashes must match exact bytes; any action selected on a stale World fence is refused. `sequence` is strictly increasing for events of the same episode.
3. The native World has exclusive authority over materials, protections, focus, revision, Stop and actual changed cells. Soft guidance never grants permission.
4. **No brush-down cursor teleport:** physically meaningful line continuity requires an explicit legal traversal; `RELOCATE_TO_BRANCH` may change logical cursor only with brush up.
5. A `candidate` or `selected` action cannot claim a native receipt/post-state; a `committed` material action must link an actual hash/size-verified receipt **artifact**, pre/post World identities, and post-native material witness. Every action's post-native World fence and material hashes are captured in its corresponding later observation/receipt. **If an operation is interrupted after a partial native write, record the partial prefix and receipt; do not call it a successful cancel without readback.**
6. For action `path`, consecutively painted points are cardinal-adjacent unless the host explicitly uses a no-paint jump; no illegal swept cells or self-intersections where the experiment forbids retracing.
7. The current Stage A/B strict observation contract caps step and history at 64/65, so D's 36-step main study fits. **72/144-step stress studies require a reviewed D-specific adapter**, not silent widening of historical checks. Contract coordinates support bounded non-512 canvases, but this does **not** mean the current native experimental harness supports arbitrary World dimensions without a separately qualified adapter.
8. The **mandatory positive scalar** guidance is distinct from native prohibition. Zero is **neutral**, not a ban. Optional preference and geometric channels have separately versioned meanings, scale, sign conventions and provenance; they never replace or reinterpret core u16. Unsupported channels must be rejected or accounted for as explicitly ignored. See [guidance review](guidance-encoding-review.md) and [A1/A2/A3 comparisons](matched-comparisons.md).
9. Model-image conditions require genuine image-capable inference; the current DRIV3R Q8 Stage A backend uses `--no-mmproj`. Material-generated PNGs must not be labelled graphical framebuffer images.
10. Deterministic policies, random controls and neural policies must receive matched eligible observations and candidate sets, with explicit information ablations separately labelled. Freeze held-out cases **before** model outcome inspection.
11. `interface.schema.json` checks shape only. Cross-record identities, world-size-dependent bounds, provenance custody, native receipts/readback, rollback semantics, value counts and hysteresis inequalities **require semantic validation** before recording milestone acceptance. Synthetic examples carry no native evidence; the checker is not itself an independent Godot/native qualification.

## Independent streams — disjoint source ownership

| Stream | Owns | Does not own |
| --- | --- | --- |
| **D-1A Controller** | `experiments/driv3r/stage_d/direct/**` — 36-decision wall offset and material-Z A→B, geometric/A* and random controls, sequential episodes | Guidance authoring, procedural generator roster, contract schema, shared native/bridge implementation |
| **D-1B Guidance** | `experiments/driv3r/stage_d/guidance/**` — soft u16 field tools, seeded stippling, before/after critic shadow tests, undo proposal validation | Direct cohort, procedural generators, host permission edits, guaranteed visual-backend support |
| **D-2 Procedural** | `experiments/driv3r/stage_d/procedural/**` — Weathering Worm, Branching Growth, Guided Coverage, cursor backtrack/branch relocation, serialized recipes and example corpus | Shared native authority, global model adapter, contract schema, other lanes |
| **Integration/Shepherd (single owner)** | `experiments/driv3r/d_common/**`, shared adapters/native, exact-source qualification, experimental matrix aggregation and formal held-out tests | Model/geometry outcomes cannot be invented, no broad refactors without a demonstrated need |

All three research streams may investigate **independently in parallel** once their candidate contract is available. **Do not integrate or declare conformance until the Owner explicitly approves the v1 freeze.** The central owner handles schema revisions and version bumps after reviewing a bundled change proposal. Nobody silently edits another worker's branch or the #568 painter workers' files.

## Milestone acceptance and rollout

- **M0 — Review candidate (this PR):** one machine-readable schema with five strict record types, normative semantics, a separate guidance-meaning decision review, synthetic examples, adverse tests and a source-pinned baseline. Status `FROZEN CANDIDATE`, not authoritative across workers.
- **M1 — Owner approves freeze:** review the agreed positive-core/optional-typed-channels proposal, dynamic dimensions, three reversals, native/synthetic custody, and matched comparison invariants; record **exact commit SHA + SHA-256 of `interface.schema.json`** in #570 and mark `ACCEPTED/FROZEN v1` **only after explicit Owner approval**. No unreviewed mutation of the accepted bytes.
- **M2 — Independent implementation:** three disjoint Draft worktrees/branches consume the accepted exact SHA and can run CPU/native exploratory work concurrently. A worker can start self-contained tests/preparation before M1, but its shared adapter/interface must be reviewed before integration.
- **M3 — Integration gate:** one integration owner rebases, validates journal compatibility and repeats native Stop/protection/replay/undo tests. Run only *explicit* opt-in model/native experiments with bounded Jobs/VMs, never hourly unbounded training.
- **M4 — Formal comparisons:** preregister paired scenarios and prompts, retain DEVELOPMENT exploration versus sealed structurally distinct held-out tests, execute matched independent Worlds, publish consolidated evidence. Parallelize independent run *execution* but serialize freeze, interpretation and acceptance.

**Deferred:** production/default editor integration, automatic merge/Ready, full node editor, fine-tuning, OpenAI Decisions API expenditure, large GPU models, new inference defaults and unsanctioned scheduling. All require their own Owner approval.

**Proposed owner decision:** accept or refine the exact qualified five-record v1 wire contract, including mandatory positive guidance and optional source-traceable typed channels. A1/A2/A3 are *configurations of this one contract*, not competing codebases. Following Owner freeze, source-pinned D-1A/B/D-2 integration and independent qualified execution continue only under separate gates.
