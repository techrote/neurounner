# D-COMMON-001 — guidance field semantics and A1/A2/A3 concurrent research choice

**Status: PROPOSED FINAL M1 CONTRACT; EXACT SOURCE/SHA APPROVAL PENDING.** Following discussion, the Owner explicitly chose a **representation-neutral, extensible single v1**, retaining `u16le` positive scalar preference (zero neutral) as the mandatory baseline **and optional named, typed channels**. **A1, A2 and A3 are configurations for subsequent concurrent research, not separate incompatible interface versions.** This note retains the earlier alternatives as design rationale; the final freeze still requires explicit Owner approval of the *new exact source and schema hashes*. No native A/B jobs or cross-branch integration are authorized by the guidance decision.

All choices below are **soft guidance**. Neither any positive nor any negative hint is a native material prohibition, collision rule, focus extension or permission grant. Protected-material constraints always reside in the host-authoritative `observation.rules` / native admission.

## Comparison

| Proposal | Meaning and suggested encoding | Advantages | Risks / costs |
|---|---|---|---|
| **A — Positive scalar desirability (current candidate)** | 0 = neutral; max = maximally preferred. Little-endian `u16` per pixel; intensity renders as seeded sparse→dense speckles | Simple brush, fixed-point replay, small footprint, naturally supports highlighted path/target and scalar scoring | Cannot express dislike without a separate host constraint/penalty; zero may be mistaken for prohibition unless explicitly called neutral; density plateaus cannot encode travel direction |
| **B — Signed scalar preference** | Negative = discourage; zero = neutral; positive = attract. Could use signed `i16` or pair of nonnegative `u16` layers | Elegant combined attraction/repulsion for corridors, n-flakes, erosion limits | Requires declared scale and clipping, semantic "repulsion" never a hard mask; signed combination may allow unrelated positive and negative preferences to cancel |
| **C — Multiple named nonnegative channels** | `u16` attraction, optional `u16` avoidance, optional target and age masks; combine via explicitly versioned recipe | Prevents cancellation ambiguity; per-channel inspection and ablation; configurable analogue-synth routing | More data, more ports, runtime complexity; blend rules and priorities must be defined and versioned |
| **D — Signed distance field** | Integer signed cell-distance from a boundary (e.g. `i16`), explicitly specify interior/exterior sign and distance transform; derive a 6–10 pixel band preference | Especially suited to wall offset, erosion depth, reentry hysteresis and contours; makes objective geometrically interpretable | Not inherently "how desirable" a point is; gradient/brush rendering is derived; overflow/transform direction and multiple walls need metadata |
| **E — Vector/flow guidance** | `(dx,dy)` steering vector per cell, e.g. two signed fixed-point channels and magnitude or unit convention | Encodes clockwise/counter-clockwise contour following, planned routes, winding paths, worm heading | Larger records, undefined vectors at junctions, not a substitute for scalar "goodness" or legal path checks |
| **F — Greyscale/colour/alpha PNG as primary** | Image channels interpreted as guidance intensities or labels with pinned decoder settings | Human-friendly import and editing | Colour-space, gamma, alpha handling, quantisation and speckle stochastic gaps make raw images less reproducible as primary semantic data |

## Selected v1 interface proposal

**Mandatory baseline:** source-hashed unsigned 16-bit little-endian scalar preference (0 neutral, 65535 maximally preferred), with optional seeded speckle visualisation. It is always present in a guidance record, including A2 and A3.

**Optional channels:** 0–8 typed channels, each with its own named `semantic_id`/version, `signal_class` (preference or geometry metric), encoding, rational numeric scale, units/zero/sign meaning, optional referenced boundary, independent tile size/canvas, SHA-256 of underlying samples and source producer/parameters/seed provenance. Known v1 encodings: `u16le` scalar, `i16le` signed scalar and `vec2_i16le` signed X/Y vector. Future semantic IDs require explicit consumer registration; unsupported channels cannot be silently interpreted, dropped or given native authority.

| Variant | Required positive core | Optional meaning |
| --- | --- | --- |
| **A1** | Same baseline raster | None; simple soft attraction |
| **A2** | Same baseline raster | Separately typed `soft_attraction/v1`, `soft_avoidance/v1`, optional `flow_vector_xy/v1` |
| **A3** | Same baseline raster | Separately typed `signed_distance/v1` (`geometry_metric`) and optional `flow_vector_xy/v1`; retain a pinned original boundary identity |

The separate `guidance_channel_handling` field in each observation declares used and ignored channel IDs, exact ignored SHA/semantic identities and a reject/explicit-ignore policy. Built-in meanings are fixed by a runtime registry and semantic validators. All optional channel bytes join the exact native provenance inventory. Synthetic-only exploration remains lightweight.

**Comparison safeguards:** Same host-owned legal recovery candidates and same numeric native starting World; guidance-derived rankings/prompt presentation may differ but must not change the hard legal candidate catalog. The contract uses a distinct `legal_candidate_set_sha256` for that parity, separate from the model-facing candidate-set hash. Record the first differing choice **and** the first truly consequential cursor/material/procedural divergence from native readback, plus productive deterministic steps unlocked per model recovery. After trajectories diverge, recalculate legality independently. See [matched-comparisons.md](matched-comparisons.md).

**Research caution:** A2 and A3 often supply *additional information*. Controls must separate information content from representation effects. A signed-distance field expresses geometry, not automatically desirability; a flow vector expresses orientation, not permission. A future node/synth editor may promote independent channels to signal ports, but its design is not part of this M1 contract.

## Semantics invariant across options

- Source World coordinate frame is integer XY, origin upper left, row-major raster. Every guidance record states its **canvas_size** and bounded frame. Field dimensions must match the referenced World when consumed.
- Numeric samples (mandatory core and each explicitly typed auxiliary) are authoritative for numeric experiments; PNG/speckles are display/evidence and may differ due to seed, gamma or sampling, with source converter/hash recorded.
- A preferred region threshold is a suggestion to consider a recovery candidate. The native host still decides legal operations, materials, focus, protected regions and Stop.
- Long-horizon comparisons must account for extra information exposed by a channel and retain equivalent controls and frozen interpretations. Typed conversions and ablations are recorded rather than treated as prompt paraphrases.
