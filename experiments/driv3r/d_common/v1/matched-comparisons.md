# DRIV3R D-COMMON-001 — paired A1/A2/A3 comparison protocol (design only)

**Status: RESEARCH PLAN / NO EXPERIMENT AUTHORIZATION.** All three arms use **the same versioned five-record contract** and the same host-owned World, focus, material permissions, Stop and native fencing. This document defines comparisons and evidence to collect **after** explicit M1 approval and separate experimental dispatch approval. It makes no performance or artistic claims.

## Three configurations, one guidance record interface

| Arm | Required baseline | Optional typed channels | Question |
| --- | --- | --- | --- |
| **A1 — simple** | `guidance.encoding=u16le_higher_is_better_zero_is_neutral`; `optional_channels=[]` | None | Is the scalar user-highlight intensity sufficient for recovery? |
| **A2 — rich preference** | Same baseline raster and interpretation as A1 | Independent `soft_attraction/v1`, `soft_avoidance/v1` and optionally `flow_vector_xy/v1`; every channel separately encoded/provenanced | Does additional attractive/repulsive/directional information help? |
| **A3 — preference + geometry** | Same baseline raster and interpretation as A1 | Separate `signed_distance/v1` to a pinned named boundary and optionally `flow_vector_xy/v1`, each `signal_class=geometry_metric` | Does explicit separation of preference from geometry improve reliability? |

A2/A3 may deliver **additional information**, not just alternative encodings. To isolate *representation* from *information advantage*, include matched-information ablations that compute the same numerical facts in a control encoding or host geometry metric, without changing task legality. Distinguish scalar-core-only baselines, added-content arms, and matched-content recodings. All interpretations are declared before evaluating held-out data.

**The base field is compulsory within any guidance record** and cannot be changed by optional fields. With no guidance at all, an observation may use `guidance_ref=null` and must declare no optional channels consumed or ignored.

## Frozen legal-candidate parity

At each *paired identical native state* (same world dimensions, actual source/readback material hashes, cursor and hard permission/focus set), the **host**, not the model, generates exactly the same legally admissible recovery candidates in every arm. Preserve the complete original candidate catalog and identifiers in the experiment evidence.

- `action.legal_candidate_set_sha256` is SHA-256 over canonical, ID-sorted host-native legal candidate IDs and exact permitted action footprints, destinations, material operations and hard limits. **Exclude user guidance values, guidance-derived rankings, model prompt wording and candidate display order.** Those may vary across A1/A2/A3 without changing what is legal. The hash must match for all identical pre-decision native states.
- `action.candidate_set_sha256` remains the full model-facing candidate presentation hash. It may differ because available guidance or derived scores differ. Never confuse these two hashes or claim parity from the display order.
- `observation.guidance_channel_handling` reports the exact optional channels consumed, or those explicitly ignored (with semantic ID, SHA and reason). No channel can disappear silently. The scalar baseline remains intact whenever a guide record exists.
- Once independently evolving runs **choose different actions and their native states diverge**, legal candidate sets are recomputed for each resulting World. It would be incorrect to force identical candidates on now-different geometry. Continue to record original choices and all actual native receipts, including divergent failures.

## Divergence and recovery leverage

Measure and retain **two distinct divergence points**, not only an image difference:
1. `first_choice_divergence_index`: first paired step where valid offered candidate IDs differ, on an otherwise matched native state and legal catalog. This can be evaluated in frozen *shadow probes* without native painting and is **not necessarily consequential**.
2. `first_consequential_divergence`: first host-observed change in **actual committed cursor trajectory, material footprint, recovery outcome or subsequent procedural continuation**. Verify from native receipts and readbacks, not a model's prediction or a choice label. Two different candidate IDs can produce equivalent material effects.

Also retain `productive_procedural_steps_after_recovery` (host-verified, nonredundant deterministic drawing steps unlocked after each model recovery intervention), `recovery_leverage` = productive steps / model interventions, total material change, path coverage, loops, self-intersections, actual material undo versus cursor-only backtracks, latency and resource costs. Record first difference in source/material PNG and genuine graphical framebuffer captures separately where each exists; never label material-derived PNG a frame capture.

Run independent matched native Worlds per arm on identical source/fixture/seed/goal and source-pinned model/runtime/decision budgets; keep deterministic strong geometric and random-legal controls. The A1/A2/A3 configurations should be **parallel-executed** but formal paired analyses must use frozen cases, source identities, objective metrics and a structurally distinct held-out confirmation set. Existing D2 exploratory assets and failures are valid DEVELOPMENT corpus, **not** automatically sealed held-out evaluation.

## Boundaries

This contract proposal adds no actual DRIV3R inference, native actions or scheduled A/B workflows. All examples and contract checks are synthetic. M1 owner freeze must precede cross-worker integration. A subsequent M3 native qualification must pass real source-matched permission/Stop/history/readback/replay before formal M4 comparisons run. Source and corpus hashes, ignored-channel policy and all adverse trajectories remain retained. No paid OpenAI/GPU endpoint, training, merge/Ready or production-default changes are authorized.
