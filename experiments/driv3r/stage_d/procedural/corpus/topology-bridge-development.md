# D-2 DEVELOPMENT: material-topology bridges versus soft-guidance continuity

**Status:** offline synthetic counterfactual only; no native execution, model inference, paid work, physical travel or Undo. This is a subsequent D-2 lane-owned experiment following the existing route study, not new authorisation or an M1 schema modification.

## Experimental design

The new topology_study.py reuses the three *original* Hilbert, Moore and serpentine DEVELOPMENT cases, their source recipe, ordered curve, original u16le scalar guidance, original PNG provenance hashes, and replayed route-study controller. A synthetic four-neighbour material audit reconstructs every paint step. It reserves 24 of the original 160 material-cell budget, leaving a shared 136-cell prefix. It then compares:

- **Original reference:** continue curve to 160 cells.
- **Equal-cell curve control:** continue the identical curve prefix to precisely the same *actual final cell count* as topology bridging, not merely the same nominal step limit.
- **Topology bridge, override soft guidance:** choose minimum-new-paint material bridges with deterministic heap priority and bounded synthetic brush-up routing; no protected-mask penetration, no material Undo, no fake native transport.
- **Topology bridge, strict guidance:** treat the original exit_below threshold as an additional *artistic* paint exclusion. This is deliberately more restrictive than the owner-approved protocol's soft guidance; never confuse guidance values with host permissions.

The five resulting views per case are original-palette synthetic PNGs, byte-hashed in topology-study.json alongside original before/midpoint/after image hashes, input episode custody, event hashes, edit budgets and route events. Generated images and reports are **not committed**; rerun the script to regenerate the corpus.

## Results, original three-case DEVELOPMENT cohort

| Pattern | 136-cell prefix components | Original 160-cell reference | Equal-cell curve control | Soft override topology bridge | Guidance-strict bridge |
|---|---:|---:|---:|---:|---:|
| Hilbert | 1 | 1 (160 cells) | 1 (136) | 1 (136; no bridge) | 1 (136) |
| Moore | **3** | **4** (160 cells) | **3** (144) | **1** (144; 8 new cells, 31 synthetic brush-up hops) | **3** (136; no permissible bridge within budget) |
| Serpentine | 1 | 1 (160 cells) | 1 (136) | 1 (136; no bridge) | 1 (136) |

**Critical qualification:** all **eight new Moore bridge cells are below the source guidance exit threshold (48,000)**. Connecting three components at equal material-cell count is therefore a real synthetic topology improvement **only if soft-guidance adherence can be overridden**. Enforcing the threshold preserves three components. Do not report the one-component result as a free improvement, native legal action, or generally preferable image. The study intentionally exposes the cost rather than hiding it.

## Reproduction and verification

From the D-2 directory in an exact-source checkout:

    python -m unittest -v test_topology_study
    python runner.py --out /tmp/d2-qualified-corpus
    python topology_study.py --corpus /tmp/d2-qualified-corpus --out /tmp/d2-topology-report --reserve 24

The unittest module creates its own disposable eight-case development corpus with existing runner.py if qualified-corpus/ is not available. Adversarial tests cover blocked material corridors, insufficient edit budgets, no physical route within the synthetic hop cap, forged per-event paint hash, same-source deterministic repeats, and independent PNG decode/roundtrip. No GitHub Actions experiment workflow is installed or dispatched.

**Next candidate:** an explicitly budgeted multi-objective *soft* guidance/continuity Pareto sweep, including bridge penalties and new unseen topologies, with source-pinned examples and owner-reviewed visual quality. Preserve strict legal host constraints independently. Only an independently authorised future M3/M4 integration can establish native movement, material receipts, real Undo, model benefit or generalisation.
