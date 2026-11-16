# D-2 synthetic route-recovery counterfactual — DEVELOPMENT (14 Nov 2026)

The standalone `recovery_study.py` consumes the **original eight-case development corpus** written by source `19347c00` (replay-audited earlier). It checks all input artifact bytes/hashes and reads original recipe, ordered curve, u16le scalar guidance and source-material record. The three original guided-coverage episodes are re-simulated under four deterministic arms using **identical original geometry, guidance, cursor start and stroke budgets**: `virtual_future` (historical result; no physical route), `routed_future` (cardinal brush-up BFS through a synthetic allowed-cell mask), `tile_restart` (nearest eligible 16-curve-index boundary with routed seek), and `stop`. This is not a neural A/B experiment and the conditional recovery trajectories are offline counterfactuals, not newly acquired native observations.

| Curve | Arm | Painted cells | Components | Brush-up reentries | Routed steps | Terminal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Hilbert | virtual / routed | 160 / 160 | 1 / 1 | 6 / 6 | 0 / 19 | Budget |
| Hilbert | tile restart / stop | 20 / 20 | 1 / 1 | 0 / 0 | 0 / 0 | No verified reentry / Stop |
| Moore | virtual / routed | 160 / 160 | **4 / 4** | 13 / 13 | 0 / 64 | Budget |
| Moore | tile restart / stop | 52 / 16 | 5 / 1 | 4 / 0 | 22 / 0 | No verified reentry / Stop |
| Serpentine | virtual / routed | 160 / 160 | 1 / 1 | 4 / 4 | 0 / 11 | Budget |
| Serpentine | tile restart / stop | 144 / 0 | 4 / 0 | 4 / 0 | 34 / 0 | No verified reentry / Stop |

**Finding:** under a fully traversable synthetic canvas, adding a legal cardinal brush-up path does **not** improve disconnected material topology at all. The route only legitimizes **cursor motion**, not material continuity. Moore still yields four isolated painted components. Tile-boundary restarts can increase fragmentation and leave much less completed work. Therefore improve *material-topology-aware reentry/bridging*, not only shortest-path transport.

**Limitations:** `routed_future` is verified only against supplied synthetic in-bounds allowed-cell masks; it cannot assert real Workbench navigation, native permissions/collision/Stop/cancellation or receipt custody. No host-owned material Undo exists here. `virtual_future` material hashes match the original three saved final development images; this supports a controlled baseline. A blocked-mask test explicitly refuses a route. `recovery_study.py` and `test_recovery_study.py` are deliberately standalone for cheap review, with no GitHub Actions job installed/dispatched. The test suite generates an ephemeral eight-case corpus through the existing source-exact `runner.py` if `qualified-corpus/` is absent beside the files; no binary fixture is committed.

Reproduce from the D-2 subtree: `python runner.py --out /tmp/qualified-corpus`, then `python recovery_study.py --corpus /tmp/qualified-corpus --out /tmp/d2-route-study.json`; `python -m unittest -v test_recovery_study` builds and destroys its own temporary corpus if none is present beside the module. Preserve the original first-pass failed trajectories. These **development** results cannot justify native/model dispatch, A1/A2/A3 superiority, training, Ready or merge.
