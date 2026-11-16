# D-2: disjoint-topology guidance/continuity trade-offs (DEVELOPMENT)

**Boundary:** Python-stdlib, offline synthetic only. No DRIV3R model, native Workbench material receipt, native cursor, physical four-neighbour movement, actual material Undo, workflow dispatch, training, paid/GPU call, or new owner gate is implied. M1 owner acceptance remains source `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6` and schema SHA-256 `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`. The separate M3-A one-cell native-backtrack qualification does not authorize native D-2 integration.

## Design, predeclared cases and controls

`pareto_study.py` intentionally does **not** reuse or tweak the Moore guidance/geometry. It constructs seven distinct 32×32 structures with explicit initial material islands, a hard permitted mask, and an independent 16-bit *soft* guidance field: `strait`, `offset-gate`, `archipelago`, `central-obstacle`, `sealed-halves`, `river-gate`, and `guide-detour`. This is a **new exploratory DEVELOPMENT cohort**, not a held-out confirmation or a real native corpus. The sealed wall is an intentional infeasible negative control.

All five policies see the **same** geometry, field, material, and hard mask per case, with a **44-new-cell cap** and a **120-hop synthetic brush-up transfer cap**. Weighted 0/2/8/32 use a deterministic bounded-state Dijkstra search. Each newly painted cell costs 8, plus a configurable penalty for soft guidance below 48,000; existing material has zero paint cost. The fifth arm makes low-guidance *painting* a strict research-only exclusion, deliberately distinct from native permissions. No arm is allowed to pass protected cells. A matched **guidance-first adjacent-frontier baseline** spends exactly the bridge arm's *actual new material cells*, rather than a nominal requested budget. All cases and losses, including incompletes and refusals, are retained. The three reversal meanings stay distinct: only brush-up cursor/branch relocation is synthetic material-preserving; `UNDO_MATERIAL_EDIT` remains unimplemented without a real host-owned history/receipt.

The report preserves first-class original PNGs, five bridge and five matched-control original PNGs per case, exact u16le guidance, seed-free deterministic generator identity, hard-mask hashes, initial/final material hashes, brush-up route records, complete synthetic bridge paths and per-edit before/after hashes. `pareto-study.json` and views are deliberately generated outputs, **not** committed as bulky media.

## Measured exploratory results

Cells column notation is **remaining material components / new cells / below-guidance new cells**; guidance-first baseline shows components at the exact *weighted-0* painted-cell count.

| Structural case | Initial components | Weight 0 | Weight 32 | Strict guidance components | Matched guidance-first components |
|---|---:|---|---|---:|---:|
| strait | 2 | 1 / 18 / 0 | 1 / 18 / 0 | 1 | 2 |
| offset-gate | 2 | 1 / 35 / 15 | 1 / 35 / 15 | 2 | 2 |
| archipelago | 4 | 2 / 34 / 24 | 2 / 36 / 22 | 4 | 4 |
| central-obstacle | 4 | 2 / 33 / 23 | 2 / 39 / 22 | 4 | 4 |
| sealed-halves | 2 | 2 / 0 / 0 | 2 / 0 / 0 | 2 | 2 |
| river-gate | 2 | 1 / 29 / 14 | 1 / 29 / 14 | 2 | 2 |
| guide-detour | 2 | 1 / 18 / 18 | 1 / 34 / 0 | 1 | 2 |

**Core finding:** the topology-oriented algorithm does generalise to several structurally different *small synthetic* settings, while explicitly **failing** to connect all archipelago and central-obstacle islands under the 44-cell cap. In `guide-detour`, the cheap direct link spends 18 new cells, all below threshold, while the stronger guidance penalty chooses a 34-cell preferred-field detour with no below-threshold cells. The strict arm also connects that case, but fails to connect `offset-gate`, `river-gate`, `archipelago` and `central-obstacle`. This is a genuine **length-versus-guidance-versus-connectivity trade-off**, not a universal recommendation to override guidance. The isolated `sealed-halves` case remains unconnected even under weight 0; no illegal bridge was fabricated. Same-budget guidance-first growth did not connect any of these seven structures.

**Interpretation caveats:** the fixtures are deliberately engineered diagnostic geometries and include a deliberately engineered detour. They have not been reserved as an unseen FINAL split, independently visually judged, or assessed on CyberSand's native permissions, texture, runtime, physics or real image quality. Frontier membership also includes do-nothing policies with zero cost, which should not be confused with useful connected outcomes. The fixed 44-cell cap may favor short links, and the greedy baseline is a credible local heuristic, not the provably best coverage baseline.

## Reproduce, provenance and future qualification

From `experiments/driv3r/stage_d/procedural`:

```sh
python -m unittest -v test_pareto_study
python pareto_study.py --out /tmp/d2-pareto --budget 44
sha256sum /tmp/d2-pareto/pareto-study.json
```

Source-local qualification: 13/13 Python stdlib tests passed, including independent PNG CRC/decompression, live-event material hash reconstruction, permitted-mask refusal, closed-wall negative control, strict guidance contrast, equal-*actual*-cell control and two identical independent output directories. Repeated seven-case report SHA-256: `4b8d837e5380354314ead7adfa0e390491019e5cd9ec9c119fb7ccd6e6abe4bb`. Local source SHA-256: `pareto_study.py` `1ef0005e605c870aedd4d7a5a3f62777f376eae3c5060bebfa3905870486cf6b`; `test_pareto_study.py` `05e0e31f2af44de5360ed9fadc97d433efd4081d02f95fa019ef6875368b7784`. Once committed, verify the Git blob identities and run the *exact committed HEAD* tests before treating this as repository-qualified. No workflow has been dispatched.

**Next gate:** blinded, separately frozen, structurally distinct confirmation geometries and a legitimate multi-objective score combining material utility, guidance consistency, route and edit budgets; then human visual assessment. Do not promote this diagnostic cohort to native/model/art acceptance.
