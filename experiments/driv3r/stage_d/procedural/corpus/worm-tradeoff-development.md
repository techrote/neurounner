# D-2 — post-hoc worm steering / composition trade-off

**Status:** DEVELOPMENT / POST-HOC SYNTHETIC DIAGNOSTIC / Draft-HOLD. No native painting, model decisions, Undo receipts or aesthetic acceptance. Input cohort (12 frozen scene/seed pairs × five arms) was previously generated at D-2 head `99be16ccfacb7f9dcd841e0273361750e8ddebee`; these secondary metrics were selected *after* observing that cohort.

**Custody:** locally archived `worm-horizon-evidence.zip` SHA-256 `940d21f6977d0139c18da9a4e6fe728c5e8151551f60883393eff1d655ae8864`. The four archived source modules match GitHub Git blobs on that exact head: horizon study `9527a438`, fixture `9ebd7af5`, sketch `b419282d`, scent `008ff421`. Frozen input manifest SHA-256 `f0e0446d0fc200c36b034b9d55277c51f5272066420ed89099154d9bdf56e3e6`. All 60 cases are individually re-simulated by the original exact-source replay verifier before analysis.

## Results

| Paired metric: sustained escape relative to strong scent ×16 | 12-pair result |
|---|---:|
| Net change in distinct painted cells | **+136** (1,367 → 1,503) |
| Newly painted locations absent from baseline | **500** |
| Baseline painted locations omitted from sustained result | **364** |
| Aggregate painted-location overlap (intersection/union) | **1,003/1,867 = 53.7%** |
| Repeated cursor visits | **−171** (789 → 618) |
| Sum of per-episode mean guide advantages (u16) | **−548**; decreases in 9/12 |
| Locally maximal-guidance choices | 810/2,304 baseline vs 812/2,304 sustained |
| First divergent cursor step | 11/12; baffled seed 79 unchanged |
| Net coverage outcomes | 10 positive, 1 negative, 1 unchanged |

The nominal +136 coverage gain hides **500 changed-in and 364 changed-out material locations**: sustained escape frequently redraws a different composition, rather than merely adding ink to the earlier line. The average guidance advantage drops, while the frequency of choosing a locally highest-guidance move remains approximately unchanged. This is an important limitation of treating increased coverage as improved *steerability*. The baffled seed 23 negative case (−7 cells) and baffled seed 79 no-change case remain visible.

**Nonclaims:** this is post-hoc exploratory measurement, not preregistered confirmation, statistical proof, meaningful native topology, physical art quality, controlled guidance A/B on matched *post-divergence* Worlds, or model performance. Scalar guidance remains a preference and never overrides legal permission.

**Offline reproducibility:** run `python -m unittest -v test_worm_tradeoff_audit.py` (six regression tests; missing-case and rehashed-artifact tamper rejection) and `python worm_tradeoff_audit.py --corpus <unpacked-evidence/evidence> --out tradeoff.json`. The corpus source archive remains an optional local input; no GitHub Actions experiment matrix runs. Repeated canonical report SHA-256 `dbaf1f91b7ae3dd16a3efa5d6d6ea8301ef805426f75840aa20e37ca1a0784aa`. The immutable M1 owner-approved common contract remains untouched.

**Next:** run full exact-head D-2 suite and source-verify the older eight-case frozen-policy confirmation separately; obtain human visual judgment before artistic conclusions. No inference, API/GPU spend, training, native integration, native Undo, merge or Ready promotion was authorised or performed.
