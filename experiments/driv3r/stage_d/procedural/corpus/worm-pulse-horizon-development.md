# D-2: one-turn steering pulses and 32-decision drawing continuations

**15 November 2026. DEVELOPMENT / offline synthetic. PR #598 stays Draft/HOLD.**

## Question

Does a brief soft-guidance intervention just change one host-legal choice, or does
it produce sustained differences in a drawn line? This follows the owner's
*slightly deranged steerable worm* objective, rather than optimisation against A*.

## Matched design

Use the 12 already source-frozen 80×80 scene/seed combinations (six
geometry families × two seeds) and four specified intervention ticks
(12, 60, 108, 156). Compare one-turn target-cell u16 guide pulses of
0 (sham), +2,048, +4,096, +8,192 in an otherwise unchanged
strong-attraction ×16 hand-coded worm. All 192 forks continue for 32 further
cardinal decisions with the original guide restored; preserve the entire
pre-intervention painted material, visited state, heading, pen rhythm, legal
candidates and tick-indexed random sequence. Source-code replay asserts exact
agreement with the committed original full 192-step strong-arm baseline
before any fork is evaluated. These are reused DEVELOPMENT inputs, **not an
independent FINAL cohort or preregistered confirmation**.

Each case stores original/baseline/fork/difference rasters, complete synthetic
step and material-hash traces, input/guide/case/algorithm provenance, explicit
null native/model receipts and independently source-rebuilt replay checks.
No material Undo is claimed or performed; no cursor teleport or protected
material mutation is allowed.

## Results (48 matched states per arm)

| Added u16 guide | First decision changed | 32-step cursor mismatches | New painted positions absent from baseline | Baseline new paint omitted | Different end cursor |
|---|---:|---:|---:|---:|---:|
| 0, sham | 0 | 0 | 0 | 0 | 0 |
| +2,048 | 7 | 204 | 53 | 56 | 5 |
| +4,096 | 27 | 811 | 334 | 365 | 24 |
| +8,192 | 42 | 1,218 | 519 | 539 | 36 |

Figures sum separate, matched 32-step fork comparisons; the same pixel may be
counted in different cases. They are NOT the area of one aggregate drawing.
A zero pulse exactly reproduces all 48 original trajectories, events and
synthetic materials. In every case where the first decision is unchanged,
the entire continuation is identical. When it changes, subsequent steps often
continue diverging even though guidance reverts immediately.

**Interpretation:** relatively moderate individual turns can materially
reshape a compact worm drawing; this is controllable causal influence over
a deterministic simulation, not demonstrated aesthetic improvement. Replaced
painted positions are as important as extra positions. Stronger pulses create
more change but can also erase a line's original compositional character.

## Reproduction and boundaries

From this procedural directory:

    python -m unittest -v test_worm_pulse_horizon_study test_worm_pulse_study
    python worm_pulse_horizon_study.py --out /tmp/d2-pulse-horizon

Verify each case with verify(path); source files and artifacts are
SHA-256 recorded. Input chronology and older source blobs are unchanged.

The five new regression tests cover source-replayed baselines, all sham cases,
material/protection legality, null native histories, exact case count, missing
or unregistered cases and tampered+rehash artifact rejection. Together with
four prior pulse tests, **9/9 selected tests passed locally** against exact
source blobs; the **full exact-head D-2 suite remains outstanding**. Evidence
has no native CyberSand/Workbench material acknowledgements, learned
d1-2B choices, inference, human visual scores, host-owned Undo or production
implications. No experiment-matrix dispatch or paid compute occurred.
