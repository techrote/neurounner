# D-2 — same-state causal guidance pulses (DEVELOPMENT synthetic shadow)

**Status:** exploratory, offline, post-hoc. No native brush strokes, model inference, host receipts, trained behaviour, aesthetic evaluation or formal held-out claims. This experiment is strictly independent of owner-accepted D-COMMON M1 and of the separately authorised native #611 wormbrain campaign.

## Why this experiment matters

Earlier sustained-escape comparisons changed 46.3% of the combined drawn material footprint but did not show stronger deliberate steering. Long trajectories are chaotic: even a weak preference difference can produce a very different finished sketch. This experiment isolates **immediate responsiveness at exactly the same cursor and decision state**, rather than mistaking divergent drawings for proven controllability.

## Method

Existing, unchanged input-frozen DEVELOPMENT fixtures: six scenes (open, slats, ring, canals, baffled, crucible), two seeds (23, 79). The original strong-attraction x16 synthetic controller is source-replayed at scheduled, non-adaptive zero-based ticks **12, 60, 108 and 156**, generating 48 identical-state comparisons. Each snapshot reconstructs the original legal cardinal catalogue, cursor, heading, visited counters, deterministic oddity, four RNG noise draws and material chain, and independently verifies the exact baseline action against its archived event.

Choose the **left** legal destination relative to incoming heading; if blocked, fall back to right, forward, reverse, in that order. The target does not depend on the baseline winning action. Pulse **only that candidate's positive u16 scalar guidance** by +0 / +2,048 / +8,192 / +32,768, saturating at 65,535, and re-score the **same** candidate set under unchanged deterministic noise. There are 192 total counterfactual *one-decision* records. Every probe returns to the original baseline; no simulated fork advances the material or changes later snapshots.

The probe is intentionally not integrated into a native Workbench driver and does **not** execute UNDO_MATERIAL_EDIT. Soft positive guidance never grants hard permissions.

## Results — 48 matched states per pulse

| Added positive guidance | Target selected | New target selections over sham | Original decision changed |
|---|---:|---:|---:|
| 0 (sham) | 5 / 48 | 0 | 0 |
| +2,048 | 12 / 48 | 7 | 7 |
| +8,192 | 47 / 48 | 42 | 42 |
| +32,768 | 48 / 48 | 43 | 43 |

There is a strong **local response threshold** in this hand-written synthetic controller. Small pulses usually cannot redirect it; large pulses almost always can, though overly strong preferences may overwhelm deliberately odd movement. This does **not** show learned-model steering or artistic improvement. The same path may have a different behaviour after several real native steps; only one action is examined here.

Scene-specific changed-decision counts for the +8,192 pulse (eight scheduled states per scene): open 6, slats 8, ring 8, canals 7, baffled 6, crucible 7. All 48 zero-pulse actions match the baseline exactly. All before/after material states are unchanged in the **counterfactual** (the original baseline itself still has its prior genuine *synthetic* material history).

## Replay/custody

Each case stores source-fenced original state and synthetic original PNG, a separate choice-overlay PNG, full legal menu, scalar guidance hash and selected original guide value, seed/tick/RNG state, original material hash, visited history, baseline choice, requested pulse target, pulsed scalar and selected counterfactual. In the overlay: red marks the cursor; orange marks requested target; cyan marks selected candidate. Source and frozen-development-input digests, file hashes, and null native/model fields are recorded. Replay regenerates all JSON/PNG bytes from exact source; merely rehashing a tampered artifact cannot pass.

Execute cheaply with Python stdlib from this directory:

```sh
python -m unittest -v test_worm_pulse_study
python worm_pulse_study.py --out /tmp/d2-pulse-development
```

On the locally executed Git-matched source, 4/4 tests passed (192/192 probes independently replayed; source/drift checks, zero-pulse sham, tamper-and-rehash rejection). Development report SHA-256: `dd4f3631b720a44ba3d22c0e6bc7687339afb8170e3d3373cf94ca1c9a0b393e`. Source SHA-256: `0cda4a01f9cf10a2fca3d4e8ff1c338fd61aebee6a9169ad6029f02f45e40502`. Full exact-head D-2 suite and the original eight-case frozen-policy source confirmation are still outstanding.

**Next research decision:** test whether moderate short-horizon guidance pulses produce useful line-composition changes on independent scenes, with explicit sham and blind human image assessments. Do not promote immediate synthetic responses into a claim of d1-2B model success, native material control or general artistic superiority. The pre-existing 2/4/8-runner matrix stays inert.
