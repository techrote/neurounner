# DREAMGRAFT DG011 — heterogeneous neural donors and calibration drift

**Status:** research only; **Draft/HOLD**, unmerged. Game target is future CyberSand game-policy/authoring behaviour. No game code, maps, materials, Godot/native/Workbench components or production providers are modified.

## Hypothesis and falsifier

DG010 used same-architecture parents in family-labelled slots and a fixed calibration probe bank. DG011 asks whether a neural receiver can inherit the *combined* behaviour of multiple independently trained donor architectures despite **hidden donor roles, random per-query donor ordering, changed probe worlds and previously unseen donor weights/architecture**.

The resulting policy must select host-legal actions in synthetic material/obstacle/paint/retrace worlds under a law combination **never used for receiver training**, and respond causally to changing only one donor. A held-out, completely unobserved three-way interaction remains a deliberate identifiability negative control.

## Experiment

- Three genuinely separate architectures: **narrow two-layer MLP**, **wide two-layer MLP**, and **gated MLP**. Each architecture bank contains six independently trained neural specialists: two signs for wet/material, two for branch/retrace, two for painting. Across all banks: **18 trained donors per seed**, followed by one DeepSets receiver and an equal-architecture law-blind control.
- Donors are trained with synthetic authored *atomic-law* utility targets. The receiving policy never sees the donor law sign, donor family label, authored action oracle or architecture identity. It receives only output signatures extracted from frozen neural donors.
- Training source models are **narrow and wide**; sealed source is the separately initialized **gated** network, never included in receiver training. Each bank has independent weights/training worlds.
- The receiver's donor encoder is **permutation invariant** (shared encoder + mean pooling). Random independent donor-slot permutations are evaluated as a strict no-change test.
- 7 of 8 three-law sign triples are trainable; the (+,+,+) triple is **sealed**, with new game worlds, new donor architecture and unseen probe worlds.
- Compare two equally sized source channels: **raw individual probe logits** versus an **interventional functional fingerprint**. For every input feature, alter that feature 0→1 in probe worlds, measure the frozen donor's genuine six-action output changes, then average across probe worlds. Both channels supply exactly 60 numbers per donor. This uses *known synthetic observation axes*, not oracle law bits.
- Each condition: 5,200 training worlds; 500 validation; 900 sealed; 220 donor optimiser updates per specialist; 750 receiver updates. Two seeds, two signature methods = four full experiment arms.
- Controls: same-capacity law-blind receiver; zero donor; changed probe-world bank; unseen donor architecture; donor-slot permutation; identical-world opposite wet donor (compare original and counterfactual law labels); unidentifiable three-way law.

## Independent local CPU training (before GitHub Actions)

All four local arms used the exact algorithm/settings above and saved actual neural weights, reports, 64 original action decisions and SHA-256 manifests. These are **local experimental findings**, not yet claimed as GitHub CI qualification.

| Seed | Donor signature | Gated donor + familiar probes | Gated donor + unseen probes | Blind control | Wrong-wet donor matches *counterfactual* action, unseen probes |
|---|---|---:|---:|---:|---:|
| 26011618 | **Interventional** | 65.11% | **65.67%** | 35.00% | **93.33%** |
| 26011618 | Raw logits | 62.78% | 28.33% | 35.00% | 68.56% |
| 26011619 | **Interventional** | 64.11% | **63.56%** | 38.89% | **92.00%** |
| 26011619 | Raw logits | 59.89% | 66.11% | 38.89% | 17.22% |

Across two seeds, interventional fingerprints retain useful counterfactual-law sensitivity under *both* donor architecture and probe-bank changes. The raw-logit method is unstable: one seed's original action accuracy collapses after calibration shift; the other seed achieves good original-law accuracy but its wrong-donor counterfactual response collapses. **Do not average away these different failure modes.**

All 900 sealed outputs/condition remain host-legal by construction. Source-slot permutation disagreement: **0** for both methods and seeds (the invariance is architectural, not learned). The entirely unseen +++-only third-order interaction is **not learned** and is not identifiable from atomic or pairwise examples; report its oracle-action match and change rate as negative evidence.

## Run locally

From repository root:

```bash
python3 -m pip install 'torch==2.5.1' 'numpy>=1.26,<3'
python3 -m experiments.memplant.neural_donors_011.experiment --selftest
python3 -m experiments.memplant.neural_donors_011.experiment --seed 26011618 --signature-mode intervention --out /tmp/dg011-intervention-18
python3 -m experiments.memplant.neural_donors_011.experiment --seed 26011618 --signature-mode raw --out /tmp/dg011-raw-18
```

GitHub Actions matrix covers both methods and two seeds, records source hashes, and retains the original 18 donor weight files, receiver/blind weights, predictions, metric report and manifest in downloadable workflow artifacts rather than Git history.

## What this proves / does not prove

This is a **small, supervised synthetic neural functional-state inheritance** result: outputs from independently trained donor networks can causally steer a separately trained receiver for a withheld game-law *combination*, even when the donor network architecture and calibration examples change. The intervention fingerprint benefits from accessing and manipulating the known **observation feature axes**. This is **not** native KV transfer, learned discovery of entirely unknown physics, unsupported cross-family transformer state translation, or real CyberSand gameplay performance. Evaluation reports *single-decision exact-action imitation*, not long-horizon game success.

Next: withhold observation **semantics** rather than only donor architecture, and train receivers to infer feature interactions from a few source-driven probe interventions rather than assuming the synthetic observation axes have stable meaning. Move toward sustained stateful game trajectories and independent native read-only trace comparison. No game-side edits permitted.
