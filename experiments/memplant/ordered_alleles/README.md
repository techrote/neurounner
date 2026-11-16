# DG005 — Ordered game-state alleles (research-only)

**Draft/HOLD. This research does not edit CyberSand game code, content, native simulation, Godot or Workbench.**

## Question

Can a small trained receiver learn two **state-dependent, noncommuting** game-law edits independently, then correctly compose them in **untrained edit orders**? Does a donor source-state channel make it useful for downstream host-legal game actions?

This extends [DG004 #236](https://github.com/techrote/neurounner/pull/236): DG004 tests a trained *static* compositional game-action receiver on XOR/XNOR. DG005 tests **sequential counterfactual edit dynamics** using the same prior [DG002](https://github.com/techrote/neurounner/pull/233) original source-Qwen attention-V archive.

Source artifact: Actions [43736088404](https://github.com/techrote/neurounner/actions/runs/43736088404), named `memplant-dg002-source-a1`, real attention-V hash `473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75`. Verify full archive manifest, payload digest, dimensions and source model revision before training using DG004 donor loader. **The inherited channel is two supervised ridge scalars, not real cross-model native KV injection.**

## Experiment

Initial game state has heat (0/1) and vibration (0/1). For state `(h,v)`:

- `H(h,v) = (1-h, v XOR h)`: the heat intervention also changes vibration conditionally.
- `V(h,v) = (h XOR v, 1-v)`: the vibration intervention also changes heat conditionally.

For three of four initial states `V(H(s)) != H(V(s))`. A constant additive allele vector is strictly commutative and fundamentally incapable of encoding this order dependence.

Three **real AdamW models** are trained: a two-scalar Qwen-derived source decoder, an MLP contextual atomic edit receiver, and a fixed additive-delta editor. The edit models see only single H or V transitions, with varying input confidence and all four law states. **No two-step or three-step labels are used for training or model selection.** Six compositions—HV, VH, HHV, VVH, HVH, VHV—are evaluated on 24 sealed DG002 source cases (six families, previously inspected in earlier research, therefore exploratory).

Controls: actual source-derived scores, zero source, within-test shuffled source, true-bit oracle initial state, and complete symbolic oracle. Compare exact **joint-law** correctness, individual bits, the expected-vs-observed HV/VH order change, and legal action decisions from fixed, source-independent CyberSand-inspired external menus. Downstream action scorer uses a **public scripted truth-table gate and legal-mask projection**; it is not a separately trained action model. The same candidate menu is reused for both edit orders.

Each of two CPU seeds produces three small `.pt` checkpoints, SHA-256 identities, learning curves, full aggregate controls and representative raw decision receipts under 30-day Actions artifacts. No training, model weights or game content are committed to Git. If source readout is weak, a poor result is retained rather than rescued with sealed labels.

## Reproduction

```sh
python3 -m experiments.memplant.ordered_alleles.experiment --selftest
python3 -m experiments.memplant.ordered_alleles.experiment \
  --source runtime/memplant/dg005/source \
  --out runtime/memplant/dg005/results \
  --seed 26011605 --updates 600
```

**Interpretation:** A positive result demonstrates *supervised law-state decoding and compositional learned edit operators*, not self-discovered physics, general compressed memory, genuine cross-architecture KV transplantation or native CyberSand behaviour. The result remains Draft/HOLD, with no merge or production admission.
