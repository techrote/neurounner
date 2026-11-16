# DG010 — Can neural donors transmit usable game-law state?

**Research classification:** synthetic, supervised, independent-neural-donor state transfer into a separately trained action policy. **Draft/HOLD**. No CyberSand runtime or content changes.

## Why this experiment exists

DG009 [#242](https://github.com/techrote/neurounner/pull/242) transferred information from **authored demonstration actions**, not from separately trained neural donors. DG010 replaces those authored demonstrations with **actual trained donor neural networks**. It asks whether a receiving policy can use frozen specialist activations to select correct legal actions under a law **combination never represented in receiver training**.

## Three independently trained specialist families

- **Wet/material** — signed attraction or avoidance of wet neighbours.
- **Branch/retrace** — signed preference for branching versus retracing.
- **Paint** — signed material-sensitive painting preference.

For each family, train two signed expert-regression donors (+/-), using fresh random initialisation and independent sample worlds. Train a **separate bank of six donor networks** for final evaluation, with disjoint parameter seeds and independently sampled supervision. Thus each DG010 seed trains **12 real donor models** before training the receiver models.

Donors are trained against explicitly authored *single-family* synthetic utility functions. Their frozen states are then probed using an invariant ten-world calibration bank. Transmitted numerical tensors contain 60 per-action **neural output logits** plus 48 **actual penultimate activation statistics** (mean/std over the calibration bank) per donor. Three donors are provided in known wet/branch/paint slots. Neither true law-sign labels nor authored utility scores are ever provided directly to the receiver.

Receivers are trained on seven out of eight signed triple combinations; **(+,+,+) is excluded from training and validation**. Evaluation uses 900 independent query worlds with the excluded triple and a separate set of neural donor checkpoints. Query worlds are never used to construct the donor states.

Each query has six candidate actions, a host-owned legal-action mask, and ten synthetic local material/geometry/history features per candidate. Both the donors and receiver receive an explicit one-hot action-identity feature to avoid aliasing action-dependent scoring functions. A local preflight exposed and repaired that architectural flaw; before the repair some specialist donors achieved only ~50-80% single-family action fidelity, whereas after repair independent heldout donor policies achieved ~94-99%. This is an intentionally simplified **single-decision surrogate** for a CyberSand-facing policy, not native CyberSand gameplay.

## Trained comparisons and controls

The same-capacity receiving architecture is trained independently under three source channels: **outputs + hidden activations**, **outputs only**, and **penultimate hidden activations only**. An **architecture-matched query-only blind receiver** receives identically sized zero donor-state tensors **during training and evaluation**; it has the same trainable parameter count as each source-conditioned receiver.

Evaluation also checks **zeroed donor state** (an out-of-distribution intervention; not a capacity-matched baseline), **opposite-wet donor on the exact same query worlds**, and donor checkpoints from the other independent bank. The swapped-donor result is scored against the actual *counterfactual action* as well as the original ground truth; this distinguishes causal adaptation from arbitrary degradation. Predictions are always checked against the host-legal mask.

A deliberately **unseen three-way (+,+,+)-only physics term** is supplied as a separate challenge. The model has no evidence whatsoever for that term. Failure there is the scientifically correct expectation, not an engineering defect.

## Reproduction and provenance

Run locally (requires torch and numpy):

    python3 -m experiments.memplant.neural_donors_010.experiment --selftest
    python3 -m experiments.memplant.neural_donors_010.experiment --out runtime/memplant/dg010/seed-26011615 --seed 26011615 --donor-steps 230 --receiver-steps 700 --train-cases 4500 --sealed-cases 900

The [dedicated workflow](../../../.github/workflows/memplant-neural-donors-010.yml) trains three independent seeds (26011615–17). Each successful job retains the 12 trained source checkpoints, three trained receiving policies, blind checkpoint, detailed validation/heldout reports, original decisions, source SHA-256, and individual artifact SHA-256s in immutable run artifacts.

### Local CPU preflight (not a claim of CI qualification)

Two locally completed 700-update receiver trainings and twelve 230-update donor trainings **per seed**:

| Sealed (+,+,+) exact-action accuracy | Seed 26011615 | Seed 26011616 |
| --- | ---: | ---: |
| Both channels, correct **unseen-seed** donors | 88.44% | 90.56% |
| Outputs-only receiver | 86.33% | 71.67% |
| Hidden-only receiver | 72.78% | 39.67% |
| Architecture-matched query-only blind policy | 27.67% | 39.56% |
| Both channels, wrong wet donor (original-law match) | 33.78% | 25.11% |
| Both channels, opposite-wet donor matches counterfactual gold | 90.33% | 91.78% |
| Both channels, hidden unseen three-way rule | 52.67% | 52.11% |

**Interpretation:** the combined output/activation channel transfers useful game-law preferences across independently trained donor weights and source-world seeds. The corrected source models reach 94-99% single-family fidelity. The combined receiver outperforms the capacity-matched blind model by 60.77 and 51.00 percentage points and recovers counterfactual target actions in >90% of cases when the wet donor changes. **Hidden-only is not independently established:** its seed-26011615 accuracy barely changes (72.78% -> 72.89%) under the wrong wet donor, while its seed-26011616 accuracy is low (39.67%). Mere accuracy with hidden activations does not demonstrate causal inheritance. The output-bearing channels provide the stronger signal.

The newly introduced three-way rule changes the oracle action on ~46% of worlds, and the model does not infer it correctly. A neural donor's state can encode supervised preferences; this experiment does **not** establish novel physics discovery.

## Hard boundaries and caveats

- No actual model-to-model native KV-cache transplant, cross-architecture adapter, or independent discovery of rules. Donors and receivers share a broad synthetic action vocabulary; all donors use the same small MLP architecture.
- Donors' utility targets are **authored and supervised**; this is not an emergent learned physics model.
- No native CyberSand observation, painting history, pixel/physics simulation, or real long-horizon performance. The observations are independent one-step synthetic worlds.
- Source channels use fixed, known family slots and a fixed calibration probe bank. This greatly simplifies identification.
- No guarantee that a neural receiver outperforms explicit decoding + symbolic composition. An oracle that knows the original laws achieves 100%; practical symbolic/factorized comparators remain needed.
- Never modify CyberSand native/Godot/game source, assets, materials, maps, Workbench, or production/default settings from this experiment.
- All research PRs remain Draft/HOLD; no merge without explicit owner approval.

## Proposed DG011 falsification target

Train donors on **different architectures** and partially unfamiliar atomic families; change calibration probes, hide family-slot ordering, and require source-causal interventions across genuinely unseen world dynamics and longer trajectories. This will test transfer beyond learned same-family utility signatures rather than relabelling ordinary context conditioning as transplantation.
