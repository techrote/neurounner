# DREAMGRAFT DG018 — sequence-conditioned inherited laws (Draft/HOLD)

## Purpose and boundaries

Research-only CPU training. The downstream target is CyberSand-like host-legal
navigation, wet-material routing, protected painting, gate closure, RETRACE
recovery and alternate spatial goals. **No CyberSand game files are edited.**
This experiment stacks on DG017 (#250); no merge or Ready promotion.

## Research question

DG017's DAgger-style corrective training decreased goal completion compared
with budget-matched history-aware behavioural cloning. Test whether (1)
recurrent action-conditioned hidden state and (2) a supervised *remaining
weighted trajectory cost* objective improve long-horizon performance and wet-law
counterfactual sensitivity without supplying Dijkstra distances or teacher
potentials to inference.

All input channels are 13 host-observable grid channels plus 24 genuinely
trained source-network output probes (or no probes for the blind control).
Different donor slots have private feature permutations/reflections.

## Conditions (each independent seed)

- **value:** GRU receiver + true remaining-weighted-cost auxiliary prediction,
  20% loss weight. Only during training; no oracle potential at inference.
- **bc:** identical GRU and compute budget, imitation loss only.
- **shuffled_value:** identical auxiliary objective and compute, but future-cost
  labels deliberately mismatched to state trajectories (negative control).
- **blind:** imitation-only GRU without donor network probes (source necessity).
- **value_reset:** same trained value model with recurrent hidden reset each
  time step (sequence-memory control).
- **wrong_goal / wrong_wet:** same world and trained value model, changing
  exactly one neural donor law (source intervention).
- **oracle:** independent Dijkstra-based teacher. Not a neural baseline.

All three conditioned models now start with **identical neural weights**, see the
**same sampled training minibatches** and receive the same number of updates;
only the auxiliary targets differ. This correction followed an initial successful
but initialization-confounded pair of CI seeds; those original immutable artifacts
remain preserved in Actions run 43939272935, rather than being rewritten.

The value estimate only influences behaviour through a jointly trained feature
representation; **it is not an online planner**. Cross-world cost-scale variance,
oracle policy imitation, and the IID/OOD maze + gate-timing confound remain.

## Protocol and acceptance

Two CPU-only, independently seeded trainings. Each creates 12 newly trained
source-network weights and four independently trained GRU receivers, with
checkpoint files, loss trace, per-world matched policy traces, source and
artifact SHA-256, a rendered SVG gallery and full IID/OOD results.

Train on seven of eight atomic law triples. Test on 100 sealed IID and 100
sealed OOD random mazes per seed (half using the withheld +++ combination).
Report goal-and-paint completion, failure-penalised cost, legal actions,
RETRACE rate and conditional-on-joint-success regret. Use a separate
oracle-selected first-action wet-law divergence cohort to avoid falsely
attributing route variance to law inheritance. Do not assume gains.

Causal interpretation demands that value improves against BOTH bc and
shuffled_value, and that wet donor changes drive corresponding matched-world
oracle changes. A good completion score alone is insufficient.

The CI workflow is pinned to CPU PyTorch 2.5.1 and NumPy 1.26.4. No paid APIs
or external GPU commitments.

## Run

From the checked-out research PR:

```sh
python3 -m experiments.memplant.trajectory_transfer_018.experiment --selftest
python3 -m experiments.memplant.trajectory_transfer_018.experiment \
  --seed 26011632 --train-worlds 200 --updates 500 --sealed 100 \
  --horizon 96 --donor-steps 140 --out runtime/memplant/dg018/26011632
```

Artifacts belong in immutable Actions job uploads, not in the repository.
Draft/HOLD until an independent reviewer checks method, test isolation,
counterfactual controls and exact-head artifacts.
