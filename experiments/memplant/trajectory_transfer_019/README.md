# DREAMGRAFT DG019 — matched-world causal wet-law transfer (Draft/HOLD)

Research-only CPU training stacked on DG018 PR #251. Do not merge, mark
Ready, or edit CyberSand. These are synthetic, host-legal game worlds designed
around prospective CyberSand navigation, painting, protected cells, wet-material
costs, timed gates, and RETRACE.

## Research question

Across DG016–DG018, goal-donor information influenced navigation, but swapping
wet-material donors almost never induced the appropriate alternative route.
Can a receiver trained on **matched observations with conflicting material-law
actions**, where the distinguishing information is produced by independent
neural donors, learn an actionable counterfactual material law?

This experiment deliberately separates *causal intervention fidelity* from
game-world completion. It does not attempt to claim KV-cache transplantation,
unsupervised discovery of new physical laws or native CyberSand success.

## Method

- Train 12 independent narrow/gated source networks, with private input
  permutations and reflections. Receiver-training signatures come from
  the narrow donors; all sealed evaluations use the withheld gated architecture.
- Generate ordinary expert imitation states in seven law triples, withholding
  **wet+ / alternate-goal+ / paint+**.
- Generate additional *paired* training observations on fresh training mazes.
  For each non-withheld goal/paint configuration, construct identical observable
  host states under both wet laws; retain pairs only when the independent oracle
  prescribes **different legal navigation actions**.
- **Paired receiver:** 128 ordinary + 64 divergence samples per update.
- **Standard BC:** same neural architecture and number of updates, 192 ordinary
  samples per update. Controls for training time but not oracle-divergence
  selection.
- **Shuffled signature control:** exactly the paired observations and labels,
  but independently permute donor outputs within each minibatch. This preserves
  oracle-selection bias, training budget and class balance while disrupting the
  per-sample source/target correspondence.
- **Source-blind control:** exactly the paired observations and labels, without
  donor outputs. A single observable state can require two opposing actions;
  the blind model cannot resolve that ambiguity from the observation alone.
- All conditioned receivers begin with identical weights and share the
  *same ordinary and paired minibatch indices* for every update. The BC
  control uses a separate RNG to draw extra ordinary samples; the shuffled
  control uses a separate RNG for probe permutations.
- At inference, policy outputs select the actual legal game action directly.
  No Dijkstra distances, oracle labels, law IDs, expert futures or path-cost
  potentials are exposed to the receiver.

## Sealed, preregistered measurements

Two fresh CPU seeds, each with newly trained donor/receiver checkpoints.
Test 80 IID and 80 OOD trajectories per seed, 96 actions maximum, with
independent replanning oracle and blind/shuffled/BC controls. Report success,
failure-penalized cost, host legality, RETRACE and route regret restricted to
joint-success worlds, never success-only cost as a replacement for failures.

Four independent oracle-selected matched-game counterfactual cohorts per
seed: IID and OOD, each tested with a withheld three-law pair (+,+,+)
versus (-,+,+), and a seen goal/painting configuration (+,-,+)
versus (-,-,+). Inspect up to 2,500 candidate worlds per cohort, retaining
up to 50 cases where alternative wet laws prescribe *different legal
navigation moves*. The selection is independent of any receiver predictions.
Report real denominators; never silently call missing cases a full cohort.

For every intervention, present the proportion in which the original model
matches the true-law oracle, the swapped model matches the opposite-law oracle,
**both** are correct on the same observation, and donor swapping actually
changes the receiver's action. Preserve a small matched-world evidence ledger.
Also measure the number of divergent cases occurring after gate closure.

Acceptance for useful material inheritance requires substantial paired-cohort
improvement over matched shuffled-signature, standard-BC and blind controls
on *both seeds*, with acceptable all-world trajectory performance and no
false oracle/state leakage. Do not use success rate alone as the criterion.

## Constraints and caveats

The narrow/gated donor outputs provide neural **functional signatures that are
constant for a particular law triple**. Consequently, a receiver might merely
learn a compact lookup or decode wet-law identity; successful scores would not
prove learned physics in general, stateful KV transfer, or spontaneous discovery
of unencoded new interactions. The supervised oracle-divergence curriculum
gives it information about the interaction. Paired-world selection biases
the training distribution toward decision boundaries. OOD geography and
gate timing change jointly. The unknown-case test withholds one three-law
combination but not the primitive law itself.

Record negative results, including cases where a source swap changes the
action incorrectly or full trajectories become worse.

## Reproduce

From the stacked research branch:

```sh
python3 -m experiments.memplant.trajectory_transfer_019.experiment --selftest
python3 -m experiments.memplant.trajectory_transfer_019.experiment \
  --seed 26011634 --train-worlds 180 --updates 600 \
  --paired-target 240 --paired-limit 850 --sealed 80 --horizon 96 \
  --donor-steps 140 --probe-target 50 --probe-limit 2500 \
  --out runtime/memplant/dg019/26011634
```

No paid API or external GPU. GitHub Actions retains source-matched neural
checkpoints, per-world data, diagnostic examples, galleries and hashes in
immutable per-run artifacts, not the repository. Report exact HEAD and the
actual passed job logs before making qualified performance claims. Independent
review and owner acceptance remain outstanding.