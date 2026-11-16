# DREAMGRAFT GAME-STATE-004 — trained compositional game-action recipient

**Research only; Draft/HOLD. No CyberSand game implementation or content edits.**

This experiment follows the DG002 frozen Qwen→Smol supervised two-rule numerical channel ([PR #233](https://github.com/techrote/neurounner/pull/233)). It is a separate path from DG003's negative result for zero-shot, frozen SmolLM2 novel causal inference ([PR #234](https://github.com/techrote/neurounner/pull/234)). The new test deliberately **trains an independent small game-decision receiver**, not the previously tested 1.7B language model.

## Research question

Can a policy trained against source-state-derived mechanics use two inherited rules to select a *legal and useful* game action when presented with a **new, previously withheld public mechanism**? Do heat-only/vibration-only source swaps produce the correct changed decision? Do combinations of scalars from two different donor contexts work? Does training with authentic source scores outperform training with zero state, and how does a vanilla dense policy compare to a rule-factored receiver?

## Actual data and training

- Donor: immutable [DG002 Actions run 43736088404](https://github.com/techrote/neurounner/actions/runs/43736088404), original `memplant-dg002-source-a1`. The **real frozen Qwen2.5-0.5B attention V** SHA-256 is `473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75`. Its complete NPZ digest and donor case text/manifests are checked before any training.
- Source conversion: the **same supervised lambda=1 training-only two-output ridge** as DG002. This is a two-scalar *learned* representation, not a universal latent or direct heterogeneous KV transplantation. Donor weights are fixed.
- Game situations: entirely **external synthetic**, CyberSand-inspired four-way local movement/paint/retrace action menus. The host legal mask excludes blocked/protected actions. A publicly supplied mechanism truth table, local inversion, material gate, painting/novelty value and movement cost determine the correct viable action. No game code, scenes, assets, content, Workbench or production providers are changed.
- Case split: 20 source families × 12 operator types × 4 turns (3,840 TRAIN); six separate families × two other operator types × six turns (288 VALIDATION); six DG002 previously inspected test-source families × **unseen XOR and XNOR operator types** × eight turns (384 test). Operator labels are never used for training or validation. Source cases in the test are **not independently fresh**.
- Three **real CPU optimizer** runs: factored learned source-conditioned policy, a same-form policy *trained with no donor signal*, and an unfactored dense model. Factored network has an explicitly programmed four-slot truth-table address; the remaining parameters and action selection are learned from game-choice targets. Checkpoint selection sees only validation.
- Falsification on identical held-out geometries: real source; zero state; shuffled source; oracle true-law state; actual heat, vibration and both-law donor-counterfactual states; and source-A heat + distinct-source-B vibration **scalar crossover**. Report both legal action accuracy and whether the *actual* decision changes when the mechanism requires it. Retain each original prediction, injected source ID and legal menu.

All counterfactual variants of a family/operator/turn share *the same external terrain/action menu*. This prevents a common label-leak shortcut through synthetic world generation. Ordinary zero-state policies are separately trained, not just evaluated with unexpected inputs.

## Interpretation boundary

Success could demonstrate a practical **state-conditioned trained game-controller prototype** computing unseen combinations of two inherited source rules. It would NOT demonstrate native cross-model KV translation, unaided LLM reasoning, a general inheritable world model, raw-state model-independent alleles, or a working native CyberSand worm.

The factored receiver is intentionally given a compositional inductive bias. Both source readout and receiver are supervised. Zero-state/dense controls and original causal swaps remain mandatory. Previously inspected source families make this **exploratory**, not confirmatory.

Expected artifact contents: three `.pt` trained checkpoint hashes, per-step loss/validation trajectories, held-out counts, complete raw individual decisions, donor provenance, and per-intervention causal response counts. Negative outcomes remain first-class evidence. GitHub Actions artifacts have 30-day retention and do not establish permanent research custody.

Reproduce using the **original verified** DG002 source artifact:

```sh
python3 -m experiments.memplant.game_state_lab.game_policy --selftest
python3 -m experiments.memplant.game_state_lab.game_policy \
  --source runtime/memplant/game-state/donor \
  --out runtime/memplant/game-state/results \
  --updates 450 --seed 260116
```

Draft/HOLD. No merge, Ready promotion, native runtime modification or paid external inference/training.
