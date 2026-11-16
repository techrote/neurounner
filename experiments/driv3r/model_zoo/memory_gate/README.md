# MZ-008 — Does a game policy remember an expired material law?

**Scope:** external, source-fenced game-policy research. **HOLD:** never merge or promote without owner acceptance. No edits to the CyberSand game, native/**, godot/**, assets, materials, maps, Workbench, physics or production defaults.

## Hypothesis

MZ-007b attained 76.28% held-out current-state game decisions, but deleting its previous observations improved its result slightly. It had not demonstrated useful recurrence.

MZ-008 constructs a deliberately history-dependent game-action task from the existing Model Zoo synthetic World. The game observes a transient four-way material/stencil law three prior observations ago. Two later irrelevant colored cues act as distractors. At decision time the material grid, goal direction and host-legal action menu remain visible, but the authoritative old rule is gone. The teacher is (rule + current_goal_direction) modulo 4 over host-legal STRAIGHT/LEFT/RIGHT/BRANCH.

Every eligible world snapshot is expanded into balanced rule-by-goal counterfactual pairs. All four hidden rules have **identical current percepts** for a given world and goal; current-only policies therefore have a mathematical 25% accuracy ceiling, not just an empirically low score.

## Real optimizer experiment

Two recurrent architectures train on the same synthetic TRAIN/DEV/TEST seed partitions:

- 8-layer 640-wide GRU, approximately 18.8M parameters (the MZ-007b-sized recurrent body).
- 2-layer 128-wide GRU, approximately 232k parameters (size-ablation diagnostic).

Each competes with a **full-context flat MLP** and a **current-observation-only MLP** under the same optimizer-update and minibatch-index budgets. The first training third teaches expired-cue recall and remaining updates teach cue+goal composition. Measure accuracy, macro-F1, law-swap and goal-swap both-correct pairs, history deletion and deliberately wrong cue substitution, raw host-illegal actions, batch-amortized CPU inference, RSS, checkpoints and four-by-four original percept comparisons.

The synthetic expired-stencil rule is NOT present in actual CyberSand. Successful performance proves only short delayed-rule credit assignment within this controlled external task, not real gameplay intelligence. The large and small models have unequal parameter counts and FLOPs. Do not infer CPU per-decision latency from amortized batch times.

## Reproduce

Run source tests, then isolated CPU model training:

    python -m unittest -v experiments.driv3r.model_zoo.memory_gate.test_memory_gate

    python -m experiments.driv3r.model_zoo.memory_gate.run --arch small --out /tmp/mz008-small

    python -m experiments.driv3r.model_zoo.memory_gate.run --arch large --out /tmp/mz008-large

The dedicated GitHub Actions workflow is authoritative for exact-source qualification and retains checkpoints in immutable workflow artifacts rather than Git history. Branch stacks on PR #650 to reuse existing synthetic World and previous policy research, without importing production game components.
