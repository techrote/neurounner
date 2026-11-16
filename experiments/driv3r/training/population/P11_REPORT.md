# P11 — paired order-memory challenge (research-only)

**Scope:** external synthetic 22×22 CyberSand-like host, using the exact P0 `World.generate`, `World.legal`, `World.feature`, `World.step` contracts. No native/Workbench/game edits or production inference. Source-faithfulness applies to the synthetic P0 action menu and progression rules, **not** a verified CyberSand runtime protocol.

**Motivation:** P10 history-GRU and shuffled-history control were too close to establish causal use of memory. P11 constructs **paired legal world trajectories** in which the current world, full legal candidate features, painted geometry, retrace tree, obstacle/protected masks, goal, heading and action-token histogram are identical but the **history order** differs. The two missions consequently require opposite next paint choices. A state-only classifier or unordered-history classifier cannot distinguish the pair even in principle.

## Causal pairing

A legal precursor might be `E, RETRACE, N, RETRACE, S`, or `N, RETRACE, E, RETRACE, S`. Both explore the same branches and then paint the same southward cell, and both trigger the scheduled change of goal at step five. At this probe the host still allows `E` and `W` painting. The synthetic mission target selects `E` or `W` according to which surveyed branch came first and the niche's orientation convention. Three branch pairs, two goal cards and all six niches generate counterbalanced labels per identical world state. Only the **action history** reveals the earlier instruction cue. Obstacles and protected cells remain in the rest of the world; only a minimal seven-cell legal pocket around a split-specific root is cleared so both counterfactuals are physically playable.

**Important limitation:** this is an intentionally constructed memory diagnostic with an **explicit auxiliary order-classification loss**, not learned style discovery, emergent goal understanding or native CyberSand competence. The auxiliary target is inferred from the synthetic mission's ordered precursor, and the shuffled-history control gets the same architecture and loss. The six niche labels are kept for cross-niche conditioning; the P11 classification task does not separately validate their artistic specialisations or material handling.

## Real matched training

P11 trains three scratch-built tiny neural models per cohort, all starting from matched initial weights and receiving the same world, label, optimizer and data ordering:

- **state_only:** current legal-feature candidates and niche, no history context;
- **token_gru:** order-sensitive 8-token action history plus nonlinear niche/history fusion;
- **history_sham:** identical architecture to GRU, but each training example's nonpadding history tokens are permuted randomly, preserving the exact action counts.

The nonlinear fusion is a deliberate repair to P10's additive-only niche/history context: the task requires their interaction. All three arms receive the same 0.5-weighted auxiliary order-label loss, so the control isolates the usable historical-order signal rather than simply comparing different amounts of supervision.

In P11's current local campaign, **three independent cohorts (646–648)** each train with **80 training-world seeds, 24 validation-world seeds, 36 never-trained held-out-world seeds**, two goal cards, three survey pairs, six niches, two counterfactual orders, 18 epochs and 40 hidden units. Per cohort: 5,760 training examples, 1,728 validation examples and 2,592 held-out decisions (1,296 matched pairs). These are paired decisions within worlds, **not independent 2,592-world trials**.

| Treatment | Cohort 646 held-out | Cohort 647 held-out | Cohort 648 held-out | Swapped-history test |
|---|---:|---:|---:|---:|
| State-only | 50.0% | 50.0% | 50.0% | 50.0% |
| Ordered-history GRU | **100.0%** | **100.0%** | **100.0%** | **0.0%** |
| Shuffled-history GRU | 50.0% | 50.0% | 50.0% | 50.0% |

All arms retain **100% menu legality** because the host masks unavailable actions. Explicitly blanking the history drops the GRU to 50%; feeding the opposite member's history causes exactly wrong next choices in these held-out tasks. Thus the result demonstrates *causal temporal-order use for this constrained diagnostic*, not general game-planning proficiency. Strongly deterministic task mechanics and shared branch-pair templates across train/test make the ceiling score unsurprising; held-out generalisation covers new seed-generated obstacles and different root positions, not novel cue grammars.

No paid APIs or external GPU costs. Large model files, full original trace SVG/JSON, failed controls, exact seeds/lineage and SHA-256 manifests belong in immutable Actions artifacts. Keep PR Draft/HOLD; do not merge or mark Ready without owner acceptance.

## Reproduce

```bash
python -m unittest discover -s experiments/driv3r/training/population -p test_population_p11.py -v
for seed in 646 647 648; do
  python experiments/driv3r/training/population/population_p11.py \
    --output "/tmp/population-p11-${seed}" --seed "$seed" \
    --train-worlds 80 --validation-worlds 24 --heldout-worlds 36 \
    --epochs 18 --width 40
done
```

NumPy and CPU PyTorch required. Checkpoints are **new from-scratch GRU students**, not pretrained SLM adapters. Report torch and NumPy versions alongside the Git HEAD and preserve deviations/failures rather than retroactively rewriting historical runs.

## Next falsification

Hold out an entirely new **cue grammar** (three-step branch order or timed material signal), lengthen the memory gap past eight decisions, introduce distractor retrace episodes, and test *multi-step* mission completion rather than one controlled branch choice. Treat any drop on these tests as a genuine limitation; do not collapse the niches or discard negative trajectories.
