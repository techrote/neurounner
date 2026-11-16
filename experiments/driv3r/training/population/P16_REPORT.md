# P16 — Soft ordinal memory vs explicit order factorization and learned temporal attention

**Scope:** `experiments/driv3r/training/population/**` only, plus one dedicated CPU Actions workflow. **Draft/HOLD; never merge or mark Ready.** All game episodes are from the frozen, self-contained P0 synthetic `World` host. This is **not** a native CyberSand evaluation, a real SLM, actual self-play, or a validation of six distinct artistic personalities.

## Question and experimental design

P15's 8,449-parameter temporal-attention model regularly inverted the held-out N-first grammar, while the 712-parameter explicitly engineered E-before-W comparator reached 86.98% mission success on six seeds. Can an even smaller, partially trainable **monotone first-cue priority** inductive bias preserve order/goal reversals without supplying the exact hard-coded pairwise E-before-W statistic?

P16 trains from scratch on eight independent seeds **664–671**, with 32 checkpoints, 20,736 real AdamW minibatch updates, and CPU-only execution. Three treatments are seeded on the same two training grammars, with identical world splits, optimizer and epochs; the fourth deliberately includes the third grammar in **training** and is a positive *grammar-exposure* comparator, **not a withheld-grammar experiment**. Full checkpoint state dictionaries are retained by GitHub Actions, not committed to Git.

- **Explicit factorization:** P15's engineered E/W pair order × engineered E/W goal progress × learned niche modulation (712 parameters).
- **Soft earliest-direction priority:** learns E/W token polarity, niche modulator and position-sharpness temperature, with an architectural monotone position bias and E/W gating; multiplies the engineered E/W goal-progress contrast (720 parameters). **Not** an emergent order-rule discovery; it is a deliberately strong bias.
- **Learned token-position attention:** P15's 8,449-parameter attention, without explicit order statistic or first-direction gating (8,449 parameters).
- **Grammar-complete attention:** identical 8,449-parameter architecture trained with the otherwise withheld N-first survey grammar. This treatment has 50% more training examples and updates at the same epoch count, so neither training data nor optimizer budget is matched to the others.

Paired counterfactuals share identical present observations, legal action menus, world geometry, protected-cell configuration and action histograms; opposite historical E/W order changes the correct answer. Reversing the current goal changes it back. Each evaluation mission is a four-decision **branch → retrace → other branch → stop** sequence, executed by the policy against host-legal menus without teacher forcing. Evaluation uses separate worlds, depth-nine distractors and unseen far-goal coordinates. There are 96 missions per treatment per seed (768 per treatment); separate validation worlds are used during optimizer reporting, not for checkpoint picking.

## Full eight-seed held-out results

| Model | Parameters | First action correct | Both paired first actions | Four-action mission success | Perfect seeds | Niche-conditioned Hamming |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hard order × goal | 712 | 97.92% | 95.83% | 93.23% | 6 / 8 | 0.3031 |
| **Soft first-cue priority** | **720** | **100.00%** | **100.00%** | **95.31%** | **6 / 8** | **0.3141** |
| Fully learned attention | 8,449 | 31.90% | 0.00% | 21.74% | 0 / 8 | 0.1775 |
| Grammar-exposed attention | 8,449 | 21.88% | 0.00% | 17.45% | 0 / 8 | 0.1927 |

These are CPU training results from local Python 3.13/PyTorch 2.10, not a claim of source-matched GitHub Actions success until the workflow qualifies. P16's recurrently evaluated outputs vary across seeds. The soft-first-cue model did **not** solve every entire mission: seed 666 reached 75% and seed 670 87.5%, despite perfect first-branch selection. Consequently, future work should target multi-stage continuation and stopping, not further specialize the first branch.

**Causal first-action swaps (mean across seeds):** soft cue original 100%, swap history 0%, swap goal 0%, swap both 100%. Hard factorization: original 97.92%, either single swap 2.08%, double swap 97.92%. Attention models frequently learned inverted held-out mappings; their below-chance accuracy must not be misrepresented as general competence.

## Independently replayed policy outcomes

The separate `p16_replay_audit.py` reconstructs each saved checkpoint and reruns every held-out mission from the original host world/seed. It checks checkpoint SHA-256, legal-menu membership on **every** action, exact correspondence with the recorded successes, and prohibited painting in obstacles/protected cells. Across all 32 checkpoints: **3,072 independent missions, 11,602 legal actions, zero protected/obstacle painting violations**, 100% host-menu legality. Conditioned Hamming is computed over action strings across six niche labels on matched worlds and goals. The labels encode an XOR parity teacher rule, so this is **not** demonstrated independent artistic specialization.

## Interpretation and boundaries

P16 strengthens the finding that explicit or partially engineered ordinal inductive biases dominate a generic attention model on this narrow and biased teacher task. Its high scores do **not** establish grammar-independent relational reasoning, naturally learned memory, native engine competence, source-verified CyberSand action semantics, genuinely diverse independent niche populations, or long-horizon autonomous painting. A 2.08-percentage-point full-mission improvement over the hard-coded comparator is too small to infer a generally superior architecture from eight correlated-seed cohorts.

The grammar-exposed attention treatment is a useful negative result: more of this particular synthetic grammar data does not rescue the unmodified architecture and optimizer. But the additional data and optimizer steps make it an **unequal-budget** comparison. No post-hoc claims of statistical significance, reward-model fidelity, or genuine self-play are warranted.

**Next:** P17 should confront the actual unsolved part: long-horizon multi-stage autonomous recovery/painting after correct first branch, richer host-generated material/obstacle perturbations, and representation generalization beyond E/W-specific engineered filters. Preserve all failing personalities and checkpoints. Do not modify game code; integration ideas are proposals only.

**Reproduction:** `python -m unittest test_population_p16 -v`; then `python population_p16.py --output /tmp/p16 --seeds 664 665 666 667 668 669 670 671 --epochs 12`; finally `python p16_replay_audit.py /tmp/p16 /tmp/p16-replay.json`. Run from the population research folder. Exact code, split hashes, seeds, per-checkpoint model state dictionaries, per-cohort metrics, counterfactual interventions, trajectory SVG/JSON, CPU environment and SHA-256 manifests are archived in the dedicated Actions artifact upon successful qualification.
