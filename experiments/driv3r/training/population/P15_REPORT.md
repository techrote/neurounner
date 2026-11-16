# P15 — Six-seed memory/goal generalization replication and architecture controls

16 November 2026 · Research-only synthetic CyberSand-game policy work · **Draft/HOLD**.

## Research question and bounded campaign

P14 achieved complete held-out mission success on one of two independent starts but failed to train reliably on the other. P15 reproduces the same goal-switch counterfactual mission using **six further independent world/training seeds (658–663)** and compares three distinct order/goal fusion architectures with two information-destroying controls. Each cohort uses four training worlds, two validation worlds, two untouched held-out worlds, all six niche labels, two training survey grammars and two training goal heights, with a third held-out grammar, unseen goal height and deeper retrace distractor. Training uses the actual P0 `World.legal`, `World.step` and `World.feature` synthetic host contract; every policy decision is drawn from a legal masked action menu. Native CyberSand equivalence remains unproven.

The shared target remains a **scripted four-step teacher**: choose the order-and-goal-dependent branch, retrace, paint the other branch and STOP. This is neither real adversarial self-play nor freeform authoring. Success requires matching all four actions. All models receive the same train/validation/held-out trajectories, evaluation seeds, action vocabulary and branch-weighted classification objective. Architectures are **not parameter-count matched** and do **not** share initial tensors. Parameters (width 32): original engineered pairwise fusion **6,273**, explicit factorized order/goal/niche comparator **712**, learned token/position attention **8,449**, history-shuffled attention **8,449**, state-only attention **8,449**.

The factorized model is intentionally granted the relation between earlier E/W actions and current goal-progress signal, but **must learn each niche's sign and action decisions through AdamW**. This is an inductive-bias upper comparator, not learned temporal abstraction or naturally emerging relational cognition. The learned-attention model sees only raw 32-position action tokens, state candidate features and niche embeddings; it has no explicit E/W ordering statistic. Shuffled attention retains token counts while disrupting temporal order. State-only suppresses history entirely.

## Completed independent local training

Six cohorts × five treatments = **30 real CPU-trained checkpoints**, **17,280 AdamW optimizer updates**, **36,864 training rows per treatment**, and **112.35 seconds total measured local CPU execution** on Python 3.13, PyTorch 2.10.0+cpu, NumPy 2.3.5, single-torch-thread. Evaluation executed **2,880 closed-loop synthetic missions and 11,004 legal host actions**. Each cohort/treatment has 96 untouched held-out complete missions, and all held-out decisions preserve the host legal-action mask. Six local P15 guardrail tests passed. Local full and per-cohort manifest checks passed without mismatch.

| Architecture / control | Held-out first branch, 6-seed mean | Matched-order pairs both correct | Complete 4-step mission mean | Perfect cohorts | Legal-action rate |
|---|---:|---:|---:|---:|---:|
| P14 engineered pairwise baseline | 71.35% | 42.71% | 71.35% | 2/6 | 100% |
| **Explicit order × goal × learned niche factorization** | **94.27%** | **88.54%** | **86.98%** | **3/6** | **100%** |
| Fully learned token-position attention | 28.47% | 0% | 11.81% | 0/6 | 100% |
| Shuffled-history attention | 50.00% | 0% | 41.49% | 0/6 | 100% |
| No-history attention | 50.00% | 0% | 42.71% | 0/6 | 100% |

Per-seed **complete mission** success, in seed order 658 / 659 / 660 / 661 / 662 / 663:

- P14 pairwise: 68.75 / 100 / 100 / 59.38 / 50 / 50%.
- Factorized: 93.75 / 100 / 100 / 53.13 / 100 / 75%.
- Learned attention: 0 / 0 / 0 / 0 / 50 / 20.83%.
- Shuffled control: 36.46 / 50 / 50 / 25 / 50 / 37.50%.
- State-only control: 43.75 / 50 / 50 / 25 / 50 / 37.50%.

The factorized comparator was more reliable than P14's general pairwise model on this deliberately factorized task, but **neither is seed-robust**: the factorized mission score dropped to 53.13% for seed 661, despite 78.13% first-choice accuracy. The learned-attention model often fitted familiar grammar yet collapsed under novel cue order/depth, sometimes producing **0%** first-branch accuracy; this can mean confidently learned the wrong rule, not random chance. Preserving the full held-out failure is essential. At six independent starts no significance claim about general tasks is warranted.

## Causal history/goal interventions on untouched holdouts

Using identical legal menus, swap the ordered-history member of each matched pair; separately swap only the two goal-dependent candidate features (`goal_progress` and `guidance`) across matched goal-side worlds; also swap both. Compare predictions against the original target, without revealing labels or metadata to the model. Six-seed mean first-branch accuracy:

| Mode | Original | History swapped | Goal swapped | Both swapped |
|---|---:|---:|---:|---:|
| P14 pairwise | 71.35% | 28.65% | 28.65% | 71.35% |
| Explicit factorization | **94.27%** | **5.73%** | **5.73%** | **94.27%** |
| Learned attention | 28.47% | 71.53% | 71.53% | 28.47% |

The factorized model exhibits the expected reversible XOR signature. The attention model exhibits an *inverted* signature on this unseen grammar: still sensitive to both inputs but learns the wrong mapping. This provides stronger evidence than raw mission success alone; none of it establishes autonomous material/obstacle planning.

## Independent checkpoint replay and niche-conditioned diversity

A second **inference-only replay program**, `p15_diversity_audit.py`, reloads every saved neural checkpoint, independently reconstructs all 2,880 held-out missions from the recorded seeds, and checks **exact agreement** with each training run's episode count, total issued action count and successful mission count. It reproduced **11,004/11,004 host-legal actions** with no protected/obstacle painting, confirming the original closed-loop metrics rather than simply trusting the first JSON summary.

For each fixed (world, card, goal, survey order) context, the audit runs all six niche-conditioned variants and measures the **normalized four-action Hamming distance** between their trajectories. This is a deliberately narrow behavioural-diversity statistic, *not artistic creativity or independent policy diversity*; in this synthetic test the teacher itself defines only an even/odd niche parity rule.

| Mode | Mean distinct niche traces / six | Mean within-context niche Hamming |
|---|---:|---:|
| Pairwise baseline | 1.427 | 0.1281 |
| Factorized order/goal | **1.906** | **0.2734** |
| Learned attention | 1.531 | 0.1509 |
| Shuffled-history attention | 1.063 | 0.0073 |
| State-only attention | 1.000 | 0.0000 |

The factorized model more often separates required niche-parity decisions, while the state-only model collapses to the same sequence across all six labels. Treat these results as **conditioning sensitivity, not six independently evolved artistic personalities**. This audit's independent replay is now part of the reproducible Actions artifact.

## Provenance and limitations

Local source SHA-256 (P15 initial qualified run):

- `population_p15.py`: `73e2c45a6b6ccbaf199b8f90b1d117540fc8d2a23cc07e0e137bc82a8534f54c`
- `test_population_p15.py`: `8b6a3cc95d6ae770ad1e07c3d5145b70a2f4f616ec4377dd1a03eed5f162be64`
- `driv3r-population-seed-replication.yml`: see source-matched Actions artifact and git blob for the final workflow revision.

`results.json`, `metrics.csv`, trained `.pt` checkpoints, per-model episode JSON/SVG traces and SHA-256 manifests are retained. The contact sheet is assembled from actual P15 checkpoint-executed P0 synthetic trajectories across six niche conditioning labels and is **not native CyberSand evidence**. All six niches were retained; no unusual but functioning personalities were culled. The six niche labels are not yet independently verified as divergent artistic behaviour. Their visible micro-trajectories can be extremely similar.

An earlier all-six launch was interrupted after finishing cohort 658 and partially writing cohort 659 because of an external execution deadline, and a separately completed pre-intervention seed-658 pilot exists. Those are *unqualified diagnostics*, not pooled into this campaign; retain them separately rather than overwrite or silently drop them.

**Boundaries:** no `native/**`, `godot/**`, Workbench, physics, renderer, maps, materials, assets or default providers touched. No paid API/GPU compute. No native playtest, SLM pretraining, native screenshot, native art evaluation or merge authorization. Large trained weights belong only in immutable GitHub Actions artifacts, not the repository. The Actions workflow runs the complete source-matched six-seed CPU campaign, preserves complete provenance and uploads even partially completed evidence on failure; **exact-HEAD CI must be verified independently after publication**. Keep PR #636 Draft/HOLD and unmerged.

**Next:** train a truly learned relational / attention architecture with balanced grammar augmentation and equivariant cue-order objectives, then extend to long autonomous changing-goal material-aware obstacle/recovery missions before claiming behavioural generalization.
