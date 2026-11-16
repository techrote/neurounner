# P14 — Goal-switch reversal and joint relational memory (synthetic CyberSand-game policies)

16 November 2026. Research-only evidence, **not native CyberSand**, not a pretrained SLM, and not a production Workbench controller. Only the P0 synthetic host's `World.legal`, `World.step`, materials/obstacle/protection contract and per-action features were used. Source-faithfulness to current native game behavior has **not** been independently established.

## Falsifiable task

Previous P13 established an engineered action-pair encoder could remember which of E/W was surveyed first on a fixed target rule. P14 makes that cue **insufficient**: after a legitimate simulated goal switch, an otherwise identical history requires the **opposite** first E/W branch if the new goal points west rather than east. An identical current world/goal snapshot requires opposite actions if the earlier survey order is swapped. Six niche conventions (even/odd parity), two training goal locations (`y=5,15`), two training survey grammars, two distractor depths and disjoint world seeds create combinatorial variation. The withheld task uses N-first grammar, depth-9 distractions and previously unused `y=19` goal positions.

For every history/goal/niche combination the teacher is balanced 50/50 within matched history pairs. Both branches are then painted and retraced in a four-decision local mission (first branch → RETRACE → other branch → STOP), using only the host legal menu. The goal instruction is available to models **only through spatial candidate features** `goal_progress` and `guidance`, not the JSON metadata or label; the no-goal ablation zeros exactly those fields. Niche labels are preserved as conditioning, **not evidence of artistic personality**.

Treatments: (1) P13's engineered pairwise-order policy with goal features, (2) same pairwise model with goal fields ablated, (3) 32-token recurrent GRU, (4) shuffled-order GRU preserving token histograms, and (5) state-only GRU. Directional targets received five times the training loss weight of RETRACE/STOP to avoid the trivial 87.5% total-action accuracy shortcut. Differences in model architecture/parameter count are not capacity-controlled.

## Independent CPU-trained outcomes

Each **source-matched weighted** cohort uses four training worlds, two validation worlds, two held-out worlds, six niches and matched 2x2 goal/order counterfactuals; 6,144 teacher-forced training stages and 12 AdamW epochs per cohort, five real neural checkpoints, **2,880 optimizer updates**. First-branch probes and completely model-driven four-step missions use 96 untouched held-out episodes per model and cohort.

| Cohort | Treatment | Held-out first branch | Counterfactual pairs both correct | Closed-loop mission | Legal actions |
|---|---|---:|---:|---:|---:|
| 656 | Pairwise+goal | 56.25% | 12.5% | 43.75% | 100% |
| 656 | Pairwise goal-blind | 50% | 0% | 0% | 100% |
| 656 | Full-history GRU | 50% | 0% | 50% | 100% |
| 656 | Shuffled-history GRU | 50% | 0% | 50% | 100% |
| 656 | State-only GRU | 50% | 0% | 50% | 100% |
| 657 | Pairwise+goal | **100%** | **100%** | **100%** | 100% |
| 657 | Pairwise goal-blind | 50% | 0% | 50% | 100% |
| 657 | Full-history GRU | 50% | 0% | 50% | 100% |
| 657 | Shuffled-history GRU | 50% | 0% | 50% | 100% |
| 657 | State-only GRU | 50% | 0% | 50% | 100% |

**Interpretation:** P14 demonstrates learnability in one of two independent starts, *not consistent generalization*. The 656 shortfall happens even on the seen grammar: this is optimizer/seed fragility, not just an OOD grammar failure. The goal-blind control cannot solve matched goal reversals; state-only/shuffled controls cannot solve matched order reversals. All success relies on a deliberately engineered temporal-order statistic; none establishes emergent general reasoning. In cohort 656 the pairwise model's 56.25% static first branch did **not** improve full-mission success (43.75%). Preserve this negative result rather than selecting only cohort 657.

Local guardrails: **6/6 passed**, including paired observation equality, matched-goal reversal, protection/paint integrity, source-legal actions, split isolation, shuffle histogram preservation and real optimizer weight updates. All cohort local manifests rehash cleanly. Local environment: Python 3.13, torch 2.10.0+cpu, NumPy 2.3.5; exact-HEAD GitHub Actions uses separately pinned Python 3.11/Torch 2.7.1 CPU/NumPy 2.3.5 and must not be claimed equivalent until independently checked.

**Source checksums (weighted local):**
- `population_p14.py`: SHA-256 `f34dc2f9d917e6b7d991730e0690dd567f7ffe1fffe0046e404f6e8ae0114b49`
- `test_population_p14.py`: SHA-256 `92b844e476d6f3ca52b94bf22876211e30b67a113e9f7c74f6eafa2a161844e2`

## Failed experiments and provenance

- Original **unweighted** 655 baseline: all five treatments reached 50% held-out first choice and 50% mission completion; 1,440 optimizer updates, full checkpoints and traces preserved. This used a different training-loss objective and must **not** be pooled with weighted 656–657.
- An earlier larger unweighted 654 attempt was terminated by a bounded local execution deadline after saving pairwise and goal-blind checkpoints; no completed report or validated manifest exists. Preserve its incomplete evidence separately, do not present it as qualified.

## Boundaries and next experiment

No native/**, godot/**, Workbench, map, materials, assets or production setting touched. No paid API or accelerator used. No merges, Ready promotion, or game integration. Checkpoints and failed traces belong in immutable run artifacts. At minimum repeat **six more independent seeds** at fixed hyperparameters and report pair-balanced accuracy, true four-step mission success, training cost and failures; then evaluate learned cross-modal attention or explicit relational-factorization baselines against the current engineered pairwise cue. Future longer autonomous missions must contain changing goals, obstacle/material planning and recovery beyond four scripted steps.

PR remains **Draft/HOLD**.
