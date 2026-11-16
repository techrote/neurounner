# P13 — Pairwise-order induction and closed-loop painting missions

16 November 2026. External model-only research; **not native CyberSand**, pretrained SLM training, or game integration. The exact sources and seed-scoped evidence are archived separately, and no game code or production settings are touched.

## Research question

P12's GRU learned the training survey grammar but one independent cohort failed to transfer to an unseen grammar with N-first distractor. Can an **explicit, learned-weight pairwise action-order representation** learn an order relation that generalizes to unseen branch-survey permutations? Can it complete a **four-decision, host-legal closed-loop task** instead of a single action probe?

## Experiment

- Source host: existing P0 `World.generate`, `World.legal`, `World.feature` and `World.step`. Disjoint seeded train, validation and held-out worlds; generated obstacles/protected cells remain except minimum cleared survey/test pocket. Directional actions paint cells, RETRACE moves backward through the parent tree, STOP terminates.
- Each matched pair surveys the same three E/W/N branches in opposite E/W order, executes an identical S/retrace distractor chain and arrives at **exactly the same observable world, mask, painting, action histogram, final eight history actions and post-switch goal**. The desired first E/W action is reversed within the pair, conditioned on niche convention.
- Supervised local mission: choose the hidden-cue-dependent first branch; retrace; paint the remaining branch; STOP. Teacher-forced four-step examples train the models. Closed-loop evaluation uses **four model-issued host-legal decisions without teacher forcing**; any wrong action fails exact mission completion.
- Train on E-first/W-first grammar families `train_a` and `train_b` and long-gap depths 5/7. Withhold N-first `heldout_c` and length-9 south/retrace distractors. Seed-disjoint validation and held-out worlds. Six original niches remain as conditioning labels, **not validated artistic personalities**.
- Four treatments: `pairwise_order` (a trainable MLP receiving normalized, antisymmetric action-pair co-occurrence counts), `recurrent_gru` (full 32-token action-history GRU), `shuffled_order` (GRU with order-destroyed history; preserved tokens/padding), and `state_only` (zero history). The pairwise representation is a **human-designed inductive bias**, not an emergently discovered world model. Model capacities differ (6,273 vs 10,659 parameters) and should not be interpreted as a capacity-matched contest.
- Two CPU-trained independent cohorts: **652** and **653**. Per cohort 12 train worlds, six validation worlds, six held-out worlds, 4,608 training stage examples, 2,304 validation examples, 10 epochs, 32 hidden units, AdamW 0.004, 1,440 optimizer steps across four models. PyTorch CPU, no paid accelerator/API.

## Verified local results

| Cohort | Model | Unseen grammar, depth-9 first decision | Both matched decisions correct | Four-action mission success | Legal actions |
|---|---|---:|---:|---:|---:|
| 652 | Pairwise-order MLP | **100%** | **100%** | **100%** | 100% |
| 652 | Full-history GRU | 50% | 0% | 50% | 100% |
| 652 | Shuffled-history GRU | 50% | 0% | 50% | 100% |
| 652 | State-only | 50% | 0% | 50% | 100% |
| 653 | Pairwise-order MLP | **100%** | **100%** | **100%** | 100% |
| 653 | Full-history GRU | 50% | 0% | 50% | 100% |
| 653 | Shuffled-history GRU | 50% | 0% | 50% | 100% |
| 653 | State-only | 50% | 0% | 50% | 100% |

The held-out four-step mission evaluates **144 episodes per model per cohort**, equal to 288 per treatment across the two cohorts. Each cohort includes 6,273-parameter pairwise model and three 10,659-parameter GRUs, total **eight real trained checkpoints**. Both cohorts include 1,440 AdamW steps, total **2,880**. **Seven** local P13 tests passed. All **212** per-cohort manifest entries (106 each) rehashed successfully.

The GRU's 50% first decision is not proof of competence; it always fails one member of every balanced matched pair. The 100% pairwise result demonstrates that a suitable *engineered order-statistic inductive bias* can be sufficient for held-out grammar transfer on this narrow synthetic task. It does **not** demonstrate that a generic recurrent model will learn this invariant, nor that open-ended agent gameplay or general material semantics emerged.

## Experimental hygiene

- No native paint simulation, game/Workbench edits, materials, maps, production provider settings, or paid APIs.
- Source-commit and dependency versions recorded in dedicated GitHub Actions evidence, plus model checkpoints, metrics, original synthetic trajectories (SVG/JSON), and both cohorts' SHA-256 manifests.
- Training/validation/holdout seed lists are serialized in each cohort's results, as are parameters, optimizer updates, per-mode checkpoint checksums, per-phase accuracy, heldout source-legal rates, mission success, and all six conditioned niches.
- No post-hoc reward culling; both successful and failed checkpoints and their trajectories are retained.

## Limitations and recommended P14

The major limitation is the engineered target rule. The pairwise encoder computes antisymmetric temporal co-occurrence rather than learning generic memory from raw tokens. It can solve the order relation without learning navigation, materials, or compositional world modeling. The four-action continuation is short and still follows a teacher-defined structure during training. Source-faithful **P0 surrogate**, not validated current Workbench action interface.

Next test: **controlled-grammar opposition** where the earlier E/W relation is uninformative and a new environmental cue or goal-switch redefines the relation; expand to >2-step adaptive planning, difficult negative constraints, curriculum with variable branch locations, and rigorous topology diversity. Ensure a position-conditioned transformer, recurrent control, pairwise bias and direct algorithmic oracle all receive adequate capacity/optimization sweeps. Do not generalize the two-seed success to SLMs or native use.

**PR must remain Draft/HOLD, unmerged.**