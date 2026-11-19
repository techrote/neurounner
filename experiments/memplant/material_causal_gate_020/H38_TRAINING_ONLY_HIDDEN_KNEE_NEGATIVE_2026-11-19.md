# DG020 H38 — hidden activation-knee trial: TRAINING-ONLY FAIL

**Date:** 19 November 2026. **Scientific disposition:** H38 training-only FAIL; last preregistered unseen-family/new-donor scientific verdict is **H36 FAIL 29/30 cells**, unchanged. Existing PR #253 remains Draft/HOLD, never merge or Ready.

## Scope and control

- Only receiver seed **26011641** was modified. It remains the same genuinely trained **209-parameter** receiver; H36 source checkpoint SHA-256 `49b31a99cfcd537d24264880e26c0d92784ed846318959dfe06ce8c11dc1a681`.
- All **16 individual genuine H21/H34/H35/H36 donor source/replicate profiles**, two unchanged physical wet-material laws (0.35 and 2.30, dry=1.00), and original dry-anchored inference were retained. Both arms share geometry, goals, fees, legal mask, receiver weights and all other inputs. Only the donor-provided material law differs.
- Exposed old H27 **2,560** worlds + formerly prospective, now **exposed** H36 **1,600** failure/regression worlds. These 4,160 paired observations generate **133,120** individual genuine-donor/law signed constraints. **No newly generated holdout**.
- Training targets use both-law oracle action correctness and symmetric hinge loss; oracle costs, target actions, family IDs, law IDs and donor IDs are **never receiver inference inputs**.
- Tested (1) linearized LP over existing hidden activation biases and output head, (2) two fixed-seed nonlinear 450-step Adam regimes with bounded hidden biases and head, (3) bounded active-set LP continuation. Original linearized LP infeasible; continuation from best nonlinear model also infeasible locally. These are limited local diagnostics, not global nonexistence proofs.

## Exposed-data results, all 16 genuine donor profiles

| Metric | H36 baseline (seed 41) | H38 best training-only |
|---|---:|---:|
| Old H27 both-law all-donor correct | **2,560/2,560** | **2,558/2,560** |
| Exposed H36 both-law all-donor correct | **1,598/1,600** | **1,599/1,600** |
| Combined both-law all-donor correct | **4,158/4,160** | **4,157/4,160** |
| Required divergent action switches correct | 1,662/1,664 | **1,664/1,664** |
| Invariant pairs both-law correct | **2,496/2,496** | **2,493/2,496** |
| Incorrect invariant law-induced action flips | **0** | **3** |
| Same-law individual donor-source action disagreements | 0 | 0 |
| Illegal actions | 0 | 0 |

**Negative finding:** both formerly missed expensive-edge divergent H36 decisions are corrected, but **three new wet-rich invariant action flips** are introduced. The mechanism traded directional correctness for invariant selectivity. This is a clear **training-only mechanism FAIL** and cannot warrant new prospective evaluation.

H36 exposed divergent negative hashes, indices 58 and 94:
- `c6bcaf20f0513f6eb3a4de56d3397152ff5ebed5f063cb11595c19614b96efa3`
- `d749bd1bac28e65519e1d27f24ebbb6644318c407fb6d38d614c016aabec9177`

New exposed H38 wet-rich invariant failures, oracle actions `[1,1]`, receiver actions `[1,0]`:
- H27 `random_scatter`, bucket index 190, SHA-256 `fd8516f108ae76c756345b7f45c2df7b73b48bb32cfd94b69401678a57f886c3`
- H27 `cyclic_run`, bucket index 112, SHA-256 `70edc9429fe1dc49efcc3f0bf44972d3c981307aa88812e2fc559d465af09dd8`
- H36 `ladder_weave`, bucket index 29, SHA-256 `54203db5288d3daae943508c7232670360f66abbca1bd4ead1f329168717aaee`

## Controls and exact provenance

Across **1,664 exposed divergent worlds**: zero-donor and wrong-law/swapped-donor each achieve **0** both-law correct (both receivers). Deterministically shuffled pre-inference law signal: H36 **418/1,664**, H38 **452/1,664**. All 16 same-law authentic donor sources produce **0** source action changes. Donor dependence remains genuine; invariance is not preserved.

H38 negative checkpoint SHA-256 `1c46fe204587cc2a4c7adcc3506a7cb1894632685ec5f93747b90a443a9f8051`. Only existing `tile.0.bias`, `tile.2.bias`, `tile.4.weight` changed; corresponding L1 movements **0.00037510693**, **0.00598484557**, **0.00021445751**, with the other three parameter tensors **bit-identical**. See `H38_FROZEN_NEGATIVE_RECEIVER_TENSOR_CUSTODY.json` for exact float32 bit representations. Never promote this candidate as accepted.

Full raw JSONL SHA-256 `4400b4b4fa9b21d1c38722bf7e7e090024ed8b7d6f4a09bf399d9b46f5fa3fe5`. A **separately implemented independent CPU replayer** checked all **4,160** exact original observation hashes, **532,480** individual genuine-network route logits, **0.0 logit drift**, exact regression indices, frozen receiver identity, zero illegal actions and controls. **Technical evidence PASS reproducing scientific mechanism FAIL.**

The **61MB owner-handoff archive** `DG020-H38-trainingonly-hidden-knee-TRADEOFF-FAIL-20261119.zip` retains executed H38 source, H36/H37 original negative evidence, source/donor checkpoints, H38 genuine checkpoint, all 4,160 raw paired observations, original-neural logits/actions per source and law, negative controls, independent auditor/log, and SHA-256 manifest. **Full binary archive not yet in GitHub custody; exact-HEAD H38 GitHub CI replay not yet qualified.**

## Smallest next mechanism change

Constrain **every exposed invariant counterfactual and individual genuine donor profile as a hard no-regression condition**, rather than minimizing a global average hinge objective. Jointly require both H36 exposed divergent failures to become correct within a bounded **existing 209-parameter activation-knee trust region**. If no local feasible solution exists, preserve the negative, interrogate identifiability, and **do not** increase model size, complexity, material-law families, or open new holdouts. Only after both-law/donor correctness, invariant preservation, sham/zero/shuffled controls and trained-receiver checks pass on the exposed screen should a separately preregistered new unseen-family/new-genuine-donor cohort be considered.

**Ownership:** techrote/neurounner existing Draft/HOLD PR #253 only. No new branches/issues/PRs, no merging/Ready, no paid API/GPU, no CyberSand or DRIV3R writes.
