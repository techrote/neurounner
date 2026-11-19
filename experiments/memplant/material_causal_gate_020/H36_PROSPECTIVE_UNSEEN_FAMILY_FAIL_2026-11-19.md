# DG020 H36 — preregistered unseen-family × genuine unseen-donor FAIL (19 November 2026)

**Scientific verdict: FAIL, 29/30 strict nonpooled cells.** The three frozen authentic 209-parameter receivers correctly handled all 2,880 invariant-law pairs but failed on two of 1,920 required divergent-law action switches. This is a more recent prospective negative result than H27's 28/30 FAIL. It does **not** establish generalized material-law inheritance.

## Prospective registration and fixed candidate

Registered **before training fresh H36 donors or generating any H36 acceptance worlds:** [PR #253 registration](https://github.com/techrote/neurounner/pull/253#issuecomment-6996446544). No acceptance-based parameter or threshold changes.

Training-only mechanism: source-blind donor dry-reference anchoring subtracts each learned donor's law-neutral dry cost bias (reference dry cost 1.0) from its donor-estimated wet cost; the same original 209-parameter 3→16→8→1 trained receiver keeps all 200 hidden parameters fixed. Only eight output-head weights were changed by a per-seed L1<=0.002 worst-donor both-law signed-route-margin LP. Fit to all twelve **exposed** genuine H21/H34/H35 donor profiles and 7,680 **exposed** H27 training worlds. In this fitted set: 7,680/7,680 both-law correct, no donor changes, no wrong invariant flips. Seed 26011641 worst fitted signed route margin remained 2.01e-5; this was known before preregistration.

Negative training-only ablation preserved: adding dry-anchor to frozen H35 without refit retained two invariant wrong decisions (made consistently across donors, showing that sham consistency alone is insufficient). Joint raw+anchored robust LP with L1<=0.002 was infeasible for seed 26011641. Anchor-only was then chosen **on exposed training data** before freezing prospective source/checkpoints; this choice is not independent validation.

## Untouched prospective design

Two **new** world families, `spiral_spoke` and `ladder_weave`, use new wet placements and contrasting neutral fee distributions. Three new world RNG seeds 26011801/26011803/26011807 correspond to three frozen receiver seeds 26011639/26011640/26011641. Five original buckets × 160 worlds × 2 families × 3 receiver seeds = 4,800 counterfactual pairs / 30 strict cells. Exactly identical world tensor, goal, legal actions, dry/wet materials, fees, textures and receiver weights in paired law arms; only authentic donor-derived wet material law changes. No law/family/source ID, oracle route distance or answer goes to the receiver. All 4,800 world hashes disjoint from the 15,360 original exposed H27 training observations.

Eight **new** authentic 4→24→24→1 trained neural donor checkpoints were trained after registration, sources 29002161/29002167 with two independently seeded replicas for both laws; original tile-only AdamW recipe (1,000 CPU steps, 8,192 training tiles and 2,048 validation tiles per net), 5×5 genuine neural output quadrature. Worst validation MSE 0.000932772, below locked 0.02.

## Frozen stringent acceptance results

| Endpoint | Actual | Locked gate |
|---|---:|---:|
| Individually passing cells | **29/30** | 30/30 |
| All genuine donors both-law correct | **4,798/4,800** | 4,800/4,800 |
| Divergent correctly switch both directions | **1,918/1,920** | 1,920/1,920 |
| Invariant both laws correct | **2,880/2,880** | 2,880/2,880 |
| Inappropriate invariant law flips | **0** | 0 |
| Individual same-law donor-source changes | **0** | 0 |
| Illegal actions | **0** | 0 |
| Zero / swapped-law divergent both-correct | **0 / 0** | 0 / 0 |
| Shuffled wrong-law divergent both-correct | **487/1,920** | at most 1,152 |
| Independently replayed raw neural logits | **4,800, drift 0.0** | exact |

**Only failing cell:** receiver seed `26011641`, `spiral_spoke / expensive_edge_divergent`, 158/160 both-law correct. Both distinct worlds require expensive-law action reversal and every new genuine donor profile failed in the same direction:

| Index | Exact new observation SHA-256 | True actions | Receiver actions (all 4 profiles) | Required law-1 margin |
|---|---|---|---|---:|
| 58 | `c6bcaf20f0513f6eb3a4de56d3397152ff5ebed5f063cb11595c19614b96efa3` | [1,0] | [1,1] | 0.000542506 |
| 94 | `d749bd1bac28e65519e1d27f24ebbb6644318c407fb6d38d614c016aabec9177` | [0,1] | [0,0] | 0.000558700 |

**Failure diagnosis:** conditioning corrected donor-source instability and new invariant selectivity, but the limiting receiver still miscalibrates the expensive-law absolute route-action margin on two newly encountered fee/topology distributions. Within-law source shams cannot detect an error shared by every donor. The remaining failure is no longer a global wet-route preference in these tested invariant worlds, but the scientific gate *still fails* because two required divergent flips are missing.

## Original frozen source and custody

- Prospective exact executed generator/training/evaluation source SHA-256: `489cbdec2a30324ff59374339949ffe16ce3030f0698de6f5ff7100f9dc8c1a4`.
- Training-only fit exact source SHA-256: `852fc2400be37f71c231c5430b9ef1b0f6388de157645b235f9707365432f093`.
- Original genuine donor recipe source SHA-256: `afbfb713777a864ec030663e65aeeebffa0d117ad28fc6ee90e387c8d256e030`.
- Three original H36 serialized 209-parameter receiver checkpoint SHA-256 (seeds 26011639/40/41): `6ec90371ecfb7fdcc7291959bc1e9a0d98df71836cef3100a2205f3dfea3fa2d`, `dc2b2fbc7b9fa04303915cc92255bdcb0fb15e7218c9e96a4e5e24717317ed05`, `49b31a99cfcd537d24264880e26c0d92784ed846318959dfe06ce8c11dc1a681`.
- Complete 4,800-observation raw receipt SHA-256: `b9c8e530d1b09995b946a2b51d271e8e8c507e6ff9f2de8b51cba25bced45648`.
- Full external evidence capsule: `DG020-H36-prospective-unseen-family-new-donor-FAIL-20261119.zip`, **28,076,144 bytes**, SHA-256 `c9e660f9fb23139b7d6980f36ab677a233d096f4533e64a5cc3a4f54a74098c2`. Contains original genuine donor and receiver checkpoints, exact executed sources, all 4,800 complete observations and raw decisions/logits, H36 training-only ablations and negative cases, independent genuine checkpoint replay, and the untouched nested H35 archive. **This binary capsule is not yet in durable GitHub custody**; no false claim of GitHub-only reproducibility.

Clean extraction: every entry passed SHA-256 manifest verification, and independent CPU replay (without importing H36 trainer or generator) verified all 4,800 receipts, all 30 cells, donor/receiver checkpoint hashes, exact two failures and **0.0 maximum logit drift**. Original failure evidence is retained, not filtered.

## Smallest justified next mechanism

**No new holdout until training-only fix.** For seed 26011641, investigate a law-neutral, symmetric positive/negative absolute route-margin calibration objective on already exposed training data, not more donor-invariance loss. Constrain both divergent directions and invariant decisions across individual authentic donor replicas. The two H36 failures are now EXPOSED regression cases; never reuse their worlds/donors as untouched validation. If no robust margin is achievable with the existing 209-parameter architecture, report a concrete capacity/identifiability limit rather than enlarging architecture or relaxing thresholds.

Only existing `techrote/neurounner` Draft/HOLD PR #253 updated. No merge/Ready, new PRs/issues/branches, paid GPU/API or CyberSand/DRIV3R modifications.