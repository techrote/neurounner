# DG020 H22 — independent post-hoc fee-placement invariance audit (18 Nov 2026)

**Disposition: POST-HOC DIAGNOSTIC; NOT H23, NOT A NEW SCIENTIFIC GATE.** The original H22 prospectively preregistered 30/30 **restricted PASS** is unchanged. Its original three near-tie errors remain exposed. This investigation does not justify larger receivers, more laws, more complex mazes, or general memory-transplant claims.

## Question and exact intervention

Can the frozen H19 receiver alter its action when **only the position of fees within an existing route changes**, even though both route totals and thus both true route costs are identical?

For each of the **4,800 already-exposed H22 observations**, four deterministic independent fee permutations were applied per variant (19,200 paired world/perturbation evaluations). Two variants:

1. **Unrestricted within-route fee permutation:** shuffle the twelve observed fees independently in each route. Wet masks, routes, textures, legal masks, goals, checkpoint weights, authentic trained donor material estimates, and **total fee of each route** remain unchanged. The true oracle route costs for both wet laws were recomputed and verified equal before inference. This deliberately changes fee-to-wet assignment, without changing the true additive route total.
2. **Within-material-group fee permutation:** independently shuffle fees only among wet cells and only among dry cells on each route. This is a stricter negative control preserving the within-group fee distributions.

For *each permuted observation*, law counterfactuals retain identical observations, legal masks and receiver weights, and modify only the wet-material cost derived from original genuinely trained H21R donors. No proxy receiver or oracle-distance inference is used. Frozen genuine H19 receiver checkpoints were replayed with PyTorch 2.10 CPU and exact baseline actions checked against each original H22 sealed receipt. There is no retraining. All 4,800 original float32 observation hashes were verified against original receipts before perturbations.

## Results

| Measure | Unrestricted within-route | Within-material group |
|---|---:|---:|
| Pair-evaluations (4 permutations × 4,800) | 19,200 | 19,200 |
| Any action changed from original matched receiver decision | **8** | **0** |
| Distinct original worlds with an action change | **6** | **0** |
| Incorrect paired actions after perturbation | **10** | **12** |
| Originally both-correct pairs made incorrect | **3** | **0** |

The original H22 cohort has three wrong paired decisions; across four repetitions that means 12 incorrect baseline pair-evaluations. The unrestricted permutation corrected some original mistakes but also introduced **three new incorrect pair-evaluations**. Eight changed pair-evaluations is **not** eight wholly independent worlds (six distinct originals), and the 19,200 perturbations are **not** 19,200 independent acceptance observations.

**Concentration:** all eight action changes occur in receiver seed `26011641`, equal-wet invariant worlds; three in `thin_center_tail_22` and five in `thick_center_tail_22`. The other two receiver seeds, all cheap/expensive divergent buckets and both non-equal-wet invariant buckets showed zero action changes under these four sampled perturbations.

Selected original receipt identities for newly introduced action errors:
- `26011641/thin_center_tail_22/equal_wet_invariant`, row 6, original SHA-256 `6a88f8e3eca7b444d8dfa7f82083ae5e4b273fd7cb2a940bbbd1b7b3d79ac6ba`: `[1,1] → [1,0]`, oracle `[1,1]`.
- Same cell row 1, SHA-256 `633b456df972c96146aa28735b83be5ff7109a5d9b18c369d3b495837a162eaa`: `[0,0] → [0,1]`, oracle `[0,0]`.
- `26011641/thick_center_tail_22/equal_wet_invariant`, row 111, SHA-256 `23b68f56e981accf2b722953d64fe49fb08b58476dad70961c741f12c9ce0be4`: `[1,1] → [1,0]`, oracle `[1,1]`.

## Interpretation and gate discipline

This is **very strong but not perfect** fee-placement robustness under the narrow tested grammar. The receiver remains slightly sensitive to the *association* of material class and positional fee, even though the true cost law is additive and positional rearrangement cannot change either route's true total cost. The within-material-group permutation control had no action changes, localizing the instability to cross-material fee assignment rather than mere order.

This is a post-hoc stress of existing H22 worlds, **not** independent held-out world-family confirmation, a new preregistered gate PASS, or a failure of the original preregistered H22 thresholds. It also does not demonstrate identical action under **all possible** rearrangements, only four deterministic samples per world and variant. Most importantly, H22's two unseen *fee-profile* families were still generated using the same world/bucket grammar. No claim of general game-world family transfer is justified yet.

The smallest possible mechanism correction, **only after original-evidence custody and pinned CI replay are closed**, is to constrain the same-size receiver to use additive route-level fee totals and donor-derived material-group costs without cross-material fee interaction. It should be trained with invariant within-route permutation pairs and divergent law-swap pairs, retaining sham/zero/shuffled controls. The previously exposed H22 worlds and fee permutations must remain diagnostics, not reused as fresh acceptance.

## Source and custody

- Original H22 locked acceptance ZIP: `DG020-H22-wet-only-causal-selectivity-PASS-20261117.zip`, SHA-256 `5a60ca1119a22916bda18ed46c0c6acecf77b4d891f5b1b841854a81533d1ca0`.
- New deterministic post-hoc audit source: `h22_posthoc_fee_permutation_audit.py`, SHA-256 `f0fe0b051158691ea1a082aea917bf134ec409e7eb5fd61e1143ece257a19293`.
- Full 30-cell JSON results: `H22_POSTHOC_FEE_PERMUTATION_RESULTS.json`, SHA-256 `8758622631a3bb9a568d43a27f589271cfae02c374b6c1e0924535214d6b8afe`.
- Audit source and raw perturbation outcomes are in the separate owner handoff ZIP. The original 30.3MB H22 archive remains **outside durable GitHub custody**; this note is not a substitute for the original genuine checkpoints, source, raw receipts and failures.
- Local CPU only for this **new** diagnostic. Existing exact-HEAD GitHub Actions H22 archive-safety tests are **not** original-checkpoint neural replay. Pinned genuine-checkpoint replay on GitHub Actions and external methodological review remain incomplete.

**Ownership:** Only existing `techrote/neurounner` Draft/HOLD PR #253. No new PRs, issues or branches. Never merge or Ready; no paid GPU/API. `techrote/cybersand` and `techrote/DRIV3R` are read-only and untouched.
