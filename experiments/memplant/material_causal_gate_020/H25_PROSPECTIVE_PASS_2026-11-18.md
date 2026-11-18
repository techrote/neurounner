# DG020 H25 — prospective same-size receiver selectivity trial (18 November 2026)

**Research verdict: 30/30 nonpooled cells PASS**, within the fixed synthetic two-law, two-route, 2×12 benchmark. **Not a general physical-law inheritance or game integration claim.**

## Registration and immutable provenance

- [Prospective preregistration, before any H25 holdout inference](https://github.com/techrote/neurounner/pull/253#issuecomment-6976760666). Source and training checkpoints frozen in that comment.
- Frozen source `h25_selectivity_repair_trial.py` SHA-256: `b1a385dc4c9bceea4aa5b3bf2f8f22382db1e9f0af7b646b755f631815a00853`.
- Imported original H23 helper `h23_paired_margin_trial.py` SHA-256: `41475d535ce4005e1f94c57385891b2f90b87f01e5f265d4ee0eec482bfb8358`.
- Frozen trained H25 receiver checkpoint SHA-256: seed 26011639 `49a74ad64fbfd79a4f5771bfa71158186b5025da038e621c87edc0331c234df3`; 26011640 `21df8ec5e5a1a9d16aa6fefd42054eb987a3092719521b7335e32699e7d789c3`; 26011641 `093750d8e0f9b35affe5c831b0c01024a002638e37672fd367db57b3d6093018`.
- H25 report SHA-256 `b12a515b55467e36c8a343f28085e0fc4ac1427476dd03b100b1567a239785bb`; independently computed receipt/checkpoint audit JSON SHA-256 `ef19aacac83417f9190463a2c87558063cc0a84a0edfc7efa9bbbde346d5bfcc`.
- Full owner evidence ZIP `DG020-H25-prospective-causal-selectivity-PASS-evidence-20261118.zip`, **9,086,490 bytes**, SHA-256 `79602f8e86acb91fdd551517997c63b5ba9f021d9a95141ba5df97a220c51d3a`. This is a **handoff** at this stage, **not GitHub-backed raw-evidence custody**.

## Minimal mechanism intervention

The three original genuine H19 receivers remain exactly **209 parameters** each; H21R donors are genuine trained neural checkpoints, evaluated with the same task-specific 5×5 quadrature. The only receiver change is retraining on the same existing training families (1,100 CPU AdamW steps, LR 0.00015, batch 128) with:

`5 × tile-cost MSE + 2 × absolute per-law route-margin MSE + 2 × paired law-delta MSE + 1 × equal-wet margin-delta MSE + 0.02 × frozen-H19 logit trust MSE`.

This explicitly fixes the H23 failure of correctly estimating *margin changes* but losing the *absolute* decision ranking, and strengthens equal-wet additive invariance. Material-law and oracle cost targets are training labels, **not receiver inputs**. The receiver sees only ordinary cell observations and trained donor-derived scalar material cost. Geometry, textures, fees, goals, legal masks and receiver weights are identical in each counterfactual pair; **only the inherited wet-material cost changes**.

## Prospective results (independent audit reproduced)

| Predeclared quantity | Observed |
|---|---:|
| Independently judged acceptance cells | **30/30 PASS** |
| Matched unseen test worlds | 4,800 |
| Divergent both-correct necessary flips | **1,920/1,920** |
| Invariant both-correct decisions | **2,880/2,880** |
| Inappropriate invariant action flips | **0** |
| Same-law alternate genuine donor-source action changes | **0/4,800** |
| Illegal actions | **0** |
| Max equal-wet cell p99 law-induced score-margin delta | **0.001992789** (limit 0.0100) |
| Maximum squared trained donor cost-estimate error | **0.00000673685** (limit 0.0200) |
| Divergent shuffled both-correct | 457/1,920 (weak control; below all per-cell limits) |
| Zero-donor divergent both-correct | 0/1,920 (mathematical sanity check only) |
| Raw original action/logit replay disagreement | **0** |
| Hash overlap with all 14,400 H22/H23/H24 exposed worlds | **0** |

Unseen prospective samples used two new wet-placement orders (`coprime_hop` and `nested_sweep`) with the same two fee-distribution classes, five existing causal categories, 160 pairs/cell and three receiver seeds. No added laws, topology, maze complexity or model width.

## Qualification limits and negative results retained

- H22 historical restricted PASS remains valid with its three wrong invariant action flips, earlier control-power deficiencies and additive-score weaknesses preserved.
- **H23 scientific FAIL, 18/30:** pair-delta-only penalty suppressed all 1,920 essential divergent flips. Full original raw results retained in owner H23 handoff.
- **H24 PROTOCOL-INVALID:** a zero-donor divergent-only criterion was erroneously evaluated for invariant cells. Its 4,800 worlds are exposed, never to be reused as prospective acceptance. H24 otherwise had 4,800/4,800 correct actions but failed two legitimately preregistered equal-wet p99 score thresholds; correction of only the control would yield 28/30 FAIL. Full H24 raw negative receipts and trained checkpoints retained in owner H24 handoff.
- Source-specific learned-donor probe produces nearly exact scalar costs. Same-law trained-donor substitutions cause tiny feature changes, and zero/shuffled controls have limited causal discriminating power. **H25 does not establish transfer of a richer latent physical law, donor weight graft, unseen physics or general game-policy behavior.**
- Local CPU PyTorch 2.10.0+cpu / NumPy 2.3.5 replay succeeded. H25 **has not yet been independently qualified in exact-HEAD GitHub Actions**; do not misrepresent existing H22 CI as H25 CI. Full raw H25 GitHub custody and truly external methodological review remain open. A maximum per-law route-margin error of 0.04327481 persists outside the registered equal-wet constraint, despite correct actions.

**Next engineering step:** commit the SHA-pinned exact H25 code/checkpoints/raw receipts into the existing lane, run independent exact-HEAD CPU neural replay and obtain external methods review before any broadened scientific claims or game implementation. Do not reuse H25 test worlds for future acceptance.

Ownership unchanged: `techrote/neurounner` PR #253 only, Draft/HOLD, never Ready/merge. No new PR/issue/branch, paid API/GPU spend, or CyberSand/DRIV3R writes.
