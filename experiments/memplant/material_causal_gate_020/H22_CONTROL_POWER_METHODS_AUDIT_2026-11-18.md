# DG020 H22 — post-hoc control-power and claim-boundary methods audit (18 November 2026)

**Status: qualified original H22 30/30 restricted PASS unchanged; methodological challenge identifies non-discriminating controls and remaining generalization limits. NOT a new acceptance experiment, external methods signoff, or a new H23.**

**Source identity:** The sealed 4,240,821-byte capsule with SHA-256 `47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667`; frozen original H22 prospective source SHA-256 `a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d`. The existing exact-HEAD genuine checkpoint replay [run 44256216395](https://github.com/techrote/neurounner/actions/runs/44256216395) remains the primary neural reproducibility authority. This audit independently parsed all 4,800 raw receipt pairs using standard-library cost recomputation and checks of each stored float32 observation SHA-256, both oracle route costs/actions, intervention and control actions, all 30 cell sizes, and the original 31 receipt manifest hashes. The original separately archived neural replayer was also rerun locally with CPU PyTorch 2.10.0+cpu: **30/30 cells, 4,800/4,800, all checkpoint logits exactly replayed (maximum drift 0.0)**.

## Reconfirmed conditional effects and retained negatives

| Original H22 result, not a new cohort | Recomputed count |
|---|---:|
| Divergent both-direction correct and necessarily flipped | **1,920 / 1,920** |
| Invariant both-direction correct and action preserved | **2,877 / 2,880** |
| Inappropriate invariant action changes | **3 / 2,880** |
| Independent foreign same-law donor source action changes | **0 / 4,800** |
| Dry-anchor-only placebo action changes | **0 / 4,800** |
| Shuffled donor divergent both-correct | **496 / 1,920** |
| Receiver seeds with all 1,600 paired observations correct | **2 / 3** |

The **three** negative cases remain exclusively receiver seed `26011641`, bucket `equal_wet_invariant`: thin-tail index 14 (oracle margin 0.0175640), thick-tail index 5 (0.0159057), thick-tail index 152 (0.0233144). Original observation SHA-256 identities, in that order:

- `25880f1571179b91941bd5733b11504130b1e574985df225ac089d695e0c9e74d`
- `d2d6ded877657bccfe4714176d26d4093a22e8e26d6c8af97daa032d2abb826e`
- `135d0a2fdc7400ee12b97ebb19ea32c784949077cdff7e0cc3298435f89c89c9`

Post-hoc margin concentration: **3/175** equal-wet observations with minimum oracle margin at most 0.025 were wrong, versus **0/785** above 0.025. This is a diagnostic stratification selected after seeing the failures, **not** a newly validated threshold. Both held-out fee profiles and all three receivers remain in the counts. A nominal exact-binomial 95% interval for the observed invariant action-change frequency is approximately **0.0215%–0.3041%**, conditional on the synthetic sample; this does not give a distribution-free real-world bound because the generator, strata, and receiver training histories are shared.

## Important new methodological finding: the shuffled and fixed-action controls lack discriminating power here

The published pre-inference shuffled control is implemented correctly as a real feature selection **before** running the receiver. Nonetheless, because the evaluated receiver is deterministic and every divergent original pair is already exactly correct, its shuffled decisions are *algebraically predetermined* by the two original donor-conditioned actions and the recorded two coin flips. This audit confirms that reconstruction matches **4,800/4,800 shuffled pair decisions**. The entire divergent shuffled success count **496/1,920** equals the **496** original divergent rows where the shuffled choices happened to be `[0,1]` (cheap in cheap arm, expensive in expensive arm). A fair independent pair of binary picks has expected success `0.25 × 1,920 = 480`; the observed 496 is consistent with that baseline (two-sided binomial p approximately **0.399**).

Consequently, the reported 496/1,920 should **not** be cited as independent evidence that the receiver learned causal rule selection, although it can still confirm that feature swapping is wired through inference. The existing shuffled `<=60%` acceptance limit is weak in this design.

Similarly, because divergent oracle actions disagree between law arms, **no single action chosen without access to the intervention can be simultaneously correct under both laws**. Thus zero/source-blind `both_correct=0` on divergent pairs is a mathematical consequence of the test definition, not an independently discriminating model-competence control. This does not nullify the real donor-conditioned paired action result; it narrows what the controls establish.

## Same-law sham power and mechanistic boundaries

The measured genuine-donor-derived wet-cost signal differs by **1.95222944** between cheap and expensive laws. With the two genuine source groups, same-law source changes are only **0.00001881** (cheap wet) and **0.00008655** (expensive wet); the dry anchor changes by **0.00020778**. The maximum of these same-law coordinate perturbations is only **0.01064%** of the true wet-law contrast. A zero-action-change sham at this amplitude is real evidence of stability to *these* independent donor sources, but is a weak test of robustness to meaningfully different representations of the same law. It should not be generalized to arbitrary same-law donors.

The original intervention changes **only a donor-derived scalar wet-material cost feature** in a frozen 209-parameter receiver, with dry anchored and world observation held fixed. It does not edit a neural donor's weights to change an underlying law. The 5×5 material-specific texture quadrature deliberately extracts a physically meaningful cost coordinate; earlier post-hoc exact-cost replacement produced the same 9,600 actions. It is correct to call the result *conditional use of a trained donor-derived material-cost feature*, but not latent physical-law discovery or general memory transplantation.

The two H22 held-out `thin_center_tail_22` and `thick_center_tail_22` fee profiles vary parameters of the **same 2×12 world generator, wet-placement grammar, two actions, and route-cost equation**. They provide unseen fee-distribution shifts, not independent world topology or physical-law families. Observations are grouped by seed, family and bucket; aggregate binomial intervals must not obscure clustering.

## Disposition and smallest next action

**Original preregistered restricted gate: PASS, 30/30. Strict game-relevant transfer/generalization claim: NOT ESTABLISHED.** This audit finds no evidence of inference-time oracle/family/answer inputs in the inspected H19/H22 feature construction. Training does explicitly use oracle-supervised route costs, which is legitimate for this supervised receiver baseline. The principal unresolved empirical defect is the three low-margin equal-wet invariant wrong flips and the previously documented fee-placement non-additivity. The historical H17 28/30 FAIL, H18 29/30 FAIL, and original three H22 negatives remain untouched.

**Do not expand** network size, topology, material-law families or claim scope. First obtain genuinely *external* methods assessment of the donor probe, the weak control nulls, and shared world generator; this internal adversarial audit is not such a signoff. If further preregistered science is authorized, prioritize a same-size receiver trained to respect additive material-specific route costs plus harder, genuinely informative within-law sham controls, with matched invariant and divergent worlds and fresh sealed holdouts. No new experimental PASS or mechanism repair is claimed here.

**Governance:** existing [PR #253](https://github.com/techrote/neurounner/pull/253) only, Draft/HOLD, never Ready/merge. No new issue, branch, PR, paid compute, CyberSand or DRIV3R changes.
