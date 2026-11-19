# DG020 H23 — paired additive-margin retraining: prospective FAIL (18 November 2026)

**SCIENTIFIC VERDICT: FAIL, 18/30 cells.** Original H22 restricted 30/30 PASS remains historical and unmodified. The H23 preregistration was filed *before holdout inference* at [comment 6975077792](https://github.com/techrote/neurounner/pull/253#issuecomment-6975077792).

## Frozen protocol

- Preregistered evaluator: `h23_paired_margin_trial.py`, exact SHA-256 `41475d535ce4005e1f94c57385891b2f90b87f01e5f265d4ee0eec482bfb8358` (source contained in separate owner handoff; GitHub H23-source/raw-evidence custody **not yet completed**).
- Original genuine H21R donor networks, each 4→24→24→1, with the locked quadrature wet-only scalar intervention. Three original trained 209-parameter H19 receivers warm-started; same architecture and paired world/goal/legal masks throughout. 700 training-only AdamW updates, LR 0.0013, same two laws 0.35/2.30. No oracle actions, law IDs, family IDs, or full-route oracle costs at inference.
- Fresh holdout: 3 receiver seeds × 2 previously unseen wet-placement patterns (`prime_step_fan` and `mirrored_sweep`) × 5 predeclared discriminative/invariant buckets × 160 matched pairs = **4,800 unique world pairs, 30 unpooled cells**. Heldout fee profiles use Beta(0.77,3.6) and Beta(3.6,0.77). Holdout observations have zero hash collisions with H23 training and original H22 receipts.
- Original frozen genuine-neural capsule SHA-256 `47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667`, already held in this PR through 11 Git-backed chunks.

## Preregistered acceptance results

| Measure | H23 result |
|---|---:|
| Cells satisfying **all** registered metrics | **18 / 30 — FAIL** |
| Discriminative divergent cells | **0 / 12 — FAIL** |
| Divergent both-actions-correct | **0 / 1,920** |
| Necessary divergent action flips | **0 / 1,920** |
| Invariant cells | **18 / 18 PASS** |
| Invariant both-actions-correct | **2,880 / 2,880** |
| Wrong invariant action flips | **0 / 2,880** |
| Same-law trained-donor-source sham changes | 1 / 4,800 |
| Equal-wet p99 absolute law-margin drift | at most 0.000330, all six cells within registered 0.0100 |
| Illegal actions / checkpoint-to-receipt replay disagreements | 0 / 0 |
| Post-hoc independently checked maximum genuine-donor quadrature square error | 0.000006737 |

**Failure mode:** a training-only penalty on the *change* in route advantage between laws strongly suppresses spurious law-induced slope variation but allows the absolute route-advantage offset to drift. All 1,920 divergent pairs ceased to change action under the law swap. The original unmodified H19 receiver, independently replayed **after** H23 exposure on these same worlds, chooses both oracle-correct actions in 1,920/1,920 divergent and 2,880/2,880 invariant pairs. That post-hoc baseline comparison is diagnostic, not prospective acceptance. New H23 absolute route-margin errors reach 0.2994 (cheap-edge) and 0.6056 (expensive-edge). Training action CE appears to have degraded near-tie cost calibration despite improved delta-law consistency; donor prediction error remained negligible.

**Protocol weakness recorded without retroactive correction:** the registered donor-MSE threshold was omitted from the evaluator's prospective PASS boolean; post-hoc independent recomputation confirms it passes comfortably. Shuffled-donor control uses precomputed actions and has weak independent causal power. These do not rescue the 12 divergent FAIL cells, but must be repaired in a fresh registration.

## Smallest next change

Keep the same 209-parameter receiver, genuine donors, two laws and 2×12 geometry. Train on both divergent/invariant counterfactuals with **absolute per-law route-margin calibration**, `L_abs=mean((A_l-(C0_l-C1_l))^2)`, *in addition to* the paired law-delta penalty. Remove or sharply reduce action CE and add an H19 output/weight trust region. Preregister entirely fresh world families/seeds and independently actionable controls **before** the next evaluation; **never reuse these exposed H23 worlds** as acceptance evidence.

Complete H23 raw receipts (4,800), three newly trained checkpoints, exact source and the independent replay/audit are retained in the owner handoff archive, SHA-256 `91750e4c4f88c00faeddf7f190a3ad95bf749098c72e35eaaaca66db6d09d3c7`, 8,723,938 bytes. H23 evidence has **not yet** been durably published to GitHub. No paid compute; no CyberSand or DRIV3R modifications; existing PR #253 stays **Draft/HOLD, never merge/Ready**.
