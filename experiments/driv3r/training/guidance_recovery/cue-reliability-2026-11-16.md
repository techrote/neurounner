# C15 — Cue reliability from observable legal-menu evidence (16 November 2026)

**Disposition: negative / retain C12; Draft/HOLD.** Research under `experiments/driv3r/training/guidance_recovery/**` only. No CyberSand native/Workbench/material/game code was edited.

## Hypothesis and implementation

C14 improved legitimate host-directed reversals but made misleading advice more harmful. C15 tested whether a small, bounded model could judge signed host advice from the **current legal action menu**, rather than blindly increasing BACKTRACK compliance. It is a 34→16→1 AdamW-trained additive head on frozen C12: 29 original candidate features plus five locally observable comparisons (BACKTRACK progress, best legal forward progress, relative progress, local forward blockage, legal-forward count). It cannot see global map topology or true advice correctness. Only BACKTRACK logits can change when a signed cue is live, by at most `1.5 × advice strength`; the host alone controls legality. `C15-B` additionally regularizes deviations in weak-advantage misleading-advice training cases.

The training samples were **672 SHA-matched C14 paired suffix-outcome receipts** (not newly generated counterfactual futures). The optimizer executed **308 real AdamW updates per variant, 616 total**, using 28 epochs and an original C12 frozen policy. C15-A training loss fell **0.464108→0.444722**; C15-B fell **0.464109→0.445782**. Both source-matched optimizer replays reproduced **byte-identical safetensors**.

- C15-A checkpoint SHA-256 `4aec9b65c1e7891f1e7b3493ae60fe5903df7e9af79721dc78c27842e98f90e4`
- C15-B checkpoint SHA-256 `60e3db2e27c38e849ceb8f5fae88a55ea5847be311550c22aa96c057c8414ef1`
- Frozen C12 SHA-256 `180075041fdc61b8ec0e22216780b4a7c76169829dffd43734a9d46c19be80cf`

## Held-out stochastic evaluation

Each cell below counts **successful complete trajectories** in withheld *hook/zigzag* families; directions refer to the actual desired goal, not to host truthfulness. Every comparison uses the same world seed, personality, scalar guidance arm, signed cue and exogenous random stream. Full action paths may diverge after the first decision.

| Outcome and cue | Independent 48-step: C12 | C15-A | C15-B |
|---|---:|---:|---:|
| Goal ahead, misleading BACKTRACK | 45/48 | 42/48 | 42/48 |
| Goal ahead, legitimate CONTINUE | 41/48 | 43/48 | 43/48 |
| Goal behind, legitimate BACKTRACK | 42/48 | 40/48 | 41/48 |
| Goal behind, misleading CONTINUE | 43/48 | 38/48 | 39/48 |

An earlier independent-seed 32-step panel (48 per condition) gave C15-A a legitimate BACKTRACK improvement **32→38**, but misleading CONTINUE worsened **37→32**. This did not generalize consistently at 48 steps.

To test whether an evidence-based restraint could reduce harm, a separate **C15-C** inference envelope tempered the learned C15-A correction using only the sign agreement between local BACKTRACK-versus-forward progress and the current advisory. The envelope was calibrated on a small validation split, so it is **not** an independently trained checkpoint. On a further 40-step, fresh-seed panel (96 per condition):

| Withheld hook/zigzag | C12 | C15-A | C15-C |
|---|---:|---:|---:|
| Legitimate BACKTRACK to behind goal | 79/96 | 81/96 | 79/96 |
| Misleading CONTINUE, retrace required | 81/96 | 71/96 | 81/96 |
| Misleading BACKTRACK, goal ahead | 84/96 | 77/96 | 84/96 |
| Legitimate CONTINUE toward goal | 75/96 | 78/96 | 77/96 |

**Interpretation:** learned cue reliability has not cleared the essential false-advice nonregression gate. The restraint mostly recovers C12's baseline behavior; a +2/96 continuation difference is inadequate to promote another checkpoint. C12 remains selected. Cue-on personality differences vary across panels; preserving alternative drawing style over long trajectories is still not demonstrated.

## Guards, provenance and limits

On each trained variant, **144 inactive/cancelled/expired observations** had bit-identical logits; **2,160 non-BACKTRACK score checks** matched exactly; **96 active observations** respected the bounded corrective range. Synthetic host legal menus applied every action, with no protected-material mutation or host-rejected action in the reported trajectories. The locally available suite passed **127/127** tests (120 prior/experimental plus seven new source-independent C15 core tests). The C5/C10 sources were absent from this local archive, so the **full PR exact-HEAD repository suite is not yet qualified**.

The repo-committed `cue_reliability_core.py` independently reproduced both identical checkpoints when supplied the seven immutable C12 prerequisites and C14's actual receipt JSON; checkpoints and raw trajectories are **external artifacts**, not Git blobs. Immutable GitHub Actions artifact custody remains outstanding. The research does not establish runtime CyberSand performance, host feedback reliability on unseen materials or general model personality preservation. No paid APIs or external GPU were used.

**Next discriminating experiment:** compare repeated *observed* host-cue reliability across a single continuous episode, rather than inferring trust from one-step progress. Evaluate host countermand timing and falsely reassuring local geometry on topology families withheld from training, with false-cue completion as a hard acceptance constraint. Do not promote C15-A/B/C.
