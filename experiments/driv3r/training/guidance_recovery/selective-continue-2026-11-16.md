# C12: selective refusal of misleading CONTINUE countermand

**Research-only; Draft/HOLD.** 16 November 2026. Owned path: `experiments/driv3r/training/guidance_recovery/**`.

## Question

C11 showed that strong signed feedback can increase countermand compliance at the cost of valid goal reversal. Can a frozen C11 worm learn a **selective veto** of a false `CONTINUE` instruction while still following true continuation advice? The host maintains material permission authority; advice never authorizes paint, erase or movement outside the candidate legal menu.

## Implementation and actual optimizer

A `29→16→1` PyTorch residual scores only the host-legal BACKTRACK candidate when the host has explicitly issued a `CONTINUE` cue. Frozen prerequisite is the C11 bounded checkpoint. Its no-cue and genuine BACKTRACK-cue logits, and the nine non-BACKTRACK logits in every condition, are structurally identical. Contextual observations remain those of the existing synthetic 15×15 worm simulator.

New **6,912 matched supervised candidate menus** from hairpin/staircase, all three guidance arms (0/.35/1), two personalities, rotation and mirroring, four depths, two cue strengths and both true/misleading CONTINUE contexts. **540 CPU AdamW steps** (10 epochs); loss `0.415340 → 0.183440`. The held-out `hook` and `zigzag` families were *not* C12 training examples, though the frozen predecessor was previously trained on some topological information. Do not call this wholly novel topology transfer to the whole model stack.

Selected C12 checkpoint SHA-256: `180075041fdc61b8ec0e22216780b4a7c76169829dffd43734a9d46c19be80cf`. The complete original training/evaluation run and an independent compact training-source replay produced an identical byte-level checkpoint and loss sequence. All predecessor checkpoints and hashes, synthetic source receipts, and raw rollout data are bundled externally, not committed to Git history. The committed `selective_continue.py` reproduces the checkpoint exactly; the expanded evaluation and SVG gallery builders are preserved in the external source bundle.

## Held-out positive and negative evidence

First-action test from 8 fresh seeds per family (384 matched candidate menus per pooled outcome/family group):

| Pool / required first action | C11 | C12 |
|---|---:|---:|
| Trained geometries / BACKTRACK after misleading CONTINUE | 148/384 | **374/384** |
| Withheld geometries / BACKTRACK after misleading CONTINUE | 76/384 | **314/384** |
| Trained geometries / productive CONTINUE | **384/384** | 380/384 |
| Withheld geometries / productive CONTINUE | 384/384 | 384/384 |

A **separate fresh-seed** 1,152-episode stochastic evaluation (576 per model, 18-decision horizon, host-executed legal menu, source and RNG matched) found:

| Outcome / cue timing | C11 | C12 |
|---|---:|---:|
| False CONTINUE, reversal still required; tick 0 | 66/144 | **92/144** |
| False CONTINUE, reversal still required; tick 2 | 90/144 | **108/144** |
| True CONTINUE to viable exit; tick 0 | 122/144 | 124/144 |
| True CONTINUE to viable exit; tick 2 | **128/144** | 125/144 |

Thus, reversed goals completed **156/288→200/288**, and legitimate continuation was effectively flat, **250/288→249/288**. Within withheld hook/zigzag, misleading-countermand recovery improved **64/144→85/144**. None of the 1,152 fresh trajectories selected an illegal raw top-1 action, was rejected by the synthetic executor, or modified protected materials. No native claims.

**Negative:** On the first smaller 576-trajectory evaluation, valid CONTINUE completions fell 135/144→126/144, demonstrating sensitivity to sampled seeds and path stochasticity. The separate larger fresh cohort improved this to a one-completion net loss; do not hide the early regression. Opposite-personality turn separation under *active* CONTINUE varies materially between topology/timing cohorts, and has not been proven stable in long-horizon drawing. The learned residual improves decisions on the particular synthetic curriculum; it has no demonstrated transfer to resident d1-2B worms, actual Workbench maps or real material Undo.

## Qualification and custody

- Local **101/101 available offline tests passed**, including five C12 checks. Earlier C5/C10 tests are not in the archived local source; the repository-wide exact-HEAD suite is **not qualified**.
- An inactive/positive-cue score guard checked 48 bit-exact inactive states and 48 bit-exact BACKTRACK-cue states; 1,296 non-BACKTRACK scores matched exactly across candidate comparisons.
- Predecessor source blobs (`experiment.py`, `topology_transfer.py`, `persistent_feedback.py`, `signed_feedback.py`) were matched against GitHub commit `fef46523dc6ad61fa699d7794b11047f44f3d77b` before C12 training.
- Immutable GitHub Actions artifact custody is **outstanding**; the binary is provided only in a SHA-256-manifested local archive. No paid GPU/API, model dispatch or game edits.

**Disposition:** Hold. Next question is whether the policy can preserve long-horizon stroke personality under active selective advice while responding to time-varying cues in more varied, contradictory spatial worlds.
