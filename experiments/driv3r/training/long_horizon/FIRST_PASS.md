# First-pass defects and negative evidence (preserved)

- **Local execution v1, 16 Nov 2026:** `train.py` called a teacher/random episode with a non-null MLP pointer even though the dispatcher correctly requires `None` for non-model controls. First local full driver run failed with `ValueError: Learned policy must have a real checkpoint` after the curriculum stage. Only this incorrect call site was repaired; no preceding training work was counted as a complete result. The original failure is not a successful artifact.
- **Pilot split contamination:** initial development trials inspected `rooms` and `funnel` under a historical label `heldout` during code tuning. These results cannot count as blind final evaluation. Reclassified those families as `PILOT_EXCLUDED_FAMILIES` and introduced two untouched structural holdout families (`archipelago`, `staggered_vault`) for the final post-optimization comparison. Original pilot comparisons were intentionally not promoted into official heldout evidence.
- **Local v2 negative:** with just 135 optimizer updates and early motif teacher settings, the tiny MLP produced far fewer branch cells than the teacher, despite comparable painted length. Approximate examined 128-turn figures: DEVELOPMENT **10.75 vs 31.17** branch cells; `rooms`/`funnel` pilot **11 vs 37.67**. This was a genuine weak-line-character outcome, *not* a qualified blind heldout score.
- **Local v3 improvement without model scale-up:** after stronger motif features and 540 optimizer updates, new structural holdout showed 39.83 MLP vs 39.17 teacher mean branch cells at 128 turns, but has no real native/SLM validation. STOP-inclusive random aborted all 6 structurally heldout episodes prematurely, proving it is a weak horizon baseline rather than proof of model artistry. The final heldout is now frozen for this v1 experiment; do not retune against it.
- **Unresolved:** no original native read-only trace *bytes* were fed into this optimizer. Native integration will require original artifact source+World+material receipts, pinned model weights and a separate explicit host approval. Existing authenticated native WORM evidence can be studied as an external donor, but synthetic menu replay is never called native validation.

- **16 Nov 2026, intermediate ablation timeout:** a combined two-seed
  large-development campaign exceeded the available execution deadline
  without writing a completed report, and was **not counted as a successful
  run**. A second two-seed campaign similarly stopped with partial artifacts.
  The precise root cause of those timeouts is unverified; incomplete runs
  are excluded from the numerical conclusions. Bounded independent
  single-seed executions were completed instead, with all source and
  checkpoint hashes plus 80 original 128-step episodes independently verified.
- **Ablation limitation:** staged vs flat-128 (two seeds; eight DEVELOPMENT
  scenarios per seed) gives modest teacher imitation and branching differences
  that do **not** establish new heldout transfer, real native recoverability,
  art quality, or d1-2B/LFM/Sprocket adapter utility. Persistent random was
  more branched than both neural controls while stopping early in many worlds.
  This is a real negative for raw branch count as a quality metric.


- **16 Nov DEV forced-divergence negative:** increasing off-policy TRAIN
  intervention fraction 13%→40% yielded different line branching, but did
  **not establish stronger late-stage recovery**: forced 128-step mean ink
  remained essentially equal (125.75 vs 125.875), forced branches 22.375 vs
  25.375, and final forced-vs-intact Jaccard 0.1801 vs 0.2409.
  Small 8-world DEV trial, one seed; the stronger teacher/novelty baseline
  was not uniformly dominated. Every learned 128-decision trajectory exceeded
  native 64 logical moves. Never present this as native or SLM success.

### 16 Nov 2026 — three-intent model ablation, preserved failures

- First `test_paired_labels_include_three_intents_per_state` ran the
action-ranker matrix with wrong indexing (`X[:,20:23]` on a 3D tensor),
raising `ValueError: attempt to get argmax of an empty sequence`. Fixed to
`X[:,0,20:23]`; this was a **test-only indexing bug**; no optimizer evidence
was counted from that failed suite.
- A large 2-seed/4-TRAIN-worlds/100-stage-update attempt was interrupted
before writing an aggregate run report; even though the first seed's four
checkpoint files and partial trajectories exist, that **incomplete attempt
must not be counted as a qualified campaign**. Original partial artifacts
remain separate from the bounded run; do not promote them as complete.
- A smaller exact bounded run (one seed, 2 TRAIN worlds per family, 25
optimizer updates per stage) **completed** and independently revalidated all
176 episodes and 15,424 original action decisions. Four actual checkpoints
and source/checkpoint SHA-256 are in the external run bundle, not Git.
- Original single-intent conditioned models failed to vary geometry across
three model-side commands (128-turn ink Jaccard=1.0). Counterfactual teacher
labels improved command responsiveness, particularly with an explicit
handcrafted action×intent interaction representation. That representation
then over-turns nearly every motif move (0.996 motif turn fraction), a
potentially trivial zigzag rather than demonstrated interesting artistry.
