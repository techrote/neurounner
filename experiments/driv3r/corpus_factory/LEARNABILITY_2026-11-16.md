# TRAIN-internal CPU learnability diagnostic — 16 November 2026

**Executed local research, isolated to experiments/driv3r/corpus_factory/**. PR #638 remains Draft/HOLD. **Not official DEV/held-out, not native gameplay, not closed-loop model drawing, not Sprocket/LFM optimizer work.** Four finite CPU SGD fits were genuinely run. Ephemeral linear-model weights were not committed.

## Source and data integrity
- Diagnostic: learnability_probe_scoped.py. Git blob 78d8c899e73d692ffe0540f6ac24dd0e173c38e0. Executed local source SHA-256 29527f7bbeac351ea48d7f3b8c1c55c618b8b089d55a80cadeada30c1222473f; independently matched to Git blob.
- Verified prior TRAIN-only source ZIP SHA-256: 80b11c731e29306dd04c0b09539ab8fc7796592933276a13e7e44e83f17f6dbc (4,313 synthetic heuristic messages / 144 TRAIN world groups).
- Verified prior experimental counterfactual ZIP SHA-256: ae3bcb2c9d06fca4ab69f427dba5e859da4638882850a26fe14a4103b97c2d2e (1,440 *unexecuted*, artificially conditioned single-step preference messages / 131 TRAIN groups).
- NumPy 2.3.5; scikit-learn 1.8.0; SGDClassifier(log_loss, alpha=0.0001, max_iter=24, tol=None, random_state=17, shuffle=True). Reproduction partition namespace diagnostic-worldfold-v1: with no world-group overlap and zero exact observable collisions across folds. All sparse features are derived from prompt-visible observations only; host legal masks apply to predictions.
- Original ZIPs are prior-run **local** exports and have NOT been immutably archived in GitHub Actions. No hosted-model qualification claimed.

## Actual trained-model results

| TRAIN-only corpus | Family visible to model | Train rows | Internal eval rows | Independent eval worlds | Legal-masked accuracy | Legal-majority baseline |
|---|---|---:|---:|---:|---:|---:|
| v1 heuristic | No | 3,315 | 998 | 36 | **82.465%** | 18.036% |
| v1 heuristic | Yes | 3,315 | 998 | 36 | **82.665%** | 18.036% |
| Artificial preference probe | No | 1,134 | 306 | 30 | **29.412%** | 33.333% |
| Artificial preference probe | Yes | 1,134 | 306 | 30 | **84.641%** | 33.333% |

Each of the four fits executed 24 SGD iterations; samples-seen receipts are 79,561 for each v1 fit and 27,217 for each probe fit. On **102** matched, unseen-world equal-state probe triples, target labels diverge 102/102, predictions diverge **0/102 without family** and **102/102 with family**. Original v1 has 45 complete equivalent triples with **0/45 source label divergence** and **0/45 model prediction divergence**.

The output JSON (original local SHA-256 **4c8dec4d00e381c0558e89faebc35ce6dd5829a0db3a8793f62abd964e4cd31f**) was reproduced byte-identically by a second independent run. Two deliberately corrupted probe sources (wrong target vs provenance; forged host-application flag) were rejected locally. No native event, official DEV test or model checkpoint promotion occurred.

## Conclusion and handoff

The v1 synthetic teacher is highly learnable for single-step legal actions but provides almost no family-conditional action signal. Artificial conditional labels can be learned by a tiny real CPU optimizer; this **does not establish artistic quality or closed-loop competence**. Their source evidence class is unexecuted counterfactual supervision, never native/host-applied action success.

The existing new family_closed_loop.py is independently executed synthetic host-legal family steering (60/60 distinct three-family world footprints), but retains a genuine negative result: **0/180 goals visited**. Preserve goal/recovery curriculum and require future primary-model comparisons on equal-seed non-TRAIN worlds, including separate goal/recovery and motif divergence metrics.

Reproduce with the two SHA-pinned ZIPs and the committed script:

    python3 experiments/driv3r/corpus_factory/learnability_probe_scoped.py --v1 /path/to/cybersand_game_corpus_v1_train_only.zip --probe /path/to/cybersand_corpus_family_contrast_probe_v1.zip --output /tmp/cs-learnability-run.json --epochs 24

Only research scope was touched. No CyberSand game modifications, paid compute, new production settings, duplicate LFM/Sprocket optimizer, merge or Ready promotion.
