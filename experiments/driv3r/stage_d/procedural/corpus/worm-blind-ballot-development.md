# D-2 — blinded visual-ballot accounting (DEVELOPMENT only)

The preceding [assessor-blind gallery](worm-blind-gallery-development.md) contains **36 paired synthetic raster comparisons**, including 12 identical-image shams. No real ratings have been collected. The new `worm_blind_ballot_analysis.py` accepts an independently completed anonymous ballot only **after all 36 items are rated**, verifies the *source-regenerated original PNGs, review inventory and sealed custody*, and then produces a deterministic analysis JSON.

## Procedure

1. Give a reviewer **only** the existing `review/` packet. Keep `custody/assignments.json` separate until the review is finalized. Use the same palette and shared context on both sides. Do not disclose the pulse/seed/policy or source names.
2. Reviewer fills a **copy** of `review/ballot.json`: set `status` to `RATED`, choose `left/tie/right` for `preference`, `none/small/clear` for `difference`, and optionally `left/tie/right` for `weirdness` (or leave it `unrated`). Preserve the anonymous IDs, filenames, order, header and every other field.
3. Once complete, use the original complete gallery root (both `review/` and separately held `custody/`) to unblind and account for all ratings:

```sh
python worm_blind_ballot_analysis.py --gallery /path/to/original-gallery --rated-ballot /path/to/sealed-filled.json --out /path/to/new-analysis.json
python -m unittest -v test_worm_blind_ballot_analysis test_worm_blind_gallery_study
```

The analyzer rejects partial/renamed/reordered ballots; forged extra fields; modified images and custody even with rehashed claims; mismatched review protocols; and existing output paths. The result SHA, complete-item count, provenance hashes, pulse-group preferences, visual-difference distributions, identical-image disagreement flags, sham false-difference counts and original item-level records are retained. There is **no silent filtering** of shams, unchanged interventions, unreliable votes, negative outcomes or tied scores.

## Interpretation limits

The actual evaluator has **not** scored any images; the regression tests use invented all-tie/all-left ballots solely as deliberately artificial fixtures, never research outcomes. A preference for a side is a descriptive subjective vote, **not** evidence of better model control, native accuracy, learnt behaviour or artistic superiority. The 36 cases reuse only 12 geometry/seed pairs, so treat observations as clustered, not independent; avoid inferential significance claims. Report all sham and identical-image disagreements explicitly. The tool does not set a discretionary post-hoc pass threshold or select a winning steering strength.

Source/test qualifications are bounded offline Python tests on exact Git blobs. The broader D-2 exact-head suite and historical eight-case frozen-policy source confirmation remain outstanding. No owner authorization for native D-2 material integration, model inference, training, Actions dispatch, merge or provider change is conferred.
