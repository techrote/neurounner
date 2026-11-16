# D-2 synthetic topology fragility follow-up (15 November 2026)

**Research status:** DEVELOPMENT-ONLY diagnostic on the **historical provisional** eight-case confirmation archive; **not source-qualified confirmation, native World execution, art acceptance or a trained-model result**. No frozen policy, fixtures, or manifest were modified.

## Evidence custody

- Archived input ZIP SHA-256: `b7048f1e6dcb27956d6cb23b96aea57f84c5f7cf5f7e7c54bfae9deb10186b03` (the original provisional contact/evidence ZIP retained outside Git).
- The source report self-identifies as `UNQUALIFIED_LOCAL_POLICY_REIMPLEMENTATION`, with `policy_source_verified=false`. This status **must not** be upgraded by downstream image analysis.
- Standalone stdlib analyzer: `fragility_study.py`; no dependency on the frozen `pareto_study.py`, fixtures or native/backend machinery. It checks source raster/view SHA-256, PNG chunk CRC, hard-permission pixel invariance, deposition-only semantics and matched stored component/edit counts. It deliberately does **not** independently reexecute the frozen policy or validate native history.
- Report generated from the original ZIP: SHA-256 `30cef168f98bf91a5883f04a6d3f2908bc37aea2c7fdbf00a65c124cc329bc0d`, eight geometries, five policies and matched controls, 40 policy/control pairs. Report is intentionally kept out of source Git; regenerate with `python experiments/driv3r/stage_d/procedural/fragility_study.py --archive /path/to/original-provisional-evidence.zip --out /tmp/fragility-report.json`.
- Repeated report generation was deterministic. Six isolated local tests passed against locally SHA-1-Git-blob-matched versions of the two committed files: `fragility_study.py` `cb1fdf1c5e4fdea1763351aa081de03f31e89cba`; `test_fragility_study.py` `52f53ca25f29ecd3be06d12c3e75472adf6f28cc`. The ZIP-backed test skips in source-only checkouts, where that original external archive is unavailable.

## Observed structural failure

Counting connected material components alone exaggerates the usefulness of the existing thin bridging policy. Under the historical *provisional* weighted-0 arm, the six geometries with any new material paint **191 total cells**. All **191/191** are **articulation points of the final four-neighbour material graph**: removing *any one* individually increases the graph's connected component count. Five cases reached one component, but on a maximally one-cell-fragile bridge. The sixth, spiral-islands, improved five components to three but did not connect all islands.

| Provisional weighted-0 geometry | Fresh cells | New articulation cells | Final components |
|---|---:|---:|---:|
| diagonal-fence | 26 | 26 | 1 |
| nested-courtyard | 29 | 29 | 1 |
| paired-vaults | 37 | 37 | 1 |
| staggered-shelves | 0 | 0 | 2 |
| misleading-scalar | 29 | 29 | 1 |
| spiral-islands | 29 | 29 | 3 |
| one-cell-choke | 41 | 41 | 1 |
| hard-separator | 0 | 0 | 2 |

This measures **graph fragility**, not physical breakage, gameplay passability or artistic quality. Matched greedy controls often create their own narrow dangling extensions but, in these cases, do **not** connect the initial components. The trade-off is not evidence the baseline is better.

## Next decision experiment

After independently executing the **exact existing frozen policy blob** and rerunning its eight preregistered cases (still outstanding), treat articulation fraction as a *separate* measure of useful structure. Under a **new, explicitly labeled post-freeze development policy**, compare narrow topology bridges with 2-cell-wide/redundant-route reinforcement and its equal-edit controls. Log initial component reduction, individual-cell articulation points, material area, soft-guidance violations, cursor travel and bounded cost; retain the negative hard-separator and budget failures. Do **not** retrofit the frozen policy after viewing these outcomes.

Native model inference, material Undo, Actions matrix dispatch, training, GPU/paid API expenditure, merge and Ready remain outside this lane's authorization.
