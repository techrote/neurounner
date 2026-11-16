# DG015 — neural donor material-routing regret and dynamic recovery

**Draft/HOLD.** Research-only; stacked on DG014 PR #247. No CyberSand production changes.

DG014's material-law donor swap left completion virtually unchanged. This experiment tests **true-law weighted routing cost** as well as completion, removes both precomputed obstacle-aware shortest-path features, and adds a gate closing after two moves. A Dijkstra-supervised receiver must navigate from a full 9×9 material/obstacle map, paint at the correct start/goal under the inherited rule, and optionally RETRACE after the new obstruction.

Two independently trained seeds; each trains twelve neural donors (six narrow, six gated privately permuted specialists) and two map-input receiving policies. The receiver learns on narrow donors across seven triples; the complete (+++) triple and gated donor architecture are withheld. On 100 sealed procedural mirrored/rotated corridor worlds per seed, compare correct neural donors, wet-law donor swap, a source-blind receiver and a same-world reactive Dijkstra oracle. Record exact source hash, checkpoints, true-law cost, **regret conditional on successful completion** (and its denominator), wet-swap first-action change, RETRACE counts, illegal actions, original trajectories and file SHA-256 hashes.

**Scientific limits:** Worlds are procedural variants of a deliberately small two-lane corridor, rather than arbitrary CyberSand maps. Donors learn authored law functions, receiver imitates a Dijkstra expert. Calibration uses neural *outputs*, not transplanted weights/KV. This cannot establish autonomous physics discovery. If material swaps fail to improve path cost, retain the negative finding. If retrace remains unused, report it. Successful-oracle completion need not imply low regret. No paid GPU/API, no merges or Ready promotion.

Reproduce with the dedicated workflow or:

~~~bash
python -m experiments.memplant.trajectory_transfer_015.experiment --selftest
python -m experiments.memplant.trajectory_transfer_015.experiment --seed 26011626 --out /tmp/dg015
~~~
