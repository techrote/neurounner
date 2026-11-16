# MZ-015 — Direction-balanced turning-map training

**Draft/HOLD. Research only.** Parent: MZ-014 PR #659 at `650bad65c76c569bb5fb33c784e72a9c16d654d6`; model definitions inherited from MZ-013. No native/Godot/Workbench/game or production provider edits.

## Research question

MZ-014 established that the 209k diagonal-state policy failed 12/12 mirror-X worlds despite 12/12 original-orientation successes. Does training explicitly on left/right/up/down and mirrored *turning spatial paths*, rather than a single left-to-right corridor, produce useful learned directional transfer?

## Implemented experiment

- Original synthetic 33×33 grid, obstacles, one-cell route, five straight segments and four route turns, three painted-material stations, safe/protected mineral forks, delayed cue and inversion, host-legal movement/PAINT/RETRACE/STOP.
- Eight equally represented coordinate frames (four rotations, mirror on/off) through train seed families. Map lengths, mineral assignments, direction and orientation vary. Held-out test/shifted-length seeds do not overlap training.
- Same 155-dimensional input and controllers as MZ-013 (diagonal 209k, GRU 237k, reactive 21k). Same minibatches and optimizer updates for fair *sample exposure*; differences in compute/hyperparameters remain.
- Correct material entry, PAINT, RETRACE and recovery behaviour are represented; one wrong-mineral entry is injected in some teacher episodes.
- Raw illegal output counting is separate from host-legal masked closed-loop execution. Closed-loop clean completion, collisions, PAINT, recurrent-history ablation and cached/full parity distinguish robust control from recovering via repeated collisions.
- Source-stamped checkpoint, CPU inference latency and process RSS, original SVG trajectory gallery, metric JSON and update-by-update loss traces go to immutable GitHub Actions artifact (not Git).

## Important limitations

This is a *procedural turning-path* host, not arbitrary free-space planning, dynamic fluid/terrain simulation or actual CyberSand. Rotating/mirroring shared route grammar does not demonstrate generalization to all unseen topologies. No promotion is automatic. Prefer 12/12 clean across orientations and unseen geometry, no illegal logits, and causal memory benefit before proposing deeper study.

Run: `python -m experiments.driv3r.model_zoo.turning_maps_mz015.run --out mz015-evidence --updates 600`

No paid resources. Checkpoints stay in Actions artifacts. **Do not merge, mark Ready or integrate in Workbench.**
