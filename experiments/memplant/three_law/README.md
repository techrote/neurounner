# DG006: three-law counterfactual game-state receiver — Draft/HOLD

Research only in Neurounner; no edits to CyberSand game, native engine, Godot, Workbench, materials, maps or assets. Stacked on DG005 #237. Do not merge without owner approval.

## Question

Can an atomic-only trained contextual model compose three noncommuting, state-dependent law edits H/V/P over 2, 3, 4, 6 and 8 successive interventions, never trained as sequences? Does real source-state information matter? Can a simpler additive model do the same?

The original source is genuine frozen Qwen2.5-0.5B attention V from DG002 run 43736088404, verified digest 473cdb663922705cbddd99b57bfc5ea63048e113ec318e238574b2420511ae75. Two supervised scalar values pass through a frozen trained DG005 source decoder from run 43795069697. A third world-protection/material-law bit is supplied explicitly as a **synthetic external observation**, not magically decoded from the donor.

The three edits are hand-specified symbolic laws; the neural editor only learns single-edit transition labels for all eight three-bit states. Forty length-varied compositions are withheld. All prior source families were historically inspected, making this exploratory, not independent confirmation.

The real-source / zero / shuffled / oracle controls share all external gameplay geometries and legal-action masks. Two models are actually optimised: contextual 3-bit MLP and a strictly commuting additive baseline. Action selection is scripted, host-masked and synthetic; it is not a separately trained game policy or native CyberSand test.

Two CPU seeds each produce trained weights, SHA-256 records, atomic loss curves, and exact joint-state/game-choice accuracy by composition length. Never commit model binaries into Git.

Reproduce with python -m experiments.memplant.three_law.experiment --selftest, then restore the DG002 Qwen archive and DG005 matching seed's checkpoint, and run with --source, --reference, --out, --seed and --updates 300.

Positive results show only supervised scalar-source state decoding and compositional learned atomic operators. They do not establish heterogeneous native KV memory transfer, inferred physics or game-native behaviour. No paid API/GPU, no game edits, no merge.
