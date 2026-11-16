# DG017 — on-policy corrective game-policy inheritance

**Draft/HOLD research only.** Stacks on DG016 PR #249 at baseline a1ab1ab85d78699368a6af676f9e646edf312bb8. Do not merge or promote without owner acceptance. No CyberSand production source, content, provider settings, or game implementation may be changed.

## Hypothesis

DG016's inherited source-blind gap demonstrated goal inheritance but only 28–34% OOD completion on unseen random mazes. A receiver trained on *its own visited states* with authoritative legal-action corrections may recover from loops and post-gate deviations better than one trained solely on expert rollouts.

This experiment adds an observable per-cell visit-history channel (channel 13), initializes the receiver by **exact output-preserving neural weight surgery** from a legacy 12-channel receiver, and compares:

- legacy: DG016 architecture and behavioural-cloning objective, 12 channels
- history_bc: same neural receiver initialization plus the added visit-history channel, trained on fresh expert trajectories
- dagger: history-aware receiver additionally corrected using states visited by its own policy
- blind_dagger: equally corrective-trained receiver without neural donor signatures
- wrong_wet and wrong_goal: correct game physics/world but counterfactual specialist-donor outputs
- oracle: Dijkstra replanning with host legality and timed gate closure

Correction batches contain the oracle's action for the **receiver's actual current state**, with a bounded teacher-action mixture (20%, then 8%), repeated-state deduplication and a 20-state-per-episode cap. Corrective learning samples 50% fresh expert states and 50% aggregated corrected states; the history-aware behavioural-cloning control prevents attributing any gains from the new observable channel to correction. Conditioned vs blind correction isolates neural donor inheritance from ordinary off-policy recovery.

## Hard boundaries

- Independent narrow/gated donor neural networks are trained on wetness, goal and painting-law families with private feature permutations and reflections.
- Frozen donor output probes are functional alleles; this is **not** evidence of raw weights/KV transplantation between unlike networks.
- All cells, material penalties, walls, goals, protected paint cells, host legal action masks, gate-closure timing, and RETRACE are in an external synthetic harness.
- Training excludes the full (+1,+1,+1) three-law combination and gated donors. IID and OOD sealed worlds use distinct seeds and gated donors.
- No game source or game assets are modified; no paid API/GPU use.

## Primary metrics

Report all-world goal+painting completion, failure-penalized cost, successful-world cost, RETRACE count, legality, withheld-triple completion, and matched-world DAgger-vs-history-BC regret. The wet counterfactual test *preselects cases using differing original-vs-alternative-law oracle first moves* and reports its actual denominator (which may be zero). No inheritance claim may be derived from completion alone.

Keep genuine negative results, including potential DAgger degradation, goal specificity without wet-law adaptation, synthetic-oracle dependence, and combined topology/timing OOD confounding.

## Reproduce

Run with pinned CPU toolchain:

~~~bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -m pip install numpy==1.26.4
python -m experiments.memplant.trajectory_transfer_017.experiment --selftest
python -m experiments.memplant.trajectory_transfer_017.experiment --seed 26011630 --train-worlds 200 --updates 500 --sealed 100 --horizon 96 --donor-steps 140 --rounds 2 --correction-worlds 65 --out /tmp/dg017
~~~

Immutable GitHub Actions artifacts retain all donor weights, all five receiver checkpoints, original SVG/JSON trajectories, correction-label denominators, reports and SHA-256 manifests. Results cannot be claimed qualified until the exact-head workflow and independent hash checks pass.
