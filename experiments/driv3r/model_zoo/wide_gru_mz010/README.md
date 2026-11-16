# MZ-010 — Does a 20M wide, shallow recurrent controller retain useful game memory?

**Research only · Draft/HOLD · no CyberSand game changes.**

## Distinct architectural question

MZ-007b's eight-layer ~19M GRU learned current observations but showed no history dependence. The parallel MZ-008 credit-assignment study (#652) found that an eight-layer ~19M GRU failed an explicit old-law recall test even when a small two-layer GRU succeeded. MZ-009 (#654) then showed a 177k two-layer GRU could achieve collision-free closed-loop material-fork completion with genuinely causally required earlier cues.

**MZ-010 makes the next falsifiable comparison:** Does a *wide, shallow, ~20M parameter GRU* retain that game-memory mechanism, or does model scaling itself break training? This is **not** an 8-layer 19M rerun and is not a new language-model catalogue entry.

The ~20M candidate uses the exact MZ-009 observation schema, legal-action menu, procedural worlds and target episodes; it has a two-layer GRU with 1,504 hidden units and an observation-to-action skip. Controls are the original 177k two-layer GRU and 4.9k current-only MLP. All three receive equal minibatch indices, size (48) and numbers (800) of actual optimizer updates. **Equal examples and updates are not compute-, parameter- or FLOP-matched.** This is an architectural scaling test, not a fairness claim about hardware budget.

## Host contract and measured endpoints

- Transient material beacon at ticks 0–1 and hazard inversion at ticks 2–3 disappear before the first fork.
- Eight-step initial corridor and two further six-step corridors lead to three changing material forks.
- Both material choices are host-legal, but the incorrect one hits protected material. RETRACE permits recovery.
- Collect held-out fork accuracy, overall action accuracy, raw illegal choices, material-beacon reversals, deletion of historical cues, and DEV scores. Group all four cue/inversion counterfactuals and injected wrong-turn episodes by *seed family* within one split.
- **Execute** host-masked policies in independently seeded closed-loop worlds. Record collision-free completion, merely eventual completion, collisions, painter progression, bounded decision counts, and forced first-fork errors.
- Compare full history versus the most recent 1/4/8 observations to distinguish remembering rules from error-prone repeated attempts.
- Record isolated **single-example, unbatched** CPU 1- and 8-observation inference latency, optimizer-step time, peak RSS, checkpoint SHA-256, exact source SHA, raw loss samples, and original synthetic SVG trajectories.
- Independently reload each trained model's state dict and reproduce TEST scores from the immutable Actions artifact.

## Run

From the stacked MZ-009 branch checkout:

~~~bash
python -m unittest -v experiments.driv3r.model_zoo.material_forks_mz009.test_world experiments.driv3r.model_zoo.wide_gru_mz010.test_wide
python -m experiments.driv3r.model_zoo.wide_gru_mz010.run \
  --out mz010-evidence --train-seeds 160 --eval-seeds 24 \
  --rollout-seeds 8 --updates 800 --batch 48
~~~

A failed or unpersuasive run remains valid negative evidence. **Do not promote** any controller merely because supervised accuracy increases: long-context dependence and *collision-free* game progress must improve independently.

All changes remain within this directory and the dedicated MZ-010 workflow. GitHub Actions retains checkpoints, metrics, traces, SHA hashes and original SVG; never commit checkpoints. No paid GPU/API spend, main branch integration, Ready conversion or merge without owner permission. The inherited MZ-009 code is imported **read-only**.

## Initial underexposure pilot (negative, retained)

The first exact-source CPU study at `d37c60808efcb5476fcbe76441cb95d77da30bfc` used only 500 steps × 8 episodes, **one-sixth the examples per step of MZ-009's successful 800×48 small-GRU study**. Despite source-valid training/checkpoint verification, the wide controller reached just 55.871% fork accuracy, zero first-fork cue-flip action switches and 18.75% clean completion; the simultaneously undertrained small control reached 54.545% and 18.75%. The majority of its 93.6% all-action accuracy came from routine corridor actions. This run cannot attribute failure to architecture because *even the previously proven small controller failed* with the undersized sample budget. Its immutable original [Actions evidence](https://github.com/techrote/cybersand/actions/runs/43872182109/artifacts/14227589438) is retained, rather than erased.

The corrected MZ-010b curriculum uses the same **800 updates × batch 48** as MZ-009 and a conservative but stronger wide-model learning rate 0.002; the inherited small control remains on 0.003. Do not compare the pilot and revised run as a randomized controlled optimization trial: the training budgets and wide learning rate both differ. Evaluate the 800×48 matched-model trial on its own merits.
