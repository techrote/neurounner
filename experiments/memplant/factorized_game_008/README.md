# DG008 — Held-out CyberSand-like law grafts (Draft/HOLD)

**Research question:** Can independently trained material and motion law components be combined to drive correct game-policy actions for a law conjunction that the receiving policy never saw during training?

This advances DG007's trained six-action synthetic game-policy receiver and its authentic Qwen2.5-0.5B attention-V → DG005 decoder → DG006 contextual three-law editing chain. **No game code or game assets are modified.**

## Experimental design

- True, explicitly labelled H/V/P laws are used during policy training; frozen, source-derived states replace them only at test time.
- A is an additional synthetic material/water-affinity rule; B is an additional synthetic novelty/motion rule. Both are externally observed, **not transplanted from Qwen**.
- All training and validation policy worlds have A/B = 00, 10, or 01. **No 11 worlds** appear in training. 11 is a sealed new-world generalisation test.
- Three separately trained models per seed: factorized architecture (base H/V/P network plus an independent A specialist and B specialist, logits added), flat MLP comparator, law-blind MLP comparator.
- Holdout scenario 1: an authored **additive** A+B law combination, where the modular architecture could reasonably generalise without demonstrations of the conjunction.
- Holdout scenario 2: an authored **hidden nonlinear A∧B physics** rule changing RETRACE/PAINT preference only when A=B=1. There are deliberately no examples revealing this rule. Nobody should expect it to be inferred; this is a negative control for **identifiability**, not an architecture failure to conceal.
- Genuine source-path test: freeze and verify DG002 donor, DG005 decoder and DG006 contextual/additive editors, compose eight previously withheld edits, present the two extra synthetic laws as 11, and evaluate **the exact same host-legal synthetic game worlds** under real/zero/shuffled/oracle source modes.

Two independently seeded bounded CPU trainings; new world identities across train, validation and both sealed arms. Report exact-choice accuracy, action distributions, state accuracy, original checkpoint hashes and source provenance. No trained receiver claims a native CyberSand material edit or fresh model-donor validation.

## Reproduce

Run `python -m experiments.memplant.factorized_game_008.experiment --selftest`.

For full training, restore the original source from run 43736088404, DG005 decoder from 43795069697, and DG006 parent editors from 43798092630. Execute the dedicated workflow, preserving original three neural model checkpoints and JSON report as immutable Actions artifacts.

## Interpretation boundaries

A positive result is evidence of architectural **compositional inductive bias for an additive synthetic game-rule family**, not general physics discovery. A negative result on hidden A∧B interactions demonstrates information is missing: training on single-factor cases does not identify a new pair-only physical law. The source-derived H/V/P part is supervised and uses donor families inspected in earlier experiments; no native KV transfer, independently new donor model, or native CyberSand Workbench validation is claimed.

Do not merge or mark Ready without owner review. No paid API/GPU. Strictly model-only Neurounner research; no edits to CyberSand game code/content/defaults.
