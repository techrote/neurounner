# C16: Group-robust recovery-cue arbitration — negative / retain C12

16 November 2026. CyberSand PR #639, Draft/HOLD. Model-only synthetic DRIV3R research; *not* CyberSand runtime evaluation.

## Hypothesis and experiment

C14's future-trajectory preference training improved valid BACKTRACK recovery but harmed false-cue performance; C15's locally observable reliability gating also failed to produce a robust gain. C16 uses the same 672 C14 source-hashed, action-matched paired suffix-reward observations but changes the optimizer objective: four separate goal/cue risk groups (reverse+BACKTRACK, reverse+CONTINUE, ahead+BACKTRACK, ahead+CONTINUE) are balanced by a smooth worst-group risk term, plus a penalty against drifting from C12. Ground-truth cue validity is used **only to balance optimizer groups** and never appears in inference inputs. Both variants use 34 observable features (29 existing candidate features and 5 local legal-menu summaries), 34→16→1 residual, and can alter only the host-legal BACKTRACK candidate under an active signed host cue. The frozen C12 controls the remaining candidate logits and every no-cue action.

C16-A: drift penalty 0.2; C16-B: drift penalty 3.0. Each received **140 CPU AdamW updates** over 20 epochs, with 672 receipts evenly split (168 per group). Two complete optimizer executions reproduced byte-identical output weights and safetensors metadata for each variant. Loss A: 0.652039→0.635352; loss B: 0.652041→0.636711. These loss values are training objectives, not game performance.

## Withheld topology stochastic trajectories

Hook and zigzag families held out from training. All three models (C12, C16-A, C16-B) used the same external random seeds, legal action menus and 0.8 sampling temperature; model decisions diverge as usual. Numbers are completed goals; 48 or 96 episodes per model/scenario, crossing personality and guidance strengths.

| Scenario | Pilot: C12 | Pilot: A | Pilot: B | Independent: C12 | Independent: A | Independent: B |
|---|---:|---:|---:|---:|---:|---:|
| Valid BACKTRACK, reverse goal | 35/48 | 41/48 | 41/48 | 74/96 | **86/96** | **87/96** |
| Misleading CONTINUE, reverse goal | 39/48 | 37/48 | 37/48 | 88/96 | 88/96 | 86/96 |
| Misleading BACKTRACK, forward goal | 48/48 | 44/48 | 44/48 | 87/96 | 80/96 | 82/96 |
| Valid CONTINUE, forward goal | 36/48 | 39/48 | 39/48 | 79/96 | 78/96 | 79/96 |
| No advice, reverse goal | 18/48 | 18/48 | 18/48 | 45/96 | 45/96 | 45/96 |
| No advice, forward goal | 48/48 | 48/48 | 48/48 | 87/96 | 87/96 | 87/96 |

Pilot: seed 779117, 32-step horizon; independent: seed 981117, 40-step horizon. Pilot 864 episodes; independent 1,728 episodes. No seed overlap with training, and hook/zigzag withheld geometries. Group-specific CI/control measures are not estimated from these small correlated episodes. The pilot and independent panels are both reported; no post-hoc substitution.

## Interpretation

Real gain on valid BACKTRACK cues (**+13/96** for C16-B, independent panel) is counterbalanced by loss when BACKTRACK advice is false (**−5/96**) and a smaller loss when CONTINUE advice is false (**−2/96**). The C16-A variant had an even larger false-BACKTRACK regression. Accordingly, **reject both variants for promotion and preserve C12**. The negative result is consistent with C14/C15's difficulty learning when to disregard host instructions from a locally restricted observation.

Personality turn-bias gap (opposite personalities): independent cue-active BACKTRACK C12 0.023923, A 0.078289, B 0.060705; CONTINUE C12 0.159511, A 0.174446, B 0.153169. These are imperfect aggregate style diagnostics, not evidence of long-horizon stylistic preservation. All no-cue paths are identical (same seeded random streams), as guaranteed by exact logits. All trajectory host applications were legal; protected cells were unchanged, and painting monotonic. No real material Undo is implied.

## Reproduction and source discipline

Research artifacts contain exact source and 7 prerequisite source-hashed safetensors from prior work, fresh C16-A/B checkpoints and duplicate optimizer outputs, C14 paired counterfactual receipts, full 2,592-evaluation-trajectory record, summary JSON, and 18-pair interactive HTML gallery. `python -m unittest discover -s source -p 'test_*.py' -q` passed **122/122** available archived tests including seven new C16 assertions. Local source snapshot lacks the separately committed C5/C10 and recently added C15 tests; therefore **this is not full repo exact-HEAD qualification**. GitHub Actions checkpoint custody not established.

No game/native/godot/material/map/rendering/config or production file was changed. No paid API/GPU used. No PR merge or Ready promotion.
