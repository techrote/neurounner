# D-2 DEVELOPMENT — bounded hysteretic loop escape for the wayward drawing worm

**Status:** SYNTHETIC / DEVELOPMENT ONLY; no native material receipt, model inference, training, workflow dispatch or artistic acceptance. This is a new *post-freeze* experiment. Do not amend or promote the preregistered eight-case source-qualified confirmation cohort. M1 remains owner accepted, and M2/native integration remains separately owner gated.

**Purpose:** The previous paired scent experiment found that high ×16 soft attraction yielded an unusually small, scribbly footprint. This is an artistic character choice, not automatically a fault. Ask whether a **temporary, bounded, host-computable escape mode** can relieve local repetition without turning the worm into a purely novelty-maximizing pathfinder.

## Source and method

- Executable: worm_escape_study.py; tests: test_worm_escape_study.py. They import frozen worm_sketch.py and worm_scent_study.py, leaving both untouched. Six matched DEVELOPMENT situations: open/slats/ring × seeds 23/79, 80×80 canvas, 192 cardinal steps, identical hard-mask/soft-u16 field/start/pen rhythm and four xorshift random noise draws per decision.
- Five arms: **strong_x16** frozen strong-attraction parity control; **fixed_x1** frozen ordinary attraction parity control; **hysteresis** strong ×16 with local escape; **always_novel** low-gain/high-visit-penalty extreme control; **sham_hysteresis** logs identical state transitions but never modifies the strong policy.
- Hysteresis enters if the last 12 cursor locations contain at most 10 unique positions. Temporarily lower scalar scent gain from ×16 to ×1 and add 120 points per recent visit up to 3 visits. Exit after at least four escape steps and five novel positions, or at 24 escape steps. Eight-decision cooldown before retriggering. The rules are intentionally simple and deterministic—not tuned on a held-out cohort.
- Every outcome keeps its source-pinned recipe, original permission raster, exact scalar guidance values, ordered cursor path, legal cardinal candidate catalogue, pen-up/down records, before/after synthetic material hashes, controller mode transitions, explicit failures/STOP, and original 80×80 before/after PNG. verify_folder() regenerates from the recorded recipe and checks the full original artifacts, dependency source hashes and manifest. No UNDO_MATERIAL_EDIT is claimed: unavailable without actual native host-owned history and receipt.

Run locally from the procedural directory:

    python -m unittest -v test_worm_sketch test_worm_scent_study test_worm_escape_study
    python worm_escape_study.py --out /tmp/d2-worm-escape-development

## All 30 DEVELOPMENT outcomes

| Arm (6 scenes × seeds) | Distinct ink cells | Cursor repeats | Turns | Escape activations | Escape steps |
|---|---:|---:|---:|---:|---:|
| Strong soft scent ×16 | 666 | 410 | 345 | 0 | 0 |
| Hysteresis escape | **684** | **386** | 330 | 14 | 85 |
| Sham hysteresis | 666 | 410 | 345 | 14 (non-intervening) | 0 |
| Ordinary soft scent ×1 | 988 | 22 | 224 | 0 | 0 |
| Always novelty-biased | 999 | 9 | 250 | 0 | 0 |

All 30 episodes used the complete 192-step budget without an early STOP. Hysteresis changed trajectories in **five of six** situations; in **open seed 23**, it entered escape but selected the exact same moves as the strong control, a retained negative case. The hysteretic arm increased distinct ink by only **18 cells (+2.7%)** across six cases, despite 14 activations and 85 escape-mode steps. Larger changes from always-novel and ×1 controls come with a visibly different, less tightly scribbled line character. These are descriptive results, not scores of attractiveness or neural effectiveness.

**Interpretation:** A revisit-density trigger plus short-lived novelty pressure is viable, legal and replayable but is **too weak to substantially overcome** the ×16 scent's tendency to keep compact, dense strokes in this development cohort. Not a proven improvement in art. Increasing escape duration or altering activation threshold could change the tradeoff, but must be new explicitly labelled DEVELOPMENT parameter arms before sealed evaluation. A promising next study should vary layout families and include human inspection of line character, avoiding a metric that implicitly defines rambling as good or scribbling as bad.

## Qualification and exclusions

- Locally executed on the exact new source/test files: **22/22 tests PASS** across frozen worm-sketch (8), frozen scent (6), and new escape (8), including full event/material reconstruction, anti-teleport hard legality, source parity against frozen controls, deterministic twin runs, original-artifact verification, rehashed-tamper rejection and fail-closed parameter admission.
- This is **not the complete repository exact-head D-2 suite**; the older preregistered eight-case source-qualified confirmation regeneration is still pending.
- Local evidence index SHA-256: ef24d2804763c5aa184faa48f3f926e5df1dac64a1104a73c216832d05389172, computed from original deterministic index.json in the 30-case corpus. Git source/testing blob SHAs and precise branch head must be recorded with the PR milestone.
- Native_receipt, model_decision and native replay are absent/null, never fabricated; movement-only events preserve material. This experiment owns **only** experiments/driv3r/stage_d/procedural/**. It does not touch #611's separately owner-authorized real wormbrain work, frozen D-COMMON M1, D-1A/B or the #568 painter lanes.
