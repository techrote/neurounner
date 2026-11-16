# SPROCKET F4 — original F3 model-logit ablation (15 Nov 2026)

**Category:** source-receipted off-policy same-observation analysis of the **already executed** four × 128 F3 inference episode. No optimizer, no new model forwards, no native Godot.

Original [F3 hosted run #43769025429](https://github.com/techrote/cybersand/actions/runs/43769025429) at source `e56e0aac49b310cc1e00aafe3ffeddb90eca6e38`; combined artifact [#14183924474](https://github.com/techrote/cybersand/actions/runs/43769025429/artifacts/14183924474), original ZIP SHA-256 `5eb64e5f7d5335809f631aa11caf9edbfcf1a67f546601d5d14e795cc61f7af6`. Original W6 recovered checkpoint inventory SHA-256 `7328159322adb831f2a1b758a70740fbd39ee17be79592c660e78d48c36df98a`.

The original immutable combined archive was re-opened locally. All **65/65 ZIP CRCs**, **62/62 original per-member SHA-256 receipts**, four identities, and **512/512 actual host-applied model-conditioned choices** verified; 512 original same-observation uniform-score selections also reproduced. A new read-only exact-artifact hosted qualification workflow was created.

| Original world | F3 actual learned unique cells | Learned vs uniform score, same observed state | Learned vs *own first-step logits held constant*, same state |
|---|---:|---:|---:|
| Greebles 8111 | 59 | 31/128 | 5/128 |
| Pipework 8123 | 77 | 43/128 | 11/128 |
| Architecture 8147 | 83 | 50/128 | 10/128 |
| Greebles 8161 | 65 | 25/128 | 7/128 |
| **Combined** | **71.0 mean** | **149/512 (29.10%)** | **33/512 (6.45%)** |

Across these four original model-driven traces, raw unmasked six-token argmax is **BRANCH 346/512** or **STRAIGHT 166/512**; never LEFT, RIGHT, RETRACE or STOP. The host prohibited STOP on **504/512** decisions (ticks 0–125 of 128 for all four worlds), exposing only eight final decisions to STOP. Six original blocked-STRAIGHT observations in Greebles 8111 still resulted in legal applied actions. Original model decisions included 113 BRANCH, 72 RETRACE and one STOP.

**Interpretation:** W6 contains substantial world/family-prior signal but offers comparatively little demonstrated *local-observation-sensitive action change*, beyond holding its own opening six logits fixed. Most of its entertaining compact loops can arise from initial family tendency, host legal masking and seeded RNG. This does not establish that no local learning exists; only four learned-path worlds are assessed.

**Critical restriction:** Frozen-first and uniform choices are *off-policy counterfactuals on the actual learned-path observations*, not independent closed-loop W0 rollouts. Once a hypothetical action diverges, no subsequent counterfactual observation/PNG exists. Original W0 drawings remain in F3 and may be inspected unchanged. Never portray this diagnostic as new model inference or native drawing.

**Concrete next W7 training hypothesis:** distinguish local 5×5 occupancy/frontier, recent turning and RETRACE availability *within the same family* through host-legal paired contrast examples, rather than teaching only family-dependent BRANCH/STRAIGHT priors. Use newly seeded training observations from the support curriculum, never copy F3 held-out evaluation observations into TRAIN. Test actual optimizer and new seed-disjoint closed-loop rollouts against random/fidget/wall, flat and per-world frozen-first-logit controls. Aesthetic loop motifs, asymmetry and obstacle recovery matter more than maximum coverage. WORM experimental training gate is open independently of old Art-v2 native TRAIN/DEV60 holds.

Historical step96 negative evidence, failed W6 exit137 producer, root-Instruct revision/EOS5/rank8 scope, sealed FINAL, no native authorization and PR #576 Draft/HOLD all remain unchanged.
