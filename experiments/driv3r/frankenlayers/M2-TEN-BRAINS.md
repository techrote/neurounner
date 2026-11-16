# DRIV3R M2 — Ten distinct Frankenlayer wormbrains

**Owner direction, 15 Nov 2026:** After reviewing [GREYHOUND and BRANCHWITCH evidence](https://github.com/techrote/cybersand/blob/4143a5891c7133af10b95e7db462fb42bb134392/experiments/driv3r/frankenlayers/results-2026-11-15.md), investigate both promising shortened/repeated-layer brains and populate the GitHub Actions matrix with up to **ten genuinely distinct brains**. **Scrambled Eggs is a proven STOP-only negative control and must NOT be included** in this M2 matrix.

Source donor: exact original successful owner Frankenlayers read-only commit `4143a5891c7133af10b95e7db462fb42bb134392`; independently isolated new research branch `research/620-franken-mutagen-ten-brains`, not a write to the current PR #626 owner branch. Retain all original prior negative artifacts unchanged.

## The ten planned graph specimens

| Brain | Executed steps | Hypothesis |
|---|---:|---|
| GREYHOUND | 13 | Reproduce original near-2× speed and BRANCH attractor |
| BRANCHWITCH | 20 | Reproduce original duplicated-layer BRANCH personality |
| GREYHOUND_restore5 | 14 | Test early convolution reinsertion |
| GREYHOUND_restore7 | 14 | Test another early convolution reinsertion |
| GREYHOUND_restore12 | 14 | Test middle sliding-attention reinsertion |
| GREYHOUND_restore16 | 14 | Test later sliding-attention reinsertion |
| BRANCHWITCH_no_repeat10 | 19 | Is deliberate second execution of layer 10 responsible for its habit? |
| BRANCHWITCH_restore8 | 21 | Restore missing sliding-attention computation |
| BRANCHWITCH_restore12 | 21 | Restore another sliding-attention layer |
| BRANCHWITCH_drop21 | 19 | Further drop a late convolution, preserving grafted layer 10 |

Every flow uses exact layer numbers with final layer 23 preserved. GREYHOUND descendants remain in ascending order; BRANCHWITCH descendants preserve its **intentional 11→10 re-execution** except the explicit no-repeat control. Each is **a different executed network**, not ten copies of the same checkpoint. The donor Q8 weights are identical across flow arms; the modified decision graph is selected by the process environment. No arbitrary scrambled reordering, no Scrambled Eggs.

## Cloud experiment and evidence

- One explicit, source-checked M2 request on a separate branch triggers a **ten-arm GitHub Actions matrix with `max-parallel: 10`**, one resident d1-2B per independent GitHub CPU runner. GitHub account concurrency/resource limits may queue some jobs; no unsupported claim of all ten physically concurrent.
- Each arm builds the modified **pinned llama.cpp b11960 LFM3 decision graph**, reuses exact original pinned d1 Q8 and source-frozen W0 legal-action host; the unmodified stock decision graph is run on the same CPU for paired latency comparison.
- Genuine typed `/v1/systemone` original Q8 decisions, 16-decision budget × Greebles/Pipework/Architecture = up to **48 model choices per mutant**, original raw probability tables, STOP/refusal, CPU times, source and compiled binary hashes, and W0 real generated PNG/JSON retained per arm. A 16-decision screen is exploratory; inspect resulting art/choice diversity before extending to more seeds or 64/128 decisions.
- Do not call weight-edit-only speedups actual compute removal; these M2 variants genuinely skip or re-execute processing blocks in the graph. Compare **within-run paired stock versus mutant p50/p95**, not across-run absolute numbers.
- The previous Scrambled Eggs STOP failure stays in the original source/report, but it has **no M2 runner slot**. If these ten models collapse to the same BRANCH action, that is a negative diversity result, not license to pretend they are distinct personalities. Next separately versioned evolution can add orthogonal **MUTAGEN weight carving** to selected live Frankenlayer graphs.
- No training, native Workbench PAINT, production provider, scheduled agent, sealed FINAL data, paid external API, Ready conversion or merge. Keep Draft/HOLD.

**Workflow:** `.github/workflows/driv3r-d1-franken-m2.yml` is a one-shot push-triggered ten-CPU experiment. Its local/PR regression checks run before any model dispatch. Check original Actions logs and artifact custody before declaring results.
