# SPROCKET WORM-001 — verified W4 complete recovery and S1 learned-logit sampling

**15 Nov 2026 UTC.** [Draft/HOLD #576](https://github.com/techrote/cybersand/pull/576), branch `codex/agent-painter-sprocket-568`. Authoritative [WORM #612](https://github.com/techrote/cybersand/issues/612); old whole-Art experiment remains separately negative, sealed FINAL uninspected. This report does **not** authorize merge, provider promotion, native access or a model freeze.

## A. W4 original source and immutable experimental receipts

- Original genuine W4 optimizer **source** `c14795efe978dadeaa9e0da60f6273e18ab17b4d`, [CPU run #43710874563](https://github.com/techrote/cybersand/actions/runs/43710874563): 96/96 previously independently qualified AdamW steps and immutable `checkpoint-000096`, original [artifact #14161217282](https://github.com/techrote/cybersand/actions/runs/43710874563/artifacts/14161217282).
- Original failed 5/6 model-image attempt [#43715212047](https://github.com/techrote/cybersand/actions/runs/43715212047) remains **FAILED**, never relabelled.
- Source-fenced **zero-optimizer W4 recovery** at commit `7953f88bf9d62c116de36537c03f611ba8655517`, [run #43721058235/job #132832927533](https://github.com/techrote/cybersand/actions/runs/43721058235/job/132832927533) **SUCCESS**, original [artifact #14165042291](https://github.com/techrote/cybersand/actions/runs/43721058235/artifacts/14165042291), downloaded anew: ZIP **101,914 bytes**, SHA-256 `87d4c797698effaf6ed90151cf2ecb050c7bc685f4d810ccdbee75737016f90b`, **57/57 ZIP CRC clean, 56/56 embedded SHA-256 checks PASS**. Original completed `results/comparison.json` read directly; `results/partial-progress.json` retains the intentionally provisional status, so only the final comparison is authoritative for completion.
- Six actual original W0-source-faithful trained 96×96 PNG/JSON pairs plus 18 matched random/wall/fidget pairs independently inspected, **24/24 total trajectories complete**. All six W4 trained streams have **24/24 raw legal EOS5 choices**, **144/144 LEFT**. Two distinct trained PNGs across the three theme labels and two obstacle seeds; family labels produce no distinct learned shape within a seed. Each trained trace has **4 unique cells, 2×2 bounding box, radius2, 21 revisit contacts**. Original matched controls on each seed: random **19/21**, wall **25/25**, fidget **21–22** unique cells. Zero training steps in recovery. No new native painting.
- Local original-PNG contact sheet is an **unaltered-image montage** (only nearest-neighbour scaling, captions and layout), SHA256 `0a67086a9ead7c9750c12a91667a0f883b201300fcbdb6b8a7e96c2633aa9840`. Source PNG bytes remain in immutable artifact, not embedded in this repository.
- **Outcome:** W4 complete recovery PASS; useful steering and learned artistic novelty FAIL. Greedy single-token decoding still collapses onto LEFT.

## B. New S1 real model-score experiment, not a new model checkpoint

W4 greedy is now a fully qualified, honest negative control. Rather than spend another ~30 minutes of optimizer compute on the same steering objective, S1 measures **actual pinned W4 next-token logits** for all six existing W4 single-digit action tokens at **each current W0 observation**, then lets the **host** mask illegal actions and sample a legal action with deterministic SHA256-derived uniform draws. Temperatures 1.0 and 3.0 are precommitted. Every score, legal set, unmasked top class, post-mask probability, draw, selected/applied action, effective action count, current-observation hash, 24-action trajectory, original PNG, revisits, bbox and radius is persisted. This is explicitly **not** raw greedy EOS5 text generation; prior W4 greedy first passes remain the original baseline. No teacher fallback, scene compiler, model-owned material/canvas mutation, FINAL, checkpoint overwrite, or changing W0.

New files exclusively within SPROCKET paths:
- `worm_score_sample_v1.py`, `test_worm_score_sample_v1.py`;
- `.github/workflows/agent-painter-sprocket-worm-score-s1.yml`;
- `worm-eval-request-s1.json`, the **sole one-shot trigger**.

New hosted CPU evaluation [#43726271494](https://github.com/techrote/cybersand/actions/runs/43726271494), exact execution/request commit `b0a68d45f54ba5f4d9b6b11c505f80d51cc8eb3b`; same pinned root Instruct `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, LoRA 182-projection W4 checkpoint, W0 commit `4bb9d6e1d0d0d78183f87d1e3c29648edafacf61`, W4 original training provenance `c14795efe978dadeaa9e0da60f6273e18ab17b4d`. Ordinary GitHub-hosted CPU with one model process and finite deadline; 12 learned trajectories (3 families × 2 seeds × 2 temperatures), plus 18 same-seed controls, 24 decisions maximum each. An always-uploaded partial artifact, failure receipt and SHA list are required.

At writing, source/host-policy and sampler tests passed in GitHub; model installation/evaluation was active. **No S1 completed model decisions, PNGs or artistic success claimed until original output passes verification.**

## C. Decision boundary

1. If sampled action probabilities retain essentially all LEFT mass, report that and avoid treating induced random novelty as learned steering.
2. If temperature increases image/path variety, compare actual action entropy and per-state dependence against uniform random, and visually inspect originals; higher entropy alone is not creativity.
3. Use the independent SPROCKET support-v2 curriculum and its group/split checks for an explicitly versioned **future** training stage only after this real scoreboard is examined. Keep all existing W1–W4 and full-Art evidence untouched.
