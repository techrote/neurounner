# DRIV3R D1-MUTAGEN M0 — Rusty-Spoon First

**Status: Draft/HOLD; standalone first-pass research under [#620](https://github.com/techrote/cybersand/issues/620).** This is an isolated d1-2B *weight-editing* and actual typed-decision evaluation pilot, deliberately **not** a new CyberSand feature, nor a change to the current DRIV3R resident model, accepted Workbench, or stock-model speed-tuning lane.

## Purpose

Ask whether **coarsely carved sections of the *actual* original d1-2B Q8_0 GGUF neural weights** change observable worm-steering intent. This is a deliberately quick anatomical sensitivity experiment: no gradient attribution, optimizer, distillation, learned restoration, native Godot build, or additional safety/permission role granted to the mutated model.

### Three first specimens

| Name | Chosen actual GGUF anatomical groups | Crude operation |
|---|---|---|
| `mlp_scoop` | Layers 8–15, `ffn_gate/up/down` | Reduce first quarter of chosen tensors to 25% of original magnitude |
| `conv_scrape` | Convolution layers 1,5,9,13,17,21, `ssm_in/out/conv1d` | Zero first half of chosen convolution-family tensors |
| `attn_gouge` | Global-attention layers 2,6,10,14,18,22, Q/K/output projections | Reverse first quarter of chosen tensor values |

Actual eligibility is established from the **pinned GGUF tensor inventory**, not assumed from abstract model dimensions. If a named category is absent, an experiment **fails explicitly with its inventory**, not a substitute random target. Q8_0 modifies only the 32 signed integer codes within each 34-byte block; the original FP16 scale and GGUF framing remain untouched. If eligible F16/F32 tensors occur, modify their numerical values without shifting metadata. The source GGUF is immutable and the separate mutant is transient, re-creatable from recorded recipe/hash.

**Important:** reducing/zeroing quantized weight codes does not remove matrix multiplication, so no CPU speedup is claimed here. [#619](https://github.com/techrote/cybersand/issues/619) owns subsequent true compute-skip/packed-quantization engineering, informed by these sensitivity observations.

## Original real decision evaluation

Source-pin original `experiments/driv3r/pins.json` (the exact Q8 model SHA256, official upstream revision, original pinned llama.cpp b11960 binary SHA) and source-qualified W0 legal action host `4bb9d6e1d0d0d78183f87d1e3c29648edafacf61`. Use the *same* seed 570 and W0 geometry for original and mutants. The original stock model chooses up to 4 live W0 decisions; each mutant up to 8 decisions for Greebles, Pipework and Architecture. These are tiny **initial exploratory screens**, not generalizable evaluation.

Question input is a typed `worm_move` choice with host-supplied legal action descriptions, W0 position, heading, visited mask, local 5×5, history and theme. Answer comes directly from the original **`/v1/systemone` API**, not fabricated heuristics and not a generative prompt/completion model. Retain exact source choice/probabilities, all original error replies, wall time, source model and mutated hashes, and derived W0 PNG/JSON traces. If the model STOPs early, accept the original STOP. If a model/runtime fails, preserve error evidence. **These PNGs are source-faithful Python W0 renderings, NOT native Godot PAINT or actual Workbench material readback.**

## Cloud execution and safety

- Standalone branch `research/620-d1-rusty-spoon-m0` stacked on the untouched DRIV3R worm pilot `research/570-wormbrain-native-pilot`. No existing model/agent branch edits.
- `.github/workflows/driv3r-d1-mutagen.yml` uses a **cheap PR-source test** and a separately gated exact `run-request.json` push-trigger for hosted CPU model experiments. Three finite jobs, maximum **two ordinary GitHub-hosted VMs concurrently**; source, model identity, GGUF structure, output SHA and complete text results are retained in Actions artifacts.
- No training, paid inference API, external Web service, native Godot writes, sealed FINAL data, default-provider change, model promotion, merge or Ready conversion. No automatic repeating schedule.
- Failures are admitted. In particular, malformed GGUF outputs, no anatomically eligible tensor, nonfinite choices, early STOP and useless drawing results are legitimate negative results; no silent alternative policy or reward-based cherry picking.

## Expected handoff

The first actual result is a table of **which selected tensor categories are very sensitive versus surprisingly tolerant** of deliberate coarse edits, alongside some genuine typed directional-choice transcripts and W0 trail PNGs. The owner can judge visual eccentricities. #619 can use this to choose which groups to skip/remove/compress for actual speed.
