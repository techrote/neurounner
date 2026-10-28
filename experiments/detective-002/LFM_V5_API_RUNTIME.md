# DETECTIVE-002D — LFM Transformers-v5 chat-template API return-shape requalification

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#135](https://github.com/techrote/neurounner/issues/135). Follows completed DETECTIVE-002C #133 / PR #134.

002C established that Transformers 5.20.1 can construct the exact frozen LFM2.5-350M `TokenizersBackend`, but its harness then raised an empty-message `AttributeError` while rendering the same B01-B04 chat messages. The model weight was never downloaded.

Transformers v5's chat-template API returns a mapping by default when tokenizing. 002D changes only that API handling. It does not rewrite 002C.

## Frozen inputs

- Stage-A LFM source blob: `a09003d37dd3ef4f2bcd0f6a8aafd1de6457c01c`;
- merged DETECTIVE-002B battery blob: `fc3e6eb91739d74aa87a714f8e22e603bf265b16`;
- merged DETECTIVE-002C config blob: `1d42cdbefe4f15b2a68aed31f17b2b7b385e2dc4`;
- merged DETECTIVE-002C result blob: `64a599c6ac9a1b2bc2f9140ffa8a6098bf5262dd`;
- exact LFM revision: `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`;
- exact model weight SHA-256: `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`;
- exact Transformers 5.20.1 wheel SHA-256: `5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19`.

The `lfm1.0` production/game-distribution gate stays active.

## Diagnostic gate — no model weight

For all four exact DETECTIVE-002B messages:

1. instantiate the frozen `TokenizersBackend`;
2. render with `tokenize=False, add_generation_prompt=True, return_dict=False`;
3. require nonempty formatted text and retain its SHA-256;
4. render again with `tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"`;
5. require a mapping containing `input_ids`;
6. access `encoded["input_ids"]`, never `encoded.shape`;
7. retain mapping keys, token count and input-ID SHA-256;
8. require <=512 prompt tokens.

Allowed diagnostic failures:

- `template_render_incompatible` — tokenizer construction or text rendering fails;
- `tokenization_api_incompatible` — text rendering succeeds but dict/tokenization handling fails.

If either occurs: `weight_downloaded=false`, no model stage.

## Model stage

Only after all four text + mapping paths pass:

- download and SHA/size-verify the same frozen 708,984,464-byte weight;
- native float32 CPU model;
- 4 Torch CPU threads;
- same system + B01-B04 messages;
- seed 260213, greedy;
- <=64 generated tokens/task / <=256 total;
- raw output/token IDs/RSS/timing/mechanical observations retained.

Model inference uses the exact already-rendered mapping tensors; the evidence verifier requires the model-stage prompt digests to equal the diagnostic-stage digests.

## Allowed final outcomes

- `runtime_compatibility_pass`;
- `template_render_incompatible`;
- `tokenization_api_incompatible`;
- `model_runtime_incompatible`.

Any frozen-source, wheel, or weight mismatch is a hard CI failure.

## Non-authority

- optimizer steps 0;
- paid API calls 0;
- hidden relevance mapping not accessed;
- body training not selected;
- LFM production license not cleared;
- population expansion not admitted.

A PASS would show that 002C's failure was the harness/API return-shape handling. It would not erase 002B or 002C; all three remain historical evidence for their frozen harnesses.

Durable evidence target: `datasets/detective-002d/`, with independent standard-library readback and an evidence-ready no-rerun fast path.


## Executed outcome — 2026-10-28

Authoritative run: [Actions 40611095538](https://github.com/techrote/neurounner/actions/runs/40611095538), exact pre-evidence source `0fd994ea8af148c5cc5ba26503313632dfa4cea9`.

**Outcome: `runtime_compatibility_pass`.**

- exact Transformers 5.20.1 wheel SHA PASS;
- text render PASS for B01–B04;
- explicit `return_dict=True` tokenization PASS, mapping type `BatchEncoding`, keys `attention_mask` + `input_ids`;
- frozen model weight SHA/bytes PASS;
- native float32 model load PASS;
- all four prompts completed, 74 generated tokens total, no truncation;
- AMD EPYC 7763, four CPU threads, model load 0.512 s, ~1.89 GiB current RSS after load, ~2.54 GiB peak RSS;
- mean completion-wall throughput ~9.08 tok/s;
- B02 mutated supplied `03:14` memory to `03:15` with 95% confidence;
- B03 correctly changed `03:14` to stamped-record `03:41`;
- B04 answered simply `No.`;
- optimizer/API/relevance access: 0/0/false;
- body selection/license clearance/population expansion: false/false/false.

This confirms the 002C failure was the v5 return-shape harness bug. 002B/002C remain historical evidence. See [REPORT_002D.md](REPORT_002D.md).

The evidence-ready branch must now run verification-only CI; live requalification must be skipped.


## Verification-only merge gate

The live 002D result is committed and independently replayable. Final PR-head CI must:

- pass offline DETECTIVE/source-control tests;
- detect `datasets/detective-002d/manifest.json`;
- skip `live-requalification`;
- run `evidence-readback` only;
- return `DETECTIVE_002D_SOURCE_READBACK_PASS`.

No further package/model/prompt/API change is authorized in this experiment.
