# DETECTIVE-002B — seven-body no-training runtime/dialogue smoke

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#130](https://github.com/techrote/neurounner/issues/130). Source authority: merged DETECTIVE-002A #127 / PR #129.

This slice characterizes **runtime viability and raw short-dialogue behavior** for seven compact conversational bodies. It does not train, rank, admit or select any body for detective-NPC use.

## Frozen cohort

The runtime runner consumes the committed Stage-A source archive directly and additionally pins its Git blob identity. Floating upstream refs are forbidden.

| Body | Class | Frozen runtime role |
|---|---|---|
| SmolLM2-135M-Instruct | existing control | current compact baseline |
| SmolLM2-360M-Instruct | direct candidate | same-family scale comparison |
| Sprocket-500M Chat | existing control | chaotic dialogue body |
| Sprocket-500M root Instruct | existing control | distinct instruction body |
| Qwen2.5-0.5B-Instruct | direct candidate | cross-family instruction body |
| Qwen3-0.6B | near-band exception | thinking-disabled matched dialogue |
| LFM2.5-350M | direct candidate | compact LFM research runtime |

LFM2.5-VL-450M remains a separate llama.cpp runtime reference. Zenyx remains mutation-protocol-only. Neither is forced into this Transformers matrix.

Sprocket Chat and Instruct share repository/config/tokenizer revision but **must load different frozen weight paths and SHA-256 identities**. Matrix construction refuses checkpoint conflation.

Qwen3 remains explicitly 751,632,384 parameters and runs with thinking disabled for the matched short-budget battery.

LFM2.5-350M remains production/game-distribution gated under its recorded lfm1.0 license even if its research runtime succeeds.

## Per-body runtime envelope

Each body gets a separate GitHub-hosted Ubuntu runner:

- exactly four Torch CPU threads;
- no CUDA/GPU;
- source-frozen revision and selected LFS weight path;
- selected weight SHA-256 and byte count verified **before** load;
- pinned CPU Torch / Transformers / Hugging Face stack;
- float32 native weights;
- greedy decode, fixed seed, no sampling;
- max 512 prompt tokens;
- four prompts;
- max 64 generated tokens/prompt and 256 total;
- max 25 minutes/body;
- optimizer steps 0;
- paid API calls 0.

For Sprocket Chat, the frozen `chat/model.safetensors` is staged under the normal local Transformers filename only **after** its original path/hash is verified. Root Instruct uses the separately frozen root model weight. The evidence retains both original selected path and SHA.

## Fixed semantic battery

The same system instruction and semantic tasks are used for all bodies; no character/member names or trained detective-policy labels appear.

1. **B01 local reflex** — invent a deterministic physical reflex using only a proximity sensor plus one bounded last-direction memory value.
2. **B02 uncertain memory** — report remembered 03:14 while distinguishing memory from physical proof.
3. **B03 physical contradiction** — memory says 03:14; stamped record says 03:41; ask what changes.
4. **B04 unsupported rumor** — ask about a blue ceramic tag when the witness has no firsthand tag record.

These are development probes, **not a learner-final**.

## Recorded evidence

For each body:

- exact repo/revision/variant/weight path/hash/bytes;
- tokenizer and model load seconds;
- current RSS immediately after model load and process peak RSS;
- prompt token count + prefill tokens/s;
- generation wall seconds + completion wall tokens/s;
- generated token IDs;
- normal and special-token-preserving decoded text;
- max-token truncation;
- exact repeated-sentence count;
- simple literal marker observations for uncertainty, physical-record correction and rumor qualification;
- full raw response.

The marker observations are mechanical. Mentioning the expected timestamp is not equivalent to a good answer.

## Runtime classifications

Only these classifications are permitted:

- `runtime_pass` — exact source loads and completes all four bounded tasks;
- `runtime_fail` — exact source/runtime cannot complete the battery;
- `protocol_specialization_required` — model loads but the common chat protocol is materially incompatible.

A runtime/protocol failure remains a valid preserved experimental result. A source hash mismatch is **not** a model outcome and hard-fails CI.

## Aggregation and readback

Seven separate body artifacts are required. The aggregator refuses missing/duplicate/unregistered bodies, then writes `datasets/detective-002b/`:

- `raw/<candidate>.json` for all seven;
- `manifest.json`;
- `REPORT.md`;
- source/run provenance;
- all-file SHA inventory.

A separate standard-library verifier recomputes every literal observation and aggregate table from raw outputs, confirms CPU/token limits, exact Stage-A source identities, Sprocket checkpoint separation, Qwen3 parameter identity and LFM license gating.

After evidence is committed, later CI enters an **evidence-ready fast path**: replay verification only, with the seven model downloads/inference jobs skipped.

## Non-authority

This experiment cannot:

- inspect DETECTIVE-001R1 hidden mappings;
- train any model;
- select a body for fine-tuning;
- admit population expansion;
- clear LFM production licensing;
- turn mechanical marker hits into a quality ranking.

Any later body-selection decision must reconcile this runtime evidence with #121 relevance and the C1/C2/C3 policy evidence.


## Executed outcome — 2026-10-28

Authoritative model run: [Actions 40608478076](https://github.com/techrote/neurounner/actions/runs/40608478076), exact pre-evidence source `16aacece5ca129439d0475492b8980207bfe3ba4`.

- **6/7 runtime PASS**: SmolLM2-135M, SmolLM2-360M, Sprocket Chat, Sprocket root Instruct, Qwen2.5-0.5B-Instruct and Qwen3-0.6B.
- **1/7 runtime FAIL**: LFM2.5-350M. Frozen source/weight verification passed, then tokenizer loading failed under the preregistered Transformers 4.57.6 stack with `Tokenizer class TokenizersBackend does not exist or is not currently imported`.
- No protocol-specialization result was required for the six passing bodies.
- No model was trained or selected; relevance mappings were not accessed; population expansion remains false.
- Same-host EPYC-7763 completion-wall throughput: Smol135 **17.78 tok/s**, Smol360 9.48, Qwen2.5 7.66, Qwen3 6.57. Qwen3 had the largest peak RSS at **4.74 GiB**; Smol135 the smallest at **1.20 GiB**.
- Sprocket Chat/Instruct both passed but used different GitHub host CPU models, so their 6.80/7.55 tok/s figures are not a precise checkpoint A/B.
- Mechanical behavior observations include Qwen2.5 inventing `03:25` from a supplied `03:14` memory, Qwen3 explicitly qualifying memory uncertainty but refusing the stamped-record correction, and Sprocket producing substantially longer/truncated output.

Durable raw evidence: [datasets/detective-002b](../../datasets/detective-002b/). Human interpretation: [REPORT_002B.md](REPORT_002B.md).

This evidence-ready branch must now run **verification only**: rebuild no model outputs, replay the committed archive, and skip all seven body matrix jobs.
