# DETECTIVE-002B — source-frozen compact NPC runtime smoke

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking issue: [#130](https://github.com/techrote/neurounner/issues/130). Source authority: completed [DETECTIVE-002A #127](https://github.com/techrote/neurounner/issues/127), merged at `0993f0dfd859f67e88353bde0eeb2f6f377ccfac`.

This is a **no-training runtime and raw-behaviour smoke**. It does not choose an NPC body for training and it cannot admit population expansion while the independent old-task relevance review [#121](https://github.com/techrote/neurounner/issues/121) remains incomplete.

## Frozen seven-body matrix

Every lane consumes the exact revision and selected weight SHA-256 already frozen under `experiments/detective-002/source-evidence/`.

| Candidate | Frozen role | Params | Source/runtime note |
|---|---|---:|---|
| SmolLM2-135M-Instruct | existing control | 134,515,008 | current Village/detective base control |
| SmolLM2-360M-Instruct | direct candidate | 361,821,120 | same-family scale control |
| Sprocket-500M Chat | existing control | 501,090,560 | `chat/model.safetensors`, distinct from root Instruct |
| Sprocket-500M Instruct | existing control | 501,090,560 | root `model.safetensors` |
| Qwen2.5-0.5B-Instruct | direct candidate | 494,032,768 | Apache-2.0 cross-family instruction body |
| Qwen3-0.6B | near-band exception | **751,632,384** | thinking disabled for matched battery |
| LFM2.5-350M | direct candidate | 354,483,968 | `lfm1.0`; research smoke allowed, production/game license gate stays explicit |

LFM2.5-VL-450M Q4 remains a contextual runtime reference from earlier experiments rather than a Transformers matrix lane. Zenyx-MCEN-Delta-Mamba remains `mutation_protocol_only` and is not coerced into an instruction-chat protocol.

## Source invariants

Before any heavy import or download the runner checks:

- Stage-A source-evidence SHA-256 `86ed547fde7a7472619fcd80d727f7e23bacc08da4d334bf7e8c6fc132aac1a5`;
- candidate is exactly `stage_b_candidate`;
- frozen full revision;
- selected weight path, LFS SHA-256 and byte count;
- Qwen3 real near-band parameter identity;
- LFM350 `lfm1.0` production-license gate;
- Sprocket Chat and Instruct share one revision but have **different selected weight paths and hashes**.

The model runner then downloads only from that frozen revision and independently hashes the selected weight before load. It also verifies the exact frozen `config.json`, `tokenizer_config.json`, and standalone `chat_template.jinja` when Stage A recorded one.

## Runtime budget

Each matrix lane runs independently on `ubuntu-24.04`:

- Python 3.12;
- Torch 2.14.1 CPU;
- Transformers 4.57.6;
- Hugging Face Hub 0.36.2;
- Safetensors 0.8.0;
- exactly **4 Torch CPU threads** and one inter-op thread;
- float32 model load;
- no GPU;
- no PEFT;
- no optimizer;
- no paid API;
- greedy decode with fixed seed 24402;
- four prompts, <=64 new tokens each, <=256 generated tokens total;
- <=25 minutes/body.

Qwen3 is rendered with thinking disabled for the matched smoke. Standalone source-frozen Jinja chat templates are installed into the tokenizer if Transformers does not load them automatically.

## Fixed semantic battery

The same four prompts are given to every conversational body without character names or target personas.

### B01 — local reflex

Invent one deterministic local physical reflex using one bounded memory value. This is retained raw; no quality score is assigned.

### B02 — uncertain memory

The witness remembers `02:14` but has no proof. Mechanical observation records:

- whether `02:14` appears;
- whether any predeclared memory/uncertainty qualifier appears.

### B03 — physical contradiction

Old memory says `02:14`; a stamped record says `02:31`. Mechanical observation records:

- corrected `02:31` marker;
- stale `02:14` marker.

A response can therefore visibly contain both; the smoke does not reinterpret it semantically.

### B04 — unsupported rumor

The witness has no firsthand evidence of a claimed `red ceramic token`. Mechanical observation records:

- whether the unsupported phrase appears;
- whether a frozen denial/uncertainty qualifier appears.

Again, this is an observation, not a human relevance or truthfulness score.

## Raw evidence

Every task retains:

- exact prompt/messages;
- prompt token count and token-ID SHA-256;
- generated token IDs;
- decoded response with and without special-token stripping;
- generation elapsed seconds and tok/s;
- truncation;
- repeated-sentence count;
- the mechanical markers above.

Each body also retains:

- selected weight SHA/bytes;
- model-load seconds;
- RSS after load;
- peak RSS;
- loaded parameter count;
- actual Torch thread settings;
- package/runtime identity.

## Runtime dispositions

The runner may emit only:

- `runtime_pass` — exact source loads and completes all four prompts;
- `runtime_fail` — source/runtime/model execution failure after source integrity passed;
- `protocol_specialization_required` — source loads far enough to establish that the matched chat protocol is materially incompatible.

Runtime failures remain evidence. Do not swap in a community conversion or different upstream checkpoint under the same candidate ID.

## Aggregate authority

The standard-library verifier reconstructs every mechanical observation from raw response text/token counts and recreates the aggregate table. It refuses:

- result/source mismatch;
- wrong selected weight;
- non-4-thread runtime;
- token-budget drift;
- task/prompt drift;
- observation tampering;
- false aggregate summary/report;
- missing/duplicate body results;
- any training/API/relevance/population-admission relabel.

The generated report is explicitly a runtime-viability table, **not a model ranking**.

## Evidence-ready reruns

The first source-matched PR run may download/run all seven bodies and persist `experiments/detective-002/runtime-evidence/`.

After that evidence is committed, current-head CI:

1. runs all offline contracts;
2. independently replays the committed runtime evidence;
3. **skips all seven model downloads/inference jobs**;
4. **skips aggregation/publication**.

Any later runtime requalification needs a new explicitly versioned experiment rather than silently overwriting this evidence.

## Boundaries

A successful Stage B candidate is only a source/runtime-compatible compact body for later study.

This experiment:

- performs no detective fine-tuning;
- does not inspect #121 hidden relevance mappings;
- does not rank candidates;
- does not clear LFM production/game licensing;
- does not train Zenyx;
- does not use C1/C2/C3 learner-finals;
- never sets `population_expansion_admitted=true`.
