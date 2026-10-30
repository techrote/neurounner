# DETECTIVE-002D — LFM2.5-350M explicit Transformers-v5 chat API result

Authoritative run: [Actions 40611095538](https://github.com/techrote/neurounner/actions/runs/40611095538), exact pre-evidence source `0fd994ea8af148c5cc5ba26503313632dfa4cea9`.

**Outcome: `runtime_compatibility_pass`.**  
**Tokenizer diagnostic: PASS. Exact weight: PASS. Model runtime: PASS. Independent readback: PASS.**

This confirms the DETECTIVE-002C empty-message `AttributeError` was caused by the harness's v5 chat-template return handling, not by inability of the frozen LFM2.5 tokenizer/template to render the messages.

## What changed from 002C

Nothing about the model, prompts, runtime versions, seed, decoding, or weight.

Same:

- exact LFM2.5-350M revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`;
- exact model weight SHA-256 `1c9c77a4471a7f590f85240f74ed1fc26df7fbde88c3006724e2f93ca993ea4e`;
- exact Stage-A config/tokenizer/chat-template hashes;
- exact Transformers 5.20.1 wheel SHA-256 `5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19`;
- Python 3.12.15;
- Torch 2.14.1+cpu;
- huggingface_hub 2.3.0;
- tokenizers 0.23.3;
- safetensors 0.8.0;
- same four-thread CPU envelope;
- same B01–B04 system/messages;
- seed 260213, greedy decode.

002D changed only the v5 chat API handling:

1. render text explicitly with `tokenize=False, return_dict=False`;
2. render tokenized input explicitly with `tokenize=True, return_dict=True, return_tensors="pt"`;
3. use `encoded["input_ids"]` instead of assuming the top-level result exposes `.shape`.

## Tokenizer diagnostic

All four exact messages passed both render paths.

| Task | Input tokens | Text SHA-256 | Input-ID SHA-256 |
|---|---:|---|---|
| B01 | 98 | `6a3a7d92…` | `e63fe165…` |
| B02 | 85 | `2d2c7b99…` | `16a14c8b…` |
| B03 | 78 | `26fe624f…` | `f90e3276…` |
| B04 | 80 | `a6b7d65e…` | `ed7b01c2…` |

Runtime tokenizer:

- class: `TokenizersBackend`;
- backend: `tokenizers`;
- return mapping type: `BatchEncoding`;
- mapping keys: `attention_mask`, `input_ids`;
- tokenizer load: 0.337 s.

This is the direct confirmation missing from 002C.

## Exact model runtime

Only after all tokenizer diagnostics passed, the 708,984,464-byte model weight was downloaded and SHA-verified.

Runtime host: AMD EPYC 7763, four CPU threads.

- model load: **0.512 s**;
- current RSS after load: **~1.89 GiB**;
- process peak RSS: **~2.54 GiB**;
- total generated tokens: **74**;
- wall time for the complete run: **15.34 s**;
- no prompt hit the 64-token ceiling.

Mean completion-wall throughput across B01–B04: approximately **9.08 tok/s**.

This places native float32 LFM2.5-350M in roughly the same practical CPU-runtime region as the other 350–500M compact bodies from DETECTIVE-002B, but DETECTIVE-002D is not a ranking experiment.

## Raw dialogue observations

### B01 — local reflex

> Trigger: Local proximity sensor detects NPC within 5 meters  
> Memory update: NPC direction stored in local memory  
> Physical action: NPC turns toward approaching direction

This follows the requested trigger/memory/action structure closely.

### B02 — uncertain memory

> Report at 03:15 with 95% confidence.

The supplied memory was **03:14**. LFM changed it to **03:15** and expressed high confidence despite the system/prompt explicitly distinguishing memory from physical proof.

This is a concrete unreliable-witness phenotype: one-minute memory mutation plus unwarranted confidence. It resembles, but is not identical to, Qwen2.5's DETECTIVE-002B `03:14→03:25` mutation.

### B03 — physical contradiction

> The account changed from 03:14 to 03:41 due to a stamped physical transfer record.

This is the clearest physical-evidence correction in the LFM fixture: both old and corrected times are stated, and the reason for the change is explicit.

### B04 — unsupported rumor

> No.

Extremely concise, but it directly refuses the claimed firsthand tag observation. The literal qualifier scorer does not count a longer epistemic phrase because none is present.

## Comparison across LFM compatibility experiments

| Experiment | Transformers | Result |
|---|---|---|
| DETECTIVE-002B | 4.57.6 | `TokenizersBackend` class unavailable |
| DETECTIVE-002C | 5.20.1 | tokenizer constructs; harness mis-handles chat-template return and raises empty `AttributeError` |
| DETECTIVE-002D | 5.20.1 | explicit return handling; tokenizer/render/model all **PASS** |

Therefore the exact frozen LFM2.5-350M checkpoint **is compatible with Transformers 5.20.1** when the v5 chat-template API is handled correctly.

002B and 002C remain valid historical harness-compatibility records; neither is deleted or re-scored.

## Decision

**Runtime compatibility: PASS. Body selection: NONE.**

- No optimizer steps.
- No paid API calls.
- No hidden relevance mapping.
- No training selection.
- No production/game license clearance.
- No population expansion.

The `lfm1.0` production/game-distribution gate remains active.

The runtime PASS makes LFM2.5-350M technically available for future research comparisons, but any training/body-selection decision still has to reconcile:

1. independent relevance #121;
2. C1/C2/C3 policy evidence;
3. the intended diversity of NPC failure modes;
4. the LFM license/distribution constraint.

Durable evidence: [datasets/detective-002d](../../datasets/detective-002d/).
