# Pseudo-embeddings and JIT context compilation

## What “pseudo-embedding” means
A temporary, continuous conditioning vector not required to correspond to a valid vocabulary token ID. It is **not** a newly inserted word in the tokenizer or a change to pretrained model weights.

Three separate techniques must not be conflated:
| Injection mode | Interface | What is learned or generated |
| --- | --- | --- |
| **Soft input prompt** | `m × d_model` vectors prepended at the embedding input | recipient-specific virtual embeddings |
| **Deep / prefix-tuned KV** | model-native key/value vectors at some or all attention layers | virtual attention positions, independent of legal token sequences |
| **Cache graft** | gated transformation of an existing recipient state | donor-conditioned native-state residuals |

Classical soft prompts and prefix tuning normally learn *static task parameters*. MEMPLANT's **JIT variant** would build a new, context- and optionally query-conditioned pseudo-context, perhaps with a hypernetwork.

## Proposed pipeline
Let `C` be authoritative text, `z=E(C)` compact memory, `q` the current query and `B` the recipient:
```text
z = encoder(C)                          # retain source IDs and original C
v_B = compiler_B(z, q, slot_budget)     # temporary soft tokens or deep KV
answer = frozen_model_B(v_B, q)         # weights stay unchanged
```
Alternative: optimise a small virtual prefix with gradients during the current session (live tuning). That is potentially more adaptable but incurs extra inference/training cost. Benchmark it separately from one-pass generation of virtual vectors.

## Why this matters
- **Corpus compression:** replace many input positions with fewer continuous memory positions, while retaining exact retrieval fallback for rare details.
- **Role-sensitive compilation:** the same stored corpus generates different prefixes for a planner, code reviewer or critic according to their current query.
- **Low-cost agent cloning:** immutable background memory can serve multiple branch-local JIT prefixes without repeating a long prompt.
- **Intermediate-representation experiment:** one stored memory `z` could, in principle, feed different decoder-specific compilers without retraining the original model.
- **JIT adaptation:** update a *temporary context representation* rather than performing LoRA or full fine-tuning of the model weights.

A 50,000-token corpus compiled into 256 virtual positions has about **195× fewer positions**, *not* 195× guaranteed information fidelity or even 195× fewer bytes. Width, depth, training cost, decoder overhead and exact-match recall all matter.

## Training and evaluation
Train against the **target model's behavior** on known answers, next-token distributions or verifiable retrieval tasks—not only Euclidean embedding distance. Compare with native full-context prefill, equal-budget extractive retrieval/summaries, static soft prompts, query-conditioned soft prompts and deep KV prefixes. Hold out entire documents and query families.

Required measures: exact factual recovery, rare needles and contradictions, multi-turn retention, negative transfer, unintended instruction following, GPU/CPU resident memory, bytes copied, cost to compile, end-to-end time-to-first-token, context branch isolation and prompt-injection resistance.

## Failure modes
Recipient vocabulary/position mismatch; deep-layer head geometry; incorrect masks; loss of unanticipated facts; overfitting to a question; expensive per-session optimisation; donor malicious instructions or secret leakage. Use source ownership, versioned compiler contracts, bounded prefixes and native/retrieval fallback.

## Direct background
[ICAE](references/08-icae.md), [CMC](references/09-cmc.md), [Prefix-Tuning](references/14-prefix-tuning.md), [Soft Prompt Tuning](references/15-soft-prompt-tuning.md), [C²KV](references/11-c2kv.md), [XKV](references/06-xkv.md).
