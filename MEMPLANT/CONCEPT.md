# The digital memory transplant perspective

## The transplant metaphor, precisely scoped
A running autoregressive language model has *inference state* derived from its prior input. In a conventional Transformer this includes layer-wise attention keys/values and position metadata; in hybrid/recurrent models it may also include recurrent and convolution state. This is not a model-neutral representation: the state is tied to weights, architecture, tokenizer and the exact history of cache operations.

**Donor (A)** has already processed a context. **Recipient (B)** is another model with different ancestry, geometry or specialization. A **graft** is state or latent evidence encoded from A and made consumable by B. A **transplant** is the controlled attempt to install and validate that graft. The source transcript remains the authoritative record.

Goal: enable B to use what A has encountered without necessarily reading A's entire context as ordinary tokens.

## Proposed two-path architecture
```text
authoritative source files, conversations and provenance
                          |
                 donor A's live state
                          |
          +---------------+------------------+
          |                                  |
   fast pairwise path                 portable-memory path
   A → B learned map                 E_A(source/state) → z
          |                                  |
      B-native KV                     persistent chunked z
          |                              clone / branch
          |                              query / select
          |                        D_B(z, task) compiler
          |                                  |
          +--------------→ recipient-native KV / prefix
                                             |
                                      B continuation
                                             |
                                   gate against native B
                                       with fallback
```
The direct path could minimise handoff latency but needs directional pair-specific mappings. The shared representation could lower a fleet's combinatorial interface count (from many A→B edges toward per-model encode/decode adapters) **if** one useful `z` can exist. That last condition is unproven.

## Six candidate operations
| Operation | Intent | Central challenge |
| --- | --- | --- |
| **Transplant** | Convert A's prior-state cache to B's receiving format | distribution and behavior drift |
| **Graft** | Add facts A saw but B did not | safe state fusion without interference |
| **Clone** | Fork one context into independent child sessions | copy-on-write and branch isolation |
| **Shard** | Split large corpora into independently cached latent blocks | cross-shard interactions and positioning |
| **Compile** | Produce B-specific pseudo-tokens/KV from selected latent memory | receiver alignment and total latency |
| **Restore** | Rehydrate a frozen state after eviction or migration | state identity, lifecycle and safety |

## Non-negotiable technical conditions
1. **A byte-shape match is not a semantic match.** Cache-head geometry, RoPE parameters, layer indexing and trained coordinates matter.
2. **Composition is not concatenation.** Separately prefilled chunks depend on different causal histories. Non-prefix reuse needs learned composition or selective recomputation (CacheBlend, C²KV).
3. **Context includes history.** Even equal displayed token prefixes can have different edited cache states (AKV).
4. **Compression is usually lossy.** A latent representation cannot guarantee exact arbitrary fact retrieval if it discards information. Maintain the original corpus and selective exact-text retrieval.
5. **Not all models have only KV.** Hybrid models may require persistent recurrent/state-space components for a meaningful transplant.
6. **Security follows the source.** An opaque cache can leak inputs. Treat grafting as a controlled movement of potentially sensitive content.
7. **Measure receiver utility.** Test next-token likelihood, factual recall, policy/instruction fidelity, continuation drift, end-to-end latency and memory—not only an adapter's numerical fit.

## Research ambition
Treat *context as an owned, versioned computational resource* independent of the model worker. A model is then a substrate that can be attached to a branch or snapshot of memory. This remains an architectural hypothesis until experiments show a worthwhile quality/cost advantage over text transfer, retrieval and native prefill.

The term **digital memory transplant** is intentionally vivid but carries no implication of awareness transfer, biological equivalence or accurate restoration of a source model's internal reasoning.
