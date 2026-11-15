# MEMPLANT — Digital Memory Transplant
**Status:** research concept and literature dossier, **not an implemented or qualified feature**.  
**Repository:** `techrote/neurounner/MEMPLANT/`  
**Research snapshot:** 2026-11-15

## Mission
Investigate whether an LLM's live computational context can be **transplanted** to a different neural-network substrate, or first encoded as portable, compressed memory and later compiled into the receiving model's native attention/recurrent state. The aspirational applications are heterogeneous model orchestration, live handoff, context cloning/branching/sharding, large-corpus compression and just-in-time (JIT) pseudo-embeddings.

This is a **computational-memory** analogy, not a claim of transferring subjective experience, identity, or exact thought. A successful transplant must be evaluated by the receiver's behavior, provenance and cost—not by a tensor loading without an error.

## Document map
- [CONCEPT.md](CONCEPT.md) — digital memory transplant perspective and proposed context-state substrate.
- [PSEUDO_EMBEDDINGS.md](PSEUDO_EMBEDDINGS.md) — virtual tokens, deep prefixes and JIT memory compilation.
- [EXPERIMENTS.md](EXPERIMENTS.md) — staged, falsifiable experiments and admission gates.
- [REFERENCES.md](REFERENCES.md) — the 15 papers named in the earlier research discussion, each with a dedicated bibliographic/summary record.
- [VERIFICATION_NOTES.md](VERIFICATION_NOTES.md) — correction register and limits of the literature check.
- [NEUROUNNER_CONTEXT.md](NEUROUNNER_CONTEXT.md) — fit with neurounner's existing experiments and evidence policies.

## Research lanes
| Lane | Hypothesis |
| --- | --- |
| Direct cache translation | A pair-specific adapter can convert donor A's KV history into recipient B's usable native cache faster than B prefill. |
| Portable memory | A compressed intermediate representation can support multiple recipients via model-specific compilers. |
| Memory grafts | Latent donor evidence can be selectively incorporated into an already-active recipient. |
| Cloning and sharding | Versioned context objects can be branched, reused and composed with explicit history and provenance. |
| JIT pseudo-embeddings | Selected memory can be compiled into temporary soft embeddings or deep KV prefixes rather than ordinary text. |

## Research discipline
- Preserve source documents/transcripts as lossless authority; encoded memory is a derived, fallible view.
- Pin weights, tokenizers, attention topology, state layout, adapters, runtime and dataset hashes.
- Keep exact raw test prompts, runs, failures and outputs according to neurounner's [dataset policy](../docs/DATASET_FILING_SYSTEM.md).
- Do not store multi-GB weights in Git; use the [storage policy](../docs/MODEL_STORAGE.md).
- Do not infer successful transfer from matching shapes, successful token output or a few handpicked examples.
- Unqualified transfers must fall back to authorized text retrieval or target-native prefill.

**This dossier introduces no new runtime dependency, CI job, model artifact or autonomous schedule.**
