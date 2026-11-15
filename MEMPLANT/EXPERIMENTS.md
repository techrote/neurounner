# MEMPLANT — staged falsifiable investigations

## M0: documentation only (this PR)
Capture the concept, exact source references, limitations, reproducibility contract and evaluation criteria. No implementation, benchmark or model-quality claim. Host remains neurounner.

## M1: direct cache-transplant proof
Pick two independently trained, small open-weight **decoder-only** models fitting available compute. Pin their exact revisions, licenses, tokenizers, KV geometries and runtime. Construct paired teacher-forced native KV observations on the same source material, splitting documents into fit and truly held-out sets.

**Controls:** native recipient full prefill; authorized text replay / summary / retrieval; source cache loaded without mapping where geometrically possible (negative); linear mapped KV; mapped KV with recipient distribution distillation if affordable.

**Evaluate:** actual recipient cache consumption, tail perplexity and next-token KL, exact-fact retrieval, multi-turn drift, memory footprint and **total** prefill/transfer/compile/first-token time. Report the failures and rejection gates. A fast but semantically degraded map is a negative result.

## M2: compressed pseudo-context
Hold source files and target weights fixed. Compare memory slots, static soft prompts, query-conditioned generated soft embeddings and deep prefixes (only if the runtime can inject K/V correctly).

**Gate:** a preregistered held-out task beats a strong equal-budget textual baseline **after compiler overhead**; rare-fact retention and negative-task degradation are disclosed. Otherwise conclude no-go for the particular method.

## M3: clone, graft and shard
Fork two independent sessions from one immutable snapshot and ask divergent questions; test donor-only facts grafted into a receiver that did not observe the source; compare separately cached chunks merged naively vs composed with correction or trained fusion.

**Gate:** no branch contamination, source-anchored evidence tracing, deterministic restore, correct inter-document recall and authorized owner/tenant isolation.

## M4: neutral latent substrate
Train an encoder and **two distinct recipient-specific decoders** for one shared versioned latent `z`. Compare with directed A→B/B→A adapters, separate native caches and text retrieval.

**Gate:** meaningful fidelity for both recipients and an aggregate storage/latency/training advantage; do not declare portability solely because both decoders can generate text.

## Artifact and authority rules
- Every run records model and tokenizer hashes, RoPE/attention geometry, runtime, compiler/mapper revision, adapter training data and calibration splits.
- Preserve all source texts, query sets, raw outputs, rejected cache injections, repair attempts, timings and evaluation code under neurounner's dataset-filing rules.
- Derived memory objects are sensitive and are never substituted for authoritative originals.
- Unexpected low-confidence transplanted states fail closed to recipient native prefill or source-grounded retrieval.
- A local result is not established by a paper's abstract or by a CI-only dry run.
