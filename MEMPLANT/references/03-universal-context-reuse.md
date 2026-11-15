---
id: "MEMPLANT-REF-03"
title: "A Universal Context-Reuse Layer for Cross-Model KV Sharing"
authors: ["Yi Li","Dongming Jiang","Yi Zhao","Bingzhe Li"]
year: 2026
venue: "arXiv preprint"
source_url: "https://arxiv.org/abs/2608.30963"
source_type: "paper"
research_tracks: ["transfer","orchestration"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# A Universal Context-Reuse Layer for Cross-Model KV Sharing

**Primary reference:** https://arxiv.org/abs/2608.30963

## Brief summary report
Investigates cross-model cache reuse across differences in architecture, tokenizer, scale and family; the abstract reports reduced prefill cost in Qwen/Gemma and Llama/Qwen examples.

## MEMPLANT relevance
Directly motivates context mobility across model substrates rather than per-model context ownership.

## Scope and limitations
The word 'universal' in the title does not establish lossless transfer or a universal standard for any possible pair.

## Recommended controlled experiment
Benchmark a deliberately cross-family handoff with shape, tokenizer and position checks.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
