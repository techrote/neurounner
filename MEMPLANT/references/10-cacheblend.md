---
id: "MEMPLANT-REF-10"
title: "CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion"
authors: ["Jiayi Yao","Hanchen Li","Yuhan Liu","Siddhant Ray","Yihua Cheng","Qizheng Zhang","Kuntai Du","Shan Lu","Junchen Jiang"]
year: 2025
venue: "EuroSys 2025"
source_url: "https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/"
source_type: "paper"
research_tracks: ["sharding","composition"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

**Primary reference:** https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/

## Brief summary report
CacheBlend selectively recomputes portions of independent cached chunks to correct dependencies when using non-prefix RAG context.

## MEMPLANT relevance
Demonstrates why sharded memory caches cannot ordinarily be concatenated naively.

## Scope and limitations
Primarily within one target model; partial recomputation is needed and does not solve cross-family mapping.

## Recommended controlled experiment
Test naive concatenation versus selective correction and full prefill.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
