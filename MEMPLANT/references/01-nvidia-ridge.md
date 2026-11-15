---
id: "MEMPLANT-REF-01"
title: "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse"
authors: ["Taekyung Heo","Rasoul Shafipour","Ritchie Zhao","Maximilian Golub","Mohammad Mahdi Kamani","Ritika Borkar","Makesh Tarun Chandran","Pantea Zardoshti","Bita Darvish Rouhani"]
year: 2026
venue: "arXiv preprint"
source_url: "https://arxiv.org/abs/2608.03893"
source_type: "paper"
research_tracks: ["transfer","baseline"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse

**Primary reference:** https://arxiv.org/abs/2608.03893

## Brief summary report
The authors fit closed-form ridge mappings from selected source KV layers to target KV layers, with positional-rotation handling. Within-family tests show substantial savings over rebuilding the receiver cache, but some pairs degrade sharply.

## MEMPLANT relevance
Minimal A→B baseline for MEMPLANT: learn per-layer K/V projection while both model weights remain frozen.

## Scope and limitations
Matched KV geometry and mostly same-family models; no universal interchange format, and measured mapper time is not whole-pipeline latency.

## Recommended controlled experiment
Replicate ridge fitting against target-native prefill; reject low-quality transferred states.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
