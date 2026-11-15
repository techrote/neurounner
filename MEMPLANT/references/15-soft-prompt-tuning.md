---
id: "MEMPLANT-REF-15"
title: "The Power of Scale for Parameter-Efficient Prompt Tuning"
authors: ["Brian Lester","Rami Al-Rfou","Noah Constant"]
year: 2021
venue: "EMNLP 2021"
source_url: "https://aclanthology.org/2021.emnlp-main.243/"
source_type: "paper"
research_tracks: ["pseudo-embeddings","baseline"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# The Power of Scale for Parameter-Efficient Prompt Tuning

**Primary reference:** https://aclanthology.org/2021.emnlp-main.243/

## Brief summary report
Learns virtual continuous input embeddings to condition a frozen pretrained model.

## MEMPLANT relevance
Cheapest entry point for MEMPLANT's JIT pseudo-embedding experiments.

## Scope and limitations
Input-layer soft prompts do not equal full multilayer KV transplant or a portable model-independent context representation.

## Recommended controlled experiment
Compare soft prompts with equal-budget text tokens and deep-prefix conditioning.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
