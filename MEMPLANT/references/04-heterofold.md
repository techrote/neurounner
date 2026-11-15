---
id: "MEMPLANT-REF-04"
title: "Prefill-Free Cross-Family KV Cache Transfer for Heterogeneous Multi-Agent LLMs"
authors: ["Vincent-Daniel Yun","Woosang Lim","Haneul Yoo","Sungjoo Yoo","Murali Annavaram","Sai Praneeth Karimireddy"]
year: 2026
venue: "arXiv preprint"
source_url: "https://arxiv.org/abs/2609.32259"
source_type: "paper"
research_tracks: ["transfer","cross-family"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# Prefill-Free Cross-Family KV Cache Transfer for Heterogeneous Multi-Agent LLMs

**Primary reference:** https://arxiv.org/abs/2609.32259

## Brief summary report
HeteroFold maps sender caches into a frozen receiver's representation while reconciling tokenization, model depth and KV shape; it evaluates multiple cross-family directions and long contexts.

## MEMPLANT relevance
Relevant implementation pattern where simple matched-head linear maps are inadequate.

## Scope and limitations
A trained model-pair handoff, not a proof of lossless cache transfer or arbitrary cache concatenation.

## Recommended controlled experiment
Compare architecture alignment with naive projection on identical held-out corpora.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
