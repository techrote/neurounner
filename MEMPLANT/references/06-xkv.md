---
id: "MEMPLANT-REF-06"
title: "Dual-Cache Latent Space Communication between Heterogeneous Language Models"
authors: ["Jiyao Liu","Qi Zhang","Yaoyi Jia","Ziwen Kan","Song Wang"]
year: 2026
venue: "arXiv preprint"
source_url: "https://arxiv.org/abs/2608.20617"
source_type: "paper"
research_tracks: ["graft","latent-memory"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# Dual-Cache Latent Space Communication between Heterogeneous Language Models

**Primary reference:** https://arxiv.org/abs/2608.20617

## Brief summary report
XKV pools both agents' caches, reconciles depth/head geometry, forms joint latent memory and decodes receiver-specific gated updates. It addresses donor knowledge the receiver did not see.

## MEMPLANT relevance
Most relevant to transplanting private-to-donor observations into a recipient that has its own state.

## Scope and limitations
XKV is this dual-cache paper, not the independent xKV low-rank compression project. Reported gains are not general transfer guarantees.

## Recommended controlled experiment
Test transfer of donor-only facts with explicit source provenance and query-specific receiver checks.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
