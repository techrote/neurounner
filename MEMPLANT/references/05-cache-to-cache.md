---
id: "MEMPLANT-REF-05"
title: "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"
authors: ["Tianyu Fu","Zihan Min","Hanling Zhang","Jichao Yan","Guohao Dai","Wanli Ouyang","Yu Wang"]
year: 2026
venue: "ICLR 2026"
source_url: "https://proceedings.iclr.cc/paper_files/paper/2026/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html"
source_type: "paper"
research_tracks: ["graft","multi-agent"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# Cache-to-Cache: Direct Semantic Communication Between Large Language Models

**Primary reference:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/474ada926b331d78f06d95e8913111cc-Abstract-Conference.html

## Brief summary report
C2C projects and fuses donor KV state with existing recipient KV state using gated target-layer updates, avoiding explicit intermediate text.

## MEMPLANT relevance
Establishes a plausible 'memory graft' operation distinct from replacing a complete conversation prefix.

## Scope and limitations
Its assumptions, models and quality gains are benchmark-specific; it does not define a universal graft API.

## Recommended controlled experiment
Compare donor-only, receiver-only, text exchange and gated latent fusion.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
