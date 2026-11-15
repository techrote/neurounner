---
id: "MEMPLANT-REF-08"
title: "In-context Autoencoder for Context Compression in a Large Language Model"
authors: ["Tao Ge","Hu Jing","Lei Wang","Xun Wang","Si-Qing Chen","Furu Wei"]
year: 2024
venue: "ICLR 2024"
source_url: "https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b276510ec2d3f6613a8b60c41ff0438-Abstract-Conference.html"
source_type: "paper"
research_tracks: ["compression","pseudo-embeddings"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# In-context Autoencoder for Context Compression in a Large Language Model

**Primary reference:** https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b276510ec2d3f6613a8b60c41ff0438-Abstract-Conference.html

## Brief summary report
ICAE trains compact memory slots to replace a longer context, reporting about 4× compression using a lightweight extra component.

## MEMPLANT relevance
Foundational model of latent slots acting as compressed pseudo-context.

## Scope and limitations
Memory slots are trained for a receiving model; arbitrary rare facts and other-model compatibility are not guaranteed.

## Recommended controlled experiment
Compare slots, text summaries and full context under matched budgets.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
