---
id: "MEMPLANT-REF-13"
title: "Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference"
authors: ["Zhifan Luo","Shuo Shao","Su Zhang","Lijing Zhou","Yuke Hu","Chenxu Zhao","Zhihao Liu","Zhan Qin"]
year: 2026
venue: "NDSS 2026"
source_url: "https://www.ndss-symposium.org/ndss-paper/shadow-in-the-cache-unveiling-and-mitigating-privacy-risks-of-kv-cache-in-llm-inference/"
source_type: "paper"
research_tracks: ["security","provenance"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference

**Primary reference:** https://www.ndss-symposium.org/ndss-paper/shadow-in-the-cache-unveiling-and-mitigating-privacy-risks-of-kv-cache-in-llm-inference/

## Brief summary report
Examines attacks that reconstruct sensitive inputs or infer information from KV state; introduces the KV-Cloak defense.

## MEMPLANT relevance
Transferred memory is sensitive source-derived data, not an anonymized numeric blob.

## Scope and limitations
The paper's attack assumptions are specific; exact universal token recovery is not established.

## Recommended controlled experiment
Require per-tenant access control, encryption, retention rules and synthetic-secret leak testing.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
