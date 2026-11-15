---
id: "MEMPLANT-REF-02"
title: "KV-Lingo: Learning KV-Cache Translators with Distillation"
authors: ["Valérie Castin","Keitaro Sakamoto","Anastasiia Filippova","João Monteiro","Marco Cuturi","Pierre Ablin"]
year: 2026
venue: "arXiv preprint"
source_url: "https://arxiv.org/abs/2609.32610"
source_type: "paper"
research_tracks: ["transfer","distillation"]
metadata_checked: "2026-11-15"
evidence_status: "publisher/arxiv abstract checked; not independently replicated"
---
# KV-Lingo: Learning KV-Cache Translators with Distillation

**Primary reference:** https://arxiv.org/abs/2609.32610

## Brief summary report
KV-Lingo learns per-target-layer linear K/V maps and distils their outputs against the receiver's native next-token distribution. Reported experiments include heterogeneous handoffs and repeated switching.

## MEMPLANT relevance
Provides a receiver-behaviour training objective beyond just numerical tensor matching.

## Scope and limitations
A pairwise directional translator is not a model-neutral latent memory; fidelity differs by model pair and task.

## Recommended controlled experiment
Compare reconstruction-only and KL-distilled maps on held-out contexts and repeated A→B→A handoffs.

**Provenance note:** This is a paraphrased research note based on public primary-source metadata and abstracts, not a claim of reproducing the results.
