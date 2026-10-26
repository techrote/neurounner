# Durable datasets

Generated corpora that matter beyond a single Actions run are archived here according to [`docs/DATASET_FILING_SYSTEM.md`](../docs/DATASET_FILING_SYSTEM.md).

| Dataset | Description | Status |
|---|---|---|
| [`babble-001`](babble-001/README.md) | 12 Qwen-generated temperament guides → 48 Sprocket babbles → 12 MiniMax-compiled personality/training packets, including raw conversation/provider history | archived |
| [`babble-002`](babble-002/README.md) | 1,200 lineage-preserving personality-training examples derived from all 12 BABBLE-001 packets, with paired generic negatives and raw NVIDIA history | archived |
| [`train-003`](train-003/README.md) | 896-example semantic mechanical compiler corpus with balanced VALID / REPAIR / INTERESTING_BROKEN / REJECT supervision and family-disjoint held-out evaluation | qualified |
| [`story-001`](story-001/README.md) | Original local-model STORY-001 attempt; 2,706-word stitched/variant-like narrative retained for provenance | superseded by story-001-v2 |
| [`story-001-v2`](story-001-v2/README.md) | Corrective NVIDIA Bolt/Super production: one continuous 3,057-word Sprocket autofactory-memory story, independent review, raw provider evidence and post-qualification semantic deduplication | qualified / authoritative |

Actions artifacts are staging only. Add new durable corpora to this index and to the retrieval map in [`RAG.md`](../RAG.md).
