# Durable datasets

Generated corpora that matter beyond a single Actions run are archived here according to [`docs/DATASET_FILING_SYSTEM.md`](../docs/DATASET_FILING_SYSTEM.md).

| Dataset | Description | Status |
|---|---|---|
| [`babble-001`](babble-001/README.md) | 12 Qwen-generated temperament guides → 48 Sprocket babbles → 12 MiniMax-compiled personality/training packets, including raw conversation/provider history | archived |
| [`babble-002`](babble-002/README.md) | 1,200 lineage-preserving personality-training examples derived from all 12 BABBLE-001 packets, with paired generic negatives and raw NVIDIA history | archived |
| [`train-003`](train-003/README.md) | 896-example semantic mechanical compiler corpus with balanced VALID / REPAIR / INTERESTING_BROKEN / REJECT supervision and family-disjoint held-out evaluation | qualified |

| [`village-001d-recovery-v-g08-c`](village-001d-recovery-v-g08-c/README.md) | Exact failed-run V-G08-C 40/60 partial and raw provider evidence from VILLAGE-001D run 40240732988 | archived |

| [`detective-004a`](detective-004a/README.md) | Source-native deterministic 36-example synthetic Sprocket detective-scripter curriculum, Git-blob-bound manifest, exact runner provenance and zero-training evidence; no final or model admission | archived Stage A / independent review pending |

Actions artifacts are staging only. Add new durable corpora to this index and to the retrieval map in [`RAG.md`](../RAG.md).
