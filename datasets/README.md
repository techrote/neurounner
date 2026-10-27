# Durable datasets

Generated corpora that matter beyond a single Actions run are archived here according to [`docs/DATASET_FILING_SYSTEM.md`](../docs/DATASET_FILING_SYSTEM.md).

| Dataset | Description | Status |
|---|---|---|
| [`babble-001`](babble-001/README.md) | 12 Qwen-generated temperament guides → 48 Sprocket babbles → 12 MiniMax-compiled personality/training packets, including raw conversation/provider history | archived |
| [`babble-002`](babble-002/README.md) | 1,200 lineage-preserving personality-training examples derived from all 12 BABBLE-001 packets, with paired generic negatives and raw NVIDIA history | archived |
| [`train-003`](train-003/README.md) | 896-example semantic mechanical compiler corpus with balanced VALID / REPAIR / INTERESTING_BROKEN / REJECT supervision and family-disjoint held-out evaluation | qualified |
| [`ncr-001-sprocket-wire-diagnostic`](ncr-001-sprocket-wire-diagnostic/README.md) | Seven fixed synthetic Sprocket Instruct cases × three frozen/untrained modes: 21 original raw outputs, published SFT wire/mask identity, full-position KV-cache and negative observations; **not training or held-out evaluation** | archived diagnostic; final #72 BLOCKED |

Actions artifacts are staging only. Add new durable corpora to this index and to the retrieval map in [`RAG.md`](../RAG.md).
