# Source verification and correction register
**Checked:** 2026-11-15

## Verification coverage
Each of the 15 papers was cross-checked against its publisher's abstract/bibliography or the arXiv record. The Markdown records paraphrase the reported contribution. This review **did not** independently reproduce benchmarks, inspect every PDF page, audit all released code or validate every cited quantitative comparison. A research paper is not evidence that the MEMPLANT system itself works.

## Important distinctions
- **NVIDIA ridge** focuses on model-family and geometry-compatible transfers; **KV-Lingo** learns pair-specific translators with target-behavior distillation; neither gives a truly model-neutral state format by itself.
- **XKV** here refers to *Dual-Cache Latent Space Communication between Heterogeneous Language Models*, not the unrelated xKV cross-layer SVD compression project.
- **CacheBlend** and **C²KV** address composition of cached chunks; they do not establish unrestricted cross-family conversion.
- **Soft prompt tuning** and **prefix tuning** establish conditional continuous vectors for frozen models. They do not show arbitrary conversational memory encoded in JIT-generated pseudo-tokens without accuracy loss.
- **KV cache inversion** warrants treating caches and compressed latent data as sensitive. The fact that some tokens can be reconstructed in tested threat models does not prove source-token recovery from every architecture and every cache.
- A prior conversational answer referenced a *November 2026 preprint on token recovery for cross-architecture transfer* without an identifiable title/URL. **UNVERIFIED — excluded** pending a matching primary record. Do not treat it as established.
- Earlier conversational numerical examples, where not directly confirmed from original experimental tables, are intentionally not repeated in these cards.

## Evidence updates
When adopting a paper in an implementation experiment, record the paper version, model IDs/hashes, benchmark protocol, code revision, negative results and variance across seeds. Do not silently replace raw failed measurements with cleaned conclusions. Do not commit paper PDFs or model weights as ordinary source blobs.
