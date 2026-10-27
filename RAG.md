# neurounner retrieval map

This file is the first retrieval/signposting document for agents, RAG systems and future development work. Read the relevant authority below before modifying experiments or interpreting evidence.

## Data and archival authority

- **[Dataset filing system](docs/DATASET_FILING_SYSTEM.md)** — mandatory rules for preserving prompts, raw generations, provider responses, lineage, repairs, hashes and Actions-artifact promotion.
- **[Dataset index](datasets/README.md)** — durable corpora currently committed to the repository.
- **[BABBLE-001 archive](datasets/babble-001/README.md)** — first fully filed babble/personality dataset: guides, 48 Sprocket histories, NVIDIA compiler history and 12 derived packets.
- **[BABBLE-002 archive](datasets/babble-002/README.md)** — 1,200 expanded personality-training examples across 12 personas, paired generic negatives, trait lineage and raw NVIDIA expansion history.
- **[TRAIN-003 corpus](datasets/train-003/README.md)** — 896 reviewed semantic mechanical-compiler examples with balanced four-way decisions and family-disjoint held-out cases.

Raw records outrank derived summaries. Do not discard failed/malformed generations merely because a later compiler produced cleaner output.

## Models and performance

- **[Model performance and behaviour journal](docs/MODEL_JOURNAL.md)** — measured throughput/RSS/CPU IDs plus task-specific behavioural impressions.
- **[Model longlist](docs/MODEL_LONGLIST.md)** — speculative and promoted model candidates.
- **[Active candidate registry](config/candidates.json)** — reproducibly pinned runnable candidates.
- **[Model storage policy](docs/MODEL_STORAGE.md)** — where weights and larger assets should live.

## Experiment design

- **[Conversation/orchestration topologies](docs/CONVERSATION_TOPOLOGIES.md)** — mutator/compiler/panel/incubator/ecology patterns.
- **[Initial hosted experiments](docs/INITIAL_EXPERIMENTS.md)**
- **[Second hosted experiments](docs/SECOND_EXPERIMENTS.md)**
- **[Eight-runner startup smoke](docs/STARTUP8_2026-10-24.md)**

## Training

- **[TRAIN-001 CPU LoRA proof](docs/TRAIN_001_CPU_LORA.md)** — 135M CPU LoRA, cross-run checkpoint/resume and measured throughput.
- **[TRAIN-003 semantic mechanical compiler](docs/TRAIN_003_SEMANTIC_COMPILER.md)** — 896-example four-decision corpus, family-disjoint held-out semantic scoring, direct TRAIN-001 comparison, and durable raw evidence.
- **[TRAIN-004 full-parameter CPU feasibility](docs/TRAIN_004_FULL_PARAMETER.md)** — measured 135M full-AdamW vs LoRA throughput/RSS/checkpoint ratios plus fresh-runner full-checkpoint resume proof.
- **[TRAIN-002b void.1 eight-lane study](docs/TRAIN_002B_VOID_EIGHT_LANE.md)** — byte-tokenizer-aware evaluation, two seed controls, six controlled training perturbations, and durable raw held-out evidence.
- **[PERSONA-001 four-cartridge tournament](docs/PERSONA_001_REPORT.md)** — four SmolLM2-135M personality LoRAs, blind raw vs style-neutral behavioural identification, base-control comparison, and durable release assets.
- **[PERSONA-002 preference continuation](docs/PERSONA_002_REPORT.md)** — anti-sanitization pairwise continuation from PERSONA-001, fixed-tournament before/after comparison, confusion/transfer deltas, and durable `persona-002-v1` cartridges.
- **[VILLAGE-001 cybernetic goblin foundry](docs/VILLAGE_001_CYBERGOBLIN_FOUNDRY.md)** — 36-member Sprocket-derived population contract combining member-specific curriculum, Sprocket feedback, bounded NVIDIA compilation/criticism, 72 CPU training lanes and hierarchical blind evaluation; epic #9, execution #10–#14.
- **[VILLAGE-001E final qualification](experiments/village-001/final-evidence/REPORT.md)** — authoritative location for the 36-member raw/style-neutral tournament, 36×36 confusion matrices, per-member verdicts, prompt-neutral pair/small-group ecology, final model registry and dataset provenance once #14 executes; the predeclared protocol is in `experiments/village-001/tournament-config.json`.
- Training datasets must follow the dataset filing-system contract before they are treated as durable evidence.

- **[NCR-001B pure PyTorch cellular core](docs/NCR_001_B_CORE.md)** — shared 8×8×4 depth-persistent NCA with explicit per-forward state, frozen toy-backbone gradient proof, reproducible CPU tests and [source-bound synthetic evidence](experiments/ncr-001/core-evidence/core-smoke.json). It is not LM integration, cache conformance or training admission.

- **[NCR-001D frozen Sprocket Chat real-model depth-hook prepass](docs/NCR_001_D_SPROCKET_CHAT.md)** — exact Chat-subfolder 1 GB checkpoint SHA-256, initial logits/no-op parity, nonzero NCA cached causal checks, and [archived raw source receipt](experiments/ncr-001/sprocket-chat-preflight/receipt.json) with [artifact provenance](experiments/ncr-001/sprocket-chat-preflight/provenance.json). Distinct from Sprocket Instruct; Chat role-template semantics and independent #72 still pending.

## Retrieval rules

1. Prefer primary dataset records over prose summaries when answering questions about exact prompts or outputs.
2. Prefer the model journal over isolated benchmark numbers when comparing models; CPU and task shape matter.
3. Use orchestrator-owned IDs/lineage from durable manifests, not model-invented provenance.
4. Treat GitHub Actions artifacts as transient staging unless their contents have been promoted under `datasets/`.
5. When an experiment fails, inspect preserved partial artifacts before rerunning; successful partial work may already be authoritative.

## NCR-001 experimental architecture (source-gated)

- **[NCR-001A preregistration](experiments/ncr-001/protocol.md)** — persistent-depth 8×8×4 NCA across **SmolLM2, Sprocket Chat, Sprocket Instruct and void.1/Qwen3**; source-bound model-specific hooks, byte-aware evaluation and four independent conformance gates; **no training admitted yet**.
- **[NCR-001 machine-readable protocol](experiments/ncr-001/protocol.json)** — provisional two-model source locks, SmolLM2/Sprocket variant guard, selected depth-tap policy, sealed TRAIN-003 holdout and outstanding fixture/checkpoint approvals.
- **[NCR-001 pinned-source readback](experiments/ncr-001/source-pins.json)** — durable **four-model** exact-revision HF config/tokenizer/void ByteTokenizer source SHA-256s, declared-but-not-downloaded checkpoint LFS hashes, and model-specific depth taps. Includes verified source metadata from read-only Actions runs [40401534069](https://github.com/techrote/neurounner/actions/runs/40401534069) and [40405051395](https://github.com/techrote/neurounner/actions/runs/40405051395). Never treat metadata as verified weight bytes, executable tokenizer review, or causal model parity.
- **[NCR-001A read-only preflight](scripts/ncr_001_protocol.py)** — verify local protocol & fixed corpus without network, or inspect exact pinned HF model/tokenizer/config metadata without downloading model weights. Do not interpret metadata-preflight success as causal conformance or training authorization.
