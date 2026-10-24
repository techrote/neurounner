# neurounner retrieval map

This file is the first retrieval/signposting document for agents, RAG systems and future development work. Read the relevant authority below before modifying experiments or interpreting evidence.

## Data and archival authority

- **[Dataset filing system](docs/DATASET_FILING_SYSTEM.md)** — mandatory rules for preserving prompts, raw generations, provider responses, lineage, repairs, hashes and Actions-artifact promotion.
- **[Dataset index](datasets/README.md)** — durable corpora currently committed to the repository.
- **[BABBLE-001 archive](datasets/babble-001/README.md)** — first fully filed babble/personality dataset: guides, 48 Sprocket histories, NVIDIA compiler history and 12 derived packets.
- **[BABBLE-002 archive](datasets/babble-002/README.md)** — 1,200 expanded personality-training examples across 12 personas, paired generic negatives, trait lineage and raw NVIDIA expansion history.

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
- **[TRAIN-002b void.1 eight-lane study](docs/TRAIN_002B_VOID_EIGHT_LANE.md)** — byte-tokenizer-aware evaluation, two seed controls, six controlled training perturbations, and durable raw held-out evidence.
- Training datasets must follow the dataset filing-system contract before they are treated as durable evidence.

## Retrieval rules

1. Prefer primary dataset records over prose summaries when answering questions about exact prompts or outputs.
2. Prefer the model journal over isolated benchmark numbers when comparing models; CPU and task shape matter.
3. Use orchestrator-owned IDs/lineage from durable manifests, not model-invented provenance.
4. Treat GitHub Actions artifacts as transient staging unless their contents have been promoted under `datasets/`.
5. When an experiment fails, inspect preserved partial artifacts before rerunning; successful partial work may already be authoritative.
