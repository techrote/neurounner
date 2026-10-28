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
- **[DETECTIVE-001 unreliable witnesses / instruct scripter pilot](experiments/detective-001/README.md)** — isolated detective-game authority/visibility/evidence contract and offline case fixture; issue #103; **not** new model training or Village/NCR evidence.
- **[DETECTIVE-001B source-matched model smoke report](experiments/detective-001/REPORT_001B.md)** — 27 October 2026 bounded native SmolLM2 base+two SFT interviews (12 turns) and fictional scripts (2 turns): 885 generated tokens, 0/2 valid script outputs, full raw evidence under [datasets/detective-001-smoke](datasets/detective-001-smoke/), source-verified but **no behavioral admission**.
- **[DETECTIVE-001B2 native presentation A/B](experiments/detective-001/REPORT_001B2.md)** — matched first-turn JSON vs natural witness formatting: 4/6 JSON object-prefix echoes vs 0/6 natural, 730 generated tokens, source-bound raw evidence in [datasets/detective-001-presentation-ab](datasets/detective-001-presentation-ab/). Descriptive, not final behavioral admission.
- **[DETECTIVE-001B3 source-correct Sprocket Instruct scripter smoke](experiments/detective-001/REPORT_001B3.md)** — publisher-native root Sprocket tool/result wire on five fictional evidence tasks: 0/2 correct novel tool selection, 1/1 restraint, 2/2 narrow result/denial use; raw source-bound evidence under [datasets/detective-001-sprocket-scripter](datasets/detective-001-sprocket-scripter/). Tool bridge source PASS; general scripting **NOT ADMITTED**.
- **[DETECTIVE-001B4 pinned LFM scripting control](experiments/detective-001/REPORT_001B4.md)** — conventional task-level control on five fictional evidence tasks: 3/3 strict script schema valid but 0/2 actionable selections, 1/1 restraint, 1/2 strict result-use; raw evidence under [datasets/detective-001-lfm-scripter](datasets/detective-001-lfm-scripter/). Not wire-matched to Sprocket; general scripting **NOT ADMITTED**.
- **[DETECTIVE-001C1 two-character witness-policy continuation](experiments/detective-001/REPORT_001C1.md)** — first detective-specific CPU LoRA continuation from immutable Village SFT parents: 64 steps/member; locked learner-final relation policy V-G02-A 2/12→6/12 and V-G09-C 1/12→4/12 with no protected-leak increase; independent 0–4 relevance remains unscored, so larger-cast expansion is **NOT ADMITTED**. Source/evidence run [40589762919](https://github.com/techrote/neurounner/actions/runs/40589762919), release assets and actual tag ref independently rechecked.
- **[DETECTIVE-001C2-A frozen proposition/relevance evaluator](experiments/detective-001/C2_EVALUATOR.md)** — source-bound C1 proposition scorer v2 plus 16-item opaque old-task relevance packet; withholding is proposition-based rather than phrase-list based, repository ratings remain null, and **no C2 optimizer work is authorized until this evaluator is merged/frozen**. Issue #115.
- **[DETECTIVE-001C2 pairwise witness-policy repair](experiments/detective-001/REPORT_001C2.md)** — two 40-step C1→C2 preference continuations with post-hash 18-case learner-final: V-G02-A 12/18→13/18 but fails lie gate and loses corrupted-belief retention; V-G09-C 8/18→14/18 and passes proposition gate. Independent relevance remains unrated, so wider-cast expansion is **NOT ADMITTED**. Authoritative run [40598884556](https://github.com/techrote/neurounner/actions/runs/40598884556), raw evidence under `training-c2-evidence/`.
- **[DETECTIVE-001C3 G02 interference repair](experiments/detective-001/REPORT_001C3.md)** — one 28-step G02-only C2→C3 repair recovered sincere corrupted-belief behavior **1/3→2/3** while retaining lie/withholding/correction **3/3**, but speculation fell **3/3→2/3** and matched-final total stayed **15/18→15/18**. Locked C3 GO = **FAIL** (no total improvement); independent relevance remains unrated and population expansion **NOT ADMITTED**. Authoritative run [40602963001](https://github.com/techrote/neurounner/actions/runs/40602963001).
- **[DETECTIVE-001R1 independent relevance review](experiments/detective-001/RELEVANCE_REVIEW_001R1.md)** — review-only custody for the opaque C2/C3 old-task packets; reconstructs hidden mappings only after complete external 0–4 ratings, computes cumulative G02 C1→C3 and G09 C1→C2 regression gates, and contains **no model/self-scoring path**. Population expansion remains blocked until ratings exist.
- **[DETECTIVE-002A compact NPC body source qualification](experiments/detective-002/SOURCE_QUALIFICATION.md)** — source-only nine-body cohort spanning SmolLM2, Sprocket, Qwen, LFM and a mutation-only Zenyx architecture control; first owner run resolves exact Hugging Face revisions/LFS identities without weight downloads, then later CI verifies the frozen archive offline. **No training or population admission** while DETECTIVE-001R1 #121 remains independently unrated.
- **[DETECTIVE-002B seven-body runtime smoke](experiments/detective-002/REPORT_002B.md)** — exact-source four-CPU native float32 characterization: **6/7 runtime PASS**, LFM2.5-350M tokenizer-runtime FAIL under frozen Transformers 4.57.6; raw outputs/metrics under [datasets/detective-002b](datasets/detective-002b/). No training/body selection/population admission; DETECTIVE-001R1 #121 remains reviewer-blocked. Issue #130.
- Training datasets must follow the dataset filing-system contract before they are treated as durable evidence.

## Retrieval rules

1. Prefer primary dataset records over prose summaries when answering questions about exact prompts or outputs.
2. Prefer the model journal over isolated benchmark numbers when comparing models; CPU and task shape matter.
3. Use orchestrator-owned IDs/lineage from durable manifests, not model-invented provenance.
4. Treat GitHub Actions artifacts as transient staging unless their contents have been promoted under `datasets/`.
5. When an experiment fails, inspect preserved partial artifacts before rerunning; successful partial work may already be authoritative.
