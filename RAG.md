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

- **[NCR-001E source-bound four-model prepass receipt audit](docs/NCR_001_E_PREPASS_AUDIT.md)** — separate standard-library readback of original SmolLM2, Sprocket Chat/Instruct and void.1 full-weight/hash/logit/cache receipts, [frozen per-model originals](experiments/ncr-001/evidence-review/) and negative tamper/leakage sentinel tests. **No fresh model execution or independent #72 conformance claim; training still prohibited.**

- **[NCR-001E fresh pinned Sprocket Chat/Instruct ordinary-role/cache reviewers](docs/NCR_001_E_SPROCKET_FRESH.md)** — verify source-card user/end/assistant role markers and real tokenizer IDs before separately pinned full frozen Chat/root-Instruct long/cache/A→B→A and cold-adapter source tests. Tool/memory semantics remain unqualified; no trained adapters.
- **[NCR-001E fresh Sprocket 512/1024/2048 frozen-host context source probe](docs/NCR_001_E_SPROCKET_EXTENDED_CONTEXT.md)** — two independently pinned original 500M variants; full vs eight true cached token continuations and raw dedicated tool/memory ID *transport only*. No tool/memory grammar, optimizer, held-out fixture or final #72 admission. Raw CI JSON must be archived after execution; never interpret a green workflow as a settled conformance gate.
- **[NCR-001E real Sprocket separate-Python-process source/cache replay](docs/NCR_001_E_SPROCKET_CROSS_PROCESS.md)** — exact pinned 500M Chat and root Instruct 256-unit/240+16 cached replay, same-runner fresh interpreter/real frozen model and adapter restore; [complete original eight JSON source receipts and provenance](experiments/ncr-001/evidence-review/sprocket-cross-process/), byte-exact nine-sentinel [no-model readback](scripts/ncr_001e_sprocket_process_receipt_audit.py). **Tool/memory grammar, full #72 admission and all training remain blocked.**

## Retrieval rules

1. Prefer primary dataset records over prose summaries when answering questions about exact prompts or outputs.
2. Prefer the model journal over isolated benchmark numbers when comparing models; CPU and task shape matter.
3. Use orchestrator-owned IDs/lineage from durable manifests, not model-invented provenance.
4. Treat GitHub Actions artifacts as transient staging unless their contents have been promoted under `datasets/`.
5. When an experiment fails, inspect preserved partial artifacts before rerunning; successful partial work may already be authoritative.

## NCR-001 experimental architecture (source-gated)

- **[NCR-001A preregistration](experiments/ncr-001/protocol.md)** — persistent-depth 8×8×4 NCA across **SmolLM2, Sprocket Chat, Sprocket Instruct and void.1/Qwen3**; source-bound model-specific hooks, byte-aware evaluation and four independent conformance gates; **no training admitted yet**.
- **[NCR-001 machine-readable protocol](experiments/ncr-001/protocol.json)** — provisional four-model source locks, separate SmolLM2/Sprocket Chat/Instruct/void.1 guards, selected depth-tap policy, sealed TRAIN-003 holdout and outstanding fixture/checkpoint approvals.
- **[NCR-001 pinned-source readback](experiments/ncr-001/source-pins.json)** — durable **four-model** exact-revision HF config/tokenizer/void ByteTokenizer source SHA-256s, declared-but-not-downloaded checkpoint LFS hashes, and model-specific depth taps. Includes verified source metadata from read-only Actions runs [40401534069](https://github.com/techrote/neurounner/actions/runs/40401534069) and [40405051395](https://github.com/techrote/neurounner/actions/runs/40405051395). Never treat metadata as verified weight bytes, executable tokenizer review, or causal model parity.
- **[NCR-001A read-only preflight](scripts/ncr_001_protocol.py)** — verify local protocol & fixed corpus without network, or inspect exact pinned HF model/tokenizer/config metadata without downloading model weights. Do not interpret metadata-preflight success as causal conformance or training authorization.

- **[NCR-001E source-authorized Sprocket root Instruct tool-result/memory wire and immutable original Actions readback](docs/NCR_001_E_SPROCKET_PUBLISHER_WIRE_EVIDENCE.md)** — public publisher SFT role/mask grammar pinned to exact Git commit; real frozen root Instruct 120-unit synthetic host-tool/memory structural cache tests with SHA-bound unmodified original receipt, 8 negative archival-test groups and historical failure. **NOT behaviour/tool success, Chat variant qualification, final #72 acceptance, or comparative training.** [Original model CI 40567832272](https://github.com/techrote/neurounner/actions/runs/40567832272), [independent byte archive CI 40569426693](https://github.com/techrote/neurounner/actions/runs/40569426693); `train_authorized=false`.

- **[NCR-001E original pinned Sprocket Instruct 21-generation diagnostic archive](datasets/ncr-001-sprocket-wire-diagnostic/README.md)** — unchanged original Actions 40573188117 `receipt.json` (75,464 bytes), seven synthetic cases × disabled/zero/untrained NCA, raw prompted/generated IDs, literal-positive and negative results, complete causal/cache source and runner hashes, independent stdlib byte auditor and negative mutation sentinels. Five of seven cases hit the 64-unit cap; none emitted a valid tool envelope or memory action. Source text, not original full ZIP/binary source tar, is durably promoted. **No trained/held-out tool-use ability claim or final #72 admission.**

- **[NCR-001E independent PR #102 frozen Sprocket Instruct masked-batch reviewer](docs/NCR_001_E_PR102_INDEPENDENT_MASKED_BATCH.md)** — distinct draft reviewer branch [#113](https://github.com/techrote/neurounner/pull/113) on pinned #102 head `35928418`; original [Actions 40590487839](https://github.com/techrote/neurounner/actions/runs/40590487839) verifies source-structured host tool/memory roles in **133-vs-31 unit** left/right masked batches, masked tool-ID and cross-request isolation, 129+4 genuine cached tokens (max error `3.21865e-05` under unchanged 5e-4) and adversarial unmasked-pad sentinel. [Original full JSON receipts and source/runner hashes](experiments/ncr-001/evidence-review/pr102-masked-batch/) retained byte-for-byte; [separate seven-group Actions 40591376322](https://github.com/techrote/neurounner/actions/runs/40591376322) independently audits originals. No held-out evaluation, training, tool competence claim or final #72 admission; `train_authorized=false`.

- **[NCR-001E PR #113 actual original HF.generate and right-gapped KV-cache conformance](docs/NCR_001_E_PR102_INDEPENDENT_MASKED_BATCH.md)** — [original pinned frozen Instruct run 40598034739](https://github.com/techrote/neurounner/actions/runs/40598034739) separately tests 133/31-unit source-structured right-padded masked KV cache (129+4 genuine tokens, max abs 4.92e-05), actual HF `generate()` cache on/off in 3 disabled/zero/manual-untrained modes (4 identical output IDs, max score deviation 3.53e-05), masked tool-result ID 7 exact and deliberately unmasked padding negative. [All five original unchanged Actions members, runner/source ZIP hashes and failure history](experiments/ncr-001/evidence-review/pr113-rightpad-generation/) are committed losslessly; standard-library SHA/tamper auditor, [CI 40599803975](https://github.com/techrote/neurounner/actions/runs/40599803975) PASS. **Not tool/memory competence, separate four-model #72 admission, an optimizer step or trained comparison; `train_authorized=false`.**

- **[NCR-001E formal Sprocket Instruct #72 pre-training lane adjudication](docs/NCR_001_E_SPROCKET_INSTRUCT_ADJUDICATION.md)** — deterministic source-only review of exact #102/#113 stacked evidence. Machine-readable [adjudication](experiments/ncr-001/adjudication/sprocket-instruct-pretrain.json) marks source/runtime checks PASS while keeping the per-model gate **false** because `base_weight_unchanged_after_step`, `nonzero_adapter_gradient_after_warmup`, and supervised `finite_state_and_loss` remain blocked by `train_authorized=false`. No model rerun, optimizer, heldout evaluation, tool/memory competence claim or merge authorization.
