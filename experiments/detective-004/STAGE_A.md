# DETECTIVE-004A — frozen task-specific Sprocket detective-scripter specialization (Stage A)

## Current owner policy (29 October 2026): relevance gate retired, source gates unchanged

Issue [#121](https://github.com/techrote/neurounner/issues/121) retired **objective relevance nonregression as a population-expansion gate** on 28 October 2026. Its 24 externally supplied C2/C3 scores are preserved as **subjective gameplay-reaction evidence**, not objective relevance qualification. Older #121 language below is historical only. Stage A grants **no model training, scripter research qualification, population expansion or production admission**. Exact source/lineage custody, leakage/proposition controls, mandatory-clue solvability, typed game authority, independently reviewed Stage-A protocol, real-tokenizer masks/512-token checks, and a separately complete Stage-B execution contract remain mandatory.

The frozen `specialization.json` field `stage_b_approved_only_after_merge=true` specifies a **minimum sequencing constraint**: Stage-A merging is necessary, never sufficient for optimizer authorization. Do not modify the hash-bound config, generator or archived 36-row corpus to record this later policy decision.

Tracking: [issue #148](https://github.com/techrote/neurounner/issues/148). Parent: [#103](https://github.com/techrote/neurounner/issues/103). This is **offline, pre-optimizer preparation only**.

## Why

The source-native root Sprocket-500M Instruct mechanism works, but zero-shot fictional detective tool selection does not transfer. DETECTIVE-001B3 selected the correct novel tool **0/2** times. DETECTIVE-003D generated a syntactically valid native call to unadmitted `get_database`, correctly rejected by the authoritative game transaction boundary. Neither result supports production scripter admission.

Stage A freezes the exact specialization experiment rather than repairing zero-shot generations after inspection. It uses only the root checkpoint `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, root `model.safetensors` SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864` (1,002,207,984 bytes). The Chat subfolder checkpoint, Village witness adapters and NCR research adapters are not substitutes.

The source contract in [specialization.json](specialization.json) binds the exact B3 config, native parser, 003C game-view config, publisher source commit `a003767e70d766bc14b40c89792543ee8641c8bc`, publisher SFT renderer Git blob `90bddf8976b9d6410fe7c5c1f084708e1cebab73`, harvest blob `97811aa4245ad3f05f66ce0c2d8aa8c353619267`, all tokenizer-related B3 hashes, and exact token IDs. The publisher source is **not** an attestation that those exact source files built the checkpoint.

## Source-native dataset

The deterministic generator is [scripts/detective_004a_protocol.py](../../scripts/detective_004a_protocol.py). A complete 36-row corpus uses **six new synthetic case families**, all disjoint from Bay Seventeen, historical B3/003D responses, detective witness learner-finals and opaque relevance packets.

- **24** tool selection: 8 `read_asset`, 8 `correlate_assets`, 8 `probe_terminal`.
- **6** explicit no-tool restraint.
- **6** source-native tool-result continuations: 3 successful returns and 3 access denials.

Every family contributes four selections, one restraint and one tool-result continuation. IDs and credentials change by case family. No hidden evidence/permission/canonical-fact authority is added to the `detective-scripter-view-v1` model-facing projection. The game returns synthetic result text as a host-owned `tool` message.

Generate and independently verify the byte-identical corpus:

```sh
python3 scripts/detective_004a_protocol.py --write-training runtime/detective-004a/owner
python3 scripts/detective_004a_protocol.py --verify-training runtime/detective-004a/owner
python3 -m unittest tests.test_detective_004a_protocol -v
```

The generated `manifest.json` binds Git blob identities for source/config/generator, SHA-256 of the complete training JSONL, all 36 IDs, family/kind/tool counts, source weight metadata, and absence of model generations, repairs, optimizer steps and provider calls. Raw rendered native message records are retained, including assistant-only character ranges. **These ranges are a provenance aid, not token-label arrays.**

### Critical publisher SFT rule

The publisher's pinned `src/train/sft_data.py` declares:

- `<|system|>`, `<|user|>` and `<|tool_result|>` headers/content/end are **masked**.
- The `<|assistant|>` header is **masked**, because the host emits it.
- Assistant content—including model-emitted `<|tool_call|>`—is **trained**.
- Every assistant closing `<|end|>` is **trained**; it is how the model yields control back to the host.

Future Stage B **must re-tokenize with the frozen checkpoint tokenizer and compare publisher token-label behavior to this role contract before training**. It must assert each example fits the frozen 512-token maximum; do not silently truncate prompts or drop the trained closing token. No tokenizer/model download, token-count inference or optimization occurs in Stage A.

An author-side [source-native token-supervision preflight](../../scripts/detective_004a_token_preflight.py)
is prepared separately from the frozen generator. It replays the pinned
publisher's per-turn content encoding and assistant-only token-label/causal
shift rules, checks B3 whole-wire prompt-ID equivalence and refuses any
example over 512 tokens. Its Stage-A CI coverage uses **only an artificial
tokenizer**, so actual checkpoint tokenizer labels, prompt IDs and length
bounds are still **unverified**. The local-only real-tokenizer command and
remaining independent custody requirements are described in
[SECONDARY_CUSTODY_AUDIT.md](SECONDARY_CUSTODY_AUDIT.md). This is neither
independent Stage-A approval nor permission to run Stage-B optimization.

## Final custody

The recipe in the frozen config defines **three other case families** and exactly **12** final tasks: three per detective tool, one restraint, one success-result and one denied-result continuation. No final tasks or output files should exist at Stage A.

The future `build_final_in_memory` entry point refuses without:

1. a byte-perfect regenerated/verified training archive and its expected SHA-256 manifest;
2. a real adapter directory with at least `adapter_config.json` and `adapter_model.safetensors`;
3. the SHA-256 of that directory's sorted file/path/size manifest.

The post-hash seed binds the fixed recipe, salt, training manifest and adapter-directory identity. Materialization, final custody and final evaluation are Stage B responsibilities **only after Stage A merges**. The recipe doesn't admit a learned model or establish secrecy merely because the digest is later available. An independent final custodian should inspect any actual post-hash materialization and reject train/final ID overlaps before allowing evaluation.

## Final evaluator and fixed research gate

Strict evaluation operates on raw assistant output. Tool tasks require output beginning with a single publisher-native `<|tool_call|>`, strict B3 native parser acceptance, exact admitted tool name, exact argument names/values and normalization into the one-step fictional typed IR. Leading prose, multiple calls, unknown actions, invented IDs, duplicate JSON keys, and field repair fail. No game/session change is granted by model text.

No-tool restraint requires nonempty prose without native action markers. Result-use requires all frozen expected markers, none of the forbidden markers and no new action. Unit tests include adversarial formatting, invented IDs, cross-role control injection and masked-tool-result failure cases.

A future model becomes a **scripter-specialization research candidate**, never a production game scripter, only with all of:

- correct tool selection **at least 7/9**, including **at least 2/3 for each tool**;
- restraint **1/1** and returned-result/denial **2/2**;
- no out-of-view argument IDs/credentials;
- tool selection at least **4/9 better** than the **same final** on the exact unadapted frozen root checkpoint.

Failure ends the experiment. No after-the-fact threshold, training-budget or final-case changes.

## Finite future Stage-B resource contract

Exactly one frozen-root LoRA adapter, rank 8, alpha 16, dropout zero, `q_proj`/`v_proj` only, 36 single-example optimizer steps/one epoch, LR 2e-4, AdamW, clip 1.0, four CPU threads, at most 35 minutes, no GPU, no paid APIs, no sweep. **All 36 training examples are a planned deterministic corpus, not observations from the model.** Source-native checkpoint/LoRA loading, real tokenizer labels, and exact runtime hashes must be independently qualified before any Stage-B optimizer step.

## Separation of authority

This lane cannot access DETECTIVE-001R1 hidden identity/relevance maps or decide Village witness population admission. #121's historical objective relevance-review prerequisite is retired as an expansion gate; externally supplied ratings survive only as subjective gameplay reactions, not objective qualification. Nothing here reopens C1/C2/C3 learner-finals, affects NPC witnesses, alters the immutable Bay Seventeen case, executes host shell/network/filesystem actions from model output, or chooses a production body.

The generated 36-row dataset and source-bound manifest have now been durably promoted under [`datasets/detective-004a`](../../datasets/detective-004a/); Actions artifacts remain auxiliary staging only. The Stage-A pull request stays **draft until exact-head CI validates the committed archive and independent review approves the frozen protocol**. Neither archive promotion nor a green mechanical test is optimizer authorization.
