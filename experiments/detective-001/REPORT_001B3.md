# DETECTIVE-001B3 — source-correct Sprocket-500M Instruct fictional scripter result

**Original native run:** [Actions 40584860609](https://github.com/techrote/neurounner/actions/runs/40584860609), exact model-execution source head e7a81b1eaca8f17f8c13e98593c38003dc97d4ba.

**Source/evidence status:** publisher Git source and blob IDs PASS; root Instruct HF checkpoint/tokenizer hashes PASS; Chat checkpoint excluded; 49 offline detective tests PASS after one earlier harness-sentinel failure was corrected; five native generations complete; independent raw/receipt/archive replay PASS. No training or paid model API.

## Measured five-task result

| Task | Observation | Mechanical verdict |
|---|---|---|
| S01 — read transfer-card | emitted native `<|tool_call|>` JSON but chose unadvertised `read_file(path=assets/case_file.txt)` | **FAIL selection**; call syntax was recognizable, but the proposed tool is not admitted by the game |
| S02 — correlate two records | emitted native `read_asset` but invented asset ID `9d8f9d8d-4fbfd3a-48fd-88dcbfd9dc4f` | **FAIL selection/arguments**; typed game interpreter returned `ACCESS_DENIED` |
| S03 — no-tool restraint | answered chain-of-custody question in prose, no action marker | **PASS restraint**; answer ignored the requested one-sentence brevity and ran to 96 tokens |
| S04 — consume positive host tool result | included `02:14` and battery, with no camera/nine-minute marker | **PASS narrow result-use scorer**, but added unsupported details such as multiple players/day-and-a-half language |
| S05 — recover from access denial | began `DENIED`, said it had no record of access, did not reproduce hidden camera fact/F-CAMERA | **PASS denial/recovery scorer**, verbose but did not leak the inaccessible evidence |

**Aggregate:** 379 generated tokens; 5/5 attempted tasks; **0/2 correct model-selected detective tool actions**, 1/1 no-tool restraint, 2/2 narrow result/denial-consumption checks. The archived manifest field `native_tool_calls_valid=1` means *successfully parsed and translated into an admitted typed game tool*, not merely syntactically recognizable publisher `<|tool_call|>` output. S01 had publisher-shaped tool-call syntax but failed translation because `read_file` is not an admitted game operation.

## Interpretation

This is stronger than the previous untuned SmolLM2 direct-script baseline in one narrow respect: the root Sprocket Instruct checkpoint clearly produced its publisher-native tool-control syntax and responded to host-injected `<|tool_result|>` turns. It did **not** demonstrate robust generalization to the new detective tool manifest. The actual game-tool selection score was 0/2: one invented a generic file tool, one selected a valid tool with a fabricated asset identifier.

The positive result-consumption observations are also weaker than a semantic correctness claim. S04 recovered the two preregistered markers from the real synthetic game receipt, but embellished the story. S05 respected denial and did not expose hidden source text. A larger battery needs proposition-level scoring so retaining a true detail does not hide adjacent inventions.

The publisher's own documentation says tool calling worked on its training distribution while memory did not. This run is compatible with a model that learned the *mechanism* and some result-following but has weak transfer to unfamiliar tool names/identifiers. That is a research hypothesis, not a demonstrated causal explanation.

## Source custody

- Model: root `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, `model.safetensors`, 1,002,207,984 bytes, SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`.
- Publisher source: `kandivault-source/Sprocket-500M@a003767e70d766bc14b40c89792543ee8641c8bc`; SFT renderer Git blob `90bddf8976b9d6410fe7c5c1f084708e1cebab73`; harvest placement validator blob `97811aa4245ad3f05f66ce0c2d8aa8c353619267`. This source relationship is **not** treated as cryptographic checkpoint-build attestation.
- Runtime: AMD EPYC 9V74, four threads, Python 3.12.15, PyTorch 2.14.1+cpu, Transformers 4.57.6; peak process RSS 3,318,656 KiB.
- Durable raw evidence: [datasets/detective-001-sprocket-scripter](../../datasets/detective-001-sprocket-scripter/), including exact prompt turns, decoded control tokens, output token IDs, deterministic typed receipts, source hashes and file inventory.

## CI history

The first PR check [40584748713](https://github.com/techrote/neurounner/actions/runs/40584748713) stopped before model execution because an offline fake-tokenizer negative test raised `StopIteration` instead of the required `ContractError`. The source gate therefore skipped native inference. The harness was hardened to convert tokenizer lookup exceptions into fail-closed contract errors. The subsequent [40584860609](https://github.com/techrote/neurounner/actions/runs/40584860609) ran all 49 offline tests, exact publisher source checks, root checkpoint inference, independent receipt replay and evidence promotion successfully.

## Decision

**Publisher/root-Instruct source and fictional tool bridge: PASS. General detective scripting ability: NOT ADMITTED.**

The root Sprocket Instruct model is now a credible candidate for **task-specific scripting continuation**, but the current zero-shot detective transfer is insufficient: 0/2 correct novel tool selections. Before training it, compare an independently source-pinned conventional small instruct model on the same fictional tool vocabulary and add a fresh case/tool-family holdout. If a Sprocket continuation is trained, preserve the original root checkpoint as its control and optimize exact tool/argument selection plus result-grounded non-invention, not just control-token syntax.