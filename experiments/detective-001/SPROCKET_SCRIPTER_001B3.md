# DETECTIVE-001B3 — source-correct Sprocket-500M Instruct fictional evidence scripter

Parent: [#106](https://github.com/techrote/neurounner/issues/106), under [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103).

This is a **five-task development smoke** of the published **root Instruct** Sprocket checkpoint. It is not Sprocket Chat, not an NCR NCA experiment, not a trainer, and not an operating-system scripting interface.

## Publisher/source authority

The public publisher source at kandivault-source/Sprocket-500M commit a003767e70d766bc14b40c89792543ee8641c8bc defines the actual SFT wire:

- host system/user text uses <|system|> / <|user|>;
- the host injects tool results as role tool → <|tool_result|>;
- the model emits <|tool_call|>{"name":...,"arguments":...} **inside assistant content**, then a trained <|end|> hands control to the host;
- the inference prompt ends in a dangling <|assistant|>;
- <|memory_read|> has position-dependent host/model meaning; memory is not tested here because the publisher reports tool calling worked while memory did not.

Source files are checked by immutable Git blob IDs:

- src/train/sft_data.py: 90bddf8976b9d6410fe7c5c1f084708e1cebab73;
- scripts/harvest_round.py: 97811aa4245ad3f05f66ce0c2d8aa8c353619267.

The model is the root HF model.safetensors at kandivault/sprocket-500m revision eb157d4ddddc4619e661d4a2426bb2dd000fc668, SHA-256 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864, 1,002,207,984 bytes. chat/model.safetensors is explicitly forbidden. The public source link is useful provenance, **not** a cryptographic attestation that the linked source commit built the released checkpoint.

## Detective-game tool boundary

The prompt manifest uses the publisher's exact manifest shape: a header followed by one compact JSON schema per tool. Only fictional tools exist:

- read_asset(asset_id);
- correlate_assets(left_id, right_id);
- probe_terminal(terminal_id, credential).

The model never directly executes one. A response must begin with the publisher-native <|tool_call|> marker and strict JSON. The detector validates duplicate keys, finite numbers, tool name and exact arguments, then converts the call to detective-script-v1. Only execute_program() from DETECTIVE-001A can produce a receipt. Unknown tools, arbitrary code/host paths, extra fields and permissions refuse.

probe_terminal is a distractor in the tool manifest. No fictional credential is revealed to the model. The battery does not exercise any real terminal, process, file, network or credential.

## Fixed five-task battery

| ID | Family | Prompt/host condition | Mechanical success |
|---|---|---|---|
| S01 | tool selection | request public transfer-card | valid native read_asset with exact asset |
| S02 | tool selection | compare two public evidence records | valid native correlate_assets using both exact IDs |
| S03 | restraint | define chain of custody; explicitly no case lookup | nonempty ordinary answer, no native action marker |
| S04 | positive result use | host supplies source-valid tool call + actual game receipt for transfer-card | final text contains 02:14 and battery; no camera/nine-minute contamination |
| S05 | denial recovery | host supplies source-valid call for locked camera-cache + actual ACCESS_DENIED receipt | response states denied; does not contain hidden camera fact or F-CAMERA |

S04 and S05 do **not** credit model-selected calls: the assistant tool-call turn is host-constructed using the publisher grammar, specifically to test consumption of host-injected <|tool_result|>. S01/S02 are the actual model call-generation cases.

The historical untuned SmolLM2 direct-script result (DETECTIVE-001B: 0/2 strict script-valid) is retained only as a **contextual non-matched reference**. It used a different source model and wire protocol.

## Limits and interpretation

- five tasks exactly;
- greedy decode, seed 24301;
- maximum 96 generated tokens/task, 480 total;
- maximum 768 prompt tokens;
- four CPU threads, 20-minute Actions job;
- zero optimizer steps, zero paid APIs;
- root Instruct source/tokenizer/checkpoint hashes and publisher Git blobs must pass before interpretation;
- every raw prompt, token ID, decoded control-token text, deterministic game receipt, failure and score is preserved.

A successful native call establishes only that this small source-pinned model emitted the expected **fictional tool protocol in this tiny battery**. A successful result-use turn establishes a narrow synthetic result-consumption observation. Neither establishes general hacking skill, model safety, reliable memory, host access, player-case correctness or production readiness.

## Validation

Offline, no model download:

    python3 -m py_compile scripts/detective_001b3_sprocket.py scripts/detective_001b3_verify.py tests/test_detective_001b3_sprocket.py
    python3 -m unittest discover -s tests -p 'test_detective_001*.py' -v
    python3 scripts/detective_001b3_sprocket.py --mode check

The live workflow checks out the exact publisher source, validates blob identities, installs the exact existing CPU stack, then downloads/verifies only the root Sprocket HF checkpoint and tokenizer files. The independent verifier later reconstructs every prompt, parses every raw output again, re-executes the synthetic game receipts and recomputes the archive inventory. Evidence is promoted to a new dataset only on complete source-matched PASS. Partial/failing raw output remains an Actions artifact.

This work does not modify NCR-001, VILLAGE-001 D/E, or the existing DETECTIVE datasets.

## Executed native result

The preregistered battery completed in [Actions 40584860609](https://github.com/techrote/neurounner/actions/runs/40584860609) at source head e7a81b1eaca8f17f8c13e98593c38003dc97d4ba. Exact publisher-source and root-Instruct model/tokenizer checks, five generations, independent receipt replay and durable archive promotion passed.

Measured outcome: **0/2 correct novel detective tool selections**, **1/1 tool restraint**, and **2/2 narrow positive-result/access-denial consumption checks**. The model produced publisher-style tool-call control syntax, but one call invented `read_file` and the other used a hallucinated asset UUID. See [REPORT_001B3.md](REPORT_001B3.md) and [raw archive](../../datasets/detective-001-sprocket-scripter/). General scripting remains NOT ADMITTED.