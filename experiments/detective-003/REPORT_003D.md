# DETECTIVE-003D — source-native Sprocket live bridge result

Authoritative live run: [Actions 40618669330](https://github.com/techrote/neurounner/actions/runs/40618669330), exact model-execution source `6a2274568a63219211b1e50b70f4bb341f860917`.

**Integration path: PASS. Scripter tool selection: FAIL. Production scripting admission: NOT ADMITTED.**

## Exact source

- root `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`;
- root `model.safetensors`, 1,002,207,984 bytes;
- SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`;
- publisher source `kandivault-source/Sprocket-500M@a003767e70d766bc14b40c89792543ee8641c8bc`;
- publisher SFT/harvest blobs matched the frozen B3 authority;
- Chat checkpoint excluded;
- tokenizer and special-token hashes matched.

## Raw model output

Sprocket generated 35 tokens:

```text
<|tool_call|>{"name":"get_database","arguments":{"query":"fictitious native tools for the player's terminal"}}<|end|>
```

This is a useful separation of concerns:

1. **publisher-native control syntax parsed successfully**;
2. the proposed tool was **not admitted by the detective game**;
3. strict one-call normalization therefore rejected it with `ContractError: tool not admitted by game`;
4. no `detective-script-v1` programme existed;
5. no DETECTIVE-003C transaction executed;
6. no access grant or evidence application occurred;
7. the interview session remained unchanged.

Mechanical result:

- native parse: **parsed**;
- normalization: **rejected**;
- exact intended `probe_terminal(camera-console, OXIDATION)`: **false**;
- transaction: **none**;
- camera-cache visible: **false**;
- evidence applied: **false**;
- outcome: `native_or_normalization_rejected`.

## Relation to historical B3

DETECTIVE-001B3 previously measured **0/2 correct novel detective tool selections**:

- one publisher-shaped call invented `read_file`;
- one used an admitted tool with a fabricated asset ID.

003D independently reproduces the same broad failure mode in the integrated game path: the model retains the native tool-call mechanism but does not reliably bind an unfamiliar detective tool vocabulary. Here it invented `get_database`.

This strengthens the narrow hypothesis that the root Instruct checkpoint learned a tool-control protocol but transfers poorly to unseen tool names/semantics. It does **not** establish why; that remains a research interpretation.

## Authority behavior

The failure is exactly what the 003C boundary is intended to contain.

Although the raw model output looked like a structured tool call:

- no fuzzy mapping converted `get_database` into `probe_terminal`;
- no textual repair replaced the arguments;
- no hidden asset ID or credential was guessed on the model's behalf;
- no game transaction executed;
- no witness received evidence;
- no canonical case state changed.

The deterministic session stayed at its initial SHA:

`7cb2c8b3e9ad1ad455769b2160e013edbae5db981bdefcab983d129013d1764d`.

## Runtime

Runner: AMD EPYC 7763, four Torch CPU threads.

- model load: **0.327 s**;
- RSS after load: ~**2.28 GiB**;
- peak RSS: ~**3.17 GiB**;
- generation: **35 tokens / 8.32 s = 4.21 tok/s**;
- no 96-token truncation.

These are incidental source-native integration measurements, not a general Sprocket benchmark.

## Decision

003D establishes that the complete live chain is safe under a real model failure:

**publisher-native generation → strict parser → strict normalization rejection → zero authority effect**

The model is **not** admitted as a zero-shot detective scripter.

The next meaningful scripter experiment should be task-specific specialization or a matched structured control—not more prompt-time repair of the same zero-shot root checkpoint. Any specialization must preserve this exact root checkpoint/output as its control and use fresh tool/case-family holdout evidence.
