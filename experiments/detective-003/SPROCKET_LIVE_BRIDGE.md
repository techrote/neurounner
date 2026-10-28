# DETECTIVE-003D — source-native Sprocket live scripter bridge

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#146](https://github.com/techrote/neurounner/issues/146). Transaction authority: merged DETECTIVE-003C #144 / PR #145.

This is the first actual **publisher-native Sprocket tool-call generation** routed into the authoritative 003C transaction bridge.

It is deliberately a no-training integration smoke. The historical DETECTIVE-001B3 result already showed weak zero-shot detective tool selection (0/2). 003D does not hide or repair another failure if one occurs.

## Exact source authority

003D Git-blob-pins:

- frozen B3 Sprocket source/config contract;
- B3 publisher-native parser/wire implementation;
- merged 003C bridge config/module;
- merged 003A interview session module;
- Bay Seventeen case.

Model:

- `kandivault/sprocket-500m`;
- revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`;
- **root Instruct** `model.safetensors`;
- SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`;
- 1,002,207,984 bytes.

`chat/model.safetensors` is explicitly forbidden.

Publisher source remains:

- `kandivault-source/Sprocket-500M@a003767e70d766bc14b40c89792543ee8641c8bc`;
- SFT renderer Git blob `90bddf8976b9d6410fe7c5c1f084708e1cebab73`;
- harvest validator blob `97811aa4245ad3f05f66ce0c2d8aa8c353619267`.

The public source relationship remains provenance, not a cryptographic checkpoint-build attestation.

## Game request

003D starts a fresh Bay Seventeen 003A session and builds one real 003C scripter request.

The game exposes:

- public `dock-clock-ledger`;
- public `transfer-card`;
- disclosed `camera-console`;
- explicitly player-known synthetic credential `OXIDATION`.

Player request:

> Use the disclosed credential on the disclosed terminal. If access grants new evidence, stop and let the game inspect the receipt.

The 003C model-facing view does **not** expose `camera-cache`, hidden evidence excerpts, fact IDs/text, terminal grants, witness identities, claim state or hidden relevance labels.

## Publisher-native prompt

The model is not run through a generic chat template.

003D reuses the frozen B3 wire:

- `<|system|>` with the three fictional tool schemas;
- `<|user|>` with the bounded player request + exact 003C game view;
- trained `<|end|>` separators;
- dangling `<|assistant|>` generation cue;
- exact publisher special-token IDs;
- greedy decode, fixed seed;
- max 96 generated tokens.

Raw generated token IDs and special-token-preserving decoded output are retained.

## Strict native normalization

The raw output first passes the frozen B3 native parser.

### Native rejected

If the output is prose, malformed JSON, contains illegal control placement, duplicate keys or otherwise fails native parsing:

- raw output remains preserved;
- no programme is produced;
- no 003C transaction executes.

### Native parsed, normalization rejected

A syntactically valid publisher call may still propose a non-game tool such as historical B3's `read_file`, wrong argument fields, or another inadmissible call.

Normalization refuses without textual repair, fuzzy mapping, argument repair or ID substitution.

No 003C transaction executes.

### Normalized

Only one exact admitted native call may become one one-step `detective-script-v1`.

The canonical programme text then enters 003C, which independently re-parses and executes it through DETECTIVE-001A.

This means the native adapter itself never grants authority.

## Expected probe observation

The game request makes the exact intended call mechanically identifiable:

```json
{
  "name": "probe_terminal",
  "arguments": {
    "terminal_id": "camera-console",
    "credential": "OXIDATION"
  }
}
```

003D records whether the raw model output equals that call.

**It is not an acceptance gate.**

A wrong valid detective tool, an invented tool, wrong argument or prose response can still establish integration PASS.

## Optional evidence application

If a valid normalized transaction's resulting visible set contains `camera-cache`, the game explicitly applies that private evidence to V-G09-C through 003C.

The resulting witness prompt must contain the private archive excerpt but no:

- `camera-cache`;
- `F-CAMERA`;
- `camera-console`;
- `OXIDATION`.

If the transaction does not grant the asset, no application occurs.

## Outcome classes

### `native_or_normalization_rejected`

No typed transaction executed.

### `normalized_executed_no_grant`

A valid one-step detective programme executed but did not make `camera-cache` visible.

### `normalized_executed_grant_applied`

The programme granted `camera-cache`, and the game separately presented it to G09.

All three are valid integration outcomes.

## Acceptance

Integration PASS requires:

- exact model/tokenizer/publisher source identities;
- one bounded source-native generation;
- exact raw output retained;
- deterministic native parse/normalization replay;
- deterministic 003C transaction replay when normalized;
- transaction execution alone does not mutate interview session;
- evidence application only when deterministic receipt grants it;
- no host execution.

It does **not** require:

- exact probe selection;
- access grant;
- evidence application;
- good prose;
- production scripter admission.

## Durable evidence

First owner run archives under `datasets/detective-003d/`:

- exact source hashes/publisher checkout;
- 003C private request receipt;
- publisher-native turns + rendered prompt;
- prompt token digest/count;
- exact raw output/token IDs;
- strict normalization result;
- optional 003C transaction receipt;
- optional resulting interview session/prompt;
- runtime/RSS/timing;
- source/run provenance.

A standard-library verifier reconstructs the request, native prompt text, parse/normalization, transaction, optional evidence application and final session from raw evidence.

After promotion, later CI uses a verifier-only fast path and skips model inference.

## Non-authority

No training, no prompt repair after observing output, no host shell/filesystem/network action, no hidden relevance or learner-final access, no body/model selection and no population expansion.

003D cannot change the historical B3 competence result or admit Sprocket as a production scripter.


## Executed outcome — 2026-10-28

Authoritative live run: [Actions 40618669330](https://github.com/techrote/neurounner/actions/runs/40618669330), exact model source `6a2274568a63219211b1e50b70f4bb341f860917`.

Sprocket produced valid publisher-native control syntax but selected an unadmitted tool:

```text
<|tool_call|>{"name":"get_database","arguments":{"query":"fictitious native tools for the player's terminal"}}<|end|>
```

Measured outcome:

- 35 generated tokens;
- native parser: **parsed**;
- one-call detective normalization: **rejected** (`tool not admitted by game`);
- exact intended `probe_terminal(camera-console, OXIDATION)`: **false**;
- 003C transaction: **none**;
- access grant: **false**;
- evidence application: **false**;
- final interview session remained unchanged;
- outcome `native_or_normalization_rejected`.

This is an **integration PASS under model failure**, not scripter competence. It reproduces the historical B3 pattern where Sprocket retains publisher tool-call syntax but generalizes poorly to unfamiliar detective tool vocabulary.

See [REPORT_003D.md](REPORT_003D.md) and [datasets/detective-003d](../../datasets/detective-003d/).

No training, prompt repair, host execution, hidden relevance/learner-final access, body selection or population expansion occurred.

Current retrieval/model-journal signposts are interpretation-only; durable raw evidence and model-free replay remain the authority.
