# DETECTIVE-003B — live two-witness session integration smoke

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking: [#142](https://github.com/techrote/neurounner/issues/142). Session authority: merged DETECTIVE-003A #137 / PR #141.

This is the first source-bound **actual model generation** routed through the authoritative multi-witness session ledger. It is an integration smoke, not a new policy benchmark or model admission.

## Exact pilot cartridges

Both witnesses use the same frozen SmolLM2-135M-Instruct base:

- revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
- model weight SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.

### V-G02-A

Exact C3 research cartridge:

- release `detective-001c3-g02-v1`;
- archive SHA `8b0f93e51750d09bc02e464f1327028c415ad738968f26e3ea6934540df4a57a`;
- adapter directory SHA `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3`.

C3's locked GO gate remains FAIL; using this cartridge here does not change that result.

### V-G09-C

Exact frozen C2 survivor/control:

- release `detective-001c2-v1`;
- archive SHA `91fb3bbf697cd39646fc22b6cbbb24902304893a7c0f1250112b0079d9670e84`;
- adapter directory SHA `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c`.

No new training occurs.

## Authority binding

003B Git-blob-binds:

- merged DETECTIVE-003A session module;
- Bay Seventeen case;
- C2 adapter-release manifest;
- C3 adapter-release manifest.

Release archives are SHA/byte checked and extracted with path-traversal/symlink/device refusal. Extracted PEFT directories must match their preregistered directory hashes.

The base model weight is downloaded at the exact revision and SHA-verified before model construction.

## Runtime

- Ubuntu GitHub runner;
- four Torch CPU threads;
- pinned Torch 2.14.1+cpu / Transformers 4.57.6 / PEFT 0.21.2;
- base loaded once;
- G02 and G09 loaded as distinct named PEFT adapters;
- orchestrator calls `set_adapter` based only on game-owned witness mapping;
- greedy fixed seed;
- <=96 generated tokens/turn;
- three model turns / <=288 generated tokens total;
- no CUDA;
- no trainable parameters;
- zero optimizer steps;
- zero paid APIs.

## Fixed Bay Seventeen sequence

1. G02: ask remembered battery handoff time.
2. G09: ask battery handoff time.
3. Game presents the public transfer receipt to G02. Any prior G02 `handoff-time` claims are marked challenged.
4. G02 follow-up with **same-witness history enabled**: the physical receipt is in front of the witness; ask whether the account changes.

Every model prompt originates from `DETECTIVE-003A build_prompt`, then uses one deterministic natural renderer. The model never sees orchestrator member IDs, adapter names, claim IDs, contradiction edges, fact IDs, asset IDs, terminal credentials, hidden relevance labels or canonical truth labels.

## Narrow game-owned annotation

This fixture does **not** introduce a general semantic judge.

After each exact raw model response, the game scans only for strict `HH:MM` literals:

- exactly one distinct time -> add a game-owned `handoff-time` statement claim;
- zero times -> `no_time`, no claim;
- multiple distinct times -> `ambiguous_times`, no claim.

Repeated occurrences of the same time still count as one distinct value.

The extractor never reads canonical truth, evidence facts, C1/C2/C3 learner-finals or a policy scorer. It cannot repair ambiguous output.

Therefore:

- claim count may be zero;
- contradiction count may be zero;
- either outcome may still establish successful session integration.

## Durable evidence

The live run stores:

- exact source/base/adapter identities;
- private prompt receipts;
- model-facing rendered prompts/messages;
- prompt token count/digest;
- exact response plus generated token IDs;
- literal-time extraction status/literals/annotations;
- evidence action/challenged-claim IDs;
- complete final session event log;
- independently replayed derived claims/contradictions;
- canonical case before/after SHA;
- runtime timing/RSS;
- source/run provenance.

A standard-library verifier reconstructs the session from the stored raw responses and fixed sequence. It recomputes each prompt receipt, natural rendering, literal annotation, challenge list, event hash chain, final session SHA, claims and contradictions.

## Acceptance

Infrastructure PASS requires:

- exact source/base/adapter hashes;
- three bounded generations;
- deterministic prompt/session replay;
- public evidence presentation through session authority;
- unchanged canonical case SHA;
- intact mandatory public clue routes.

It does **not** require correct testimony, a claim, a contradiction or a policy score.

## Evidence-ready fast path

The first owner PR run may download the exact base/adapters and execute the three turns.

After independently verified evidence is committed under `datasets/detective-003b/`, subsequent CI runs **skip all model downloads/inference** and replay only the durable archive.

## Non-authority

003B cannot:

- change the C2/C3 policy verdicts;
- satisfy DETECTIVE-001R1 #121;
- select a model/body for further training;
- read any learner-final prompt;
- train any model;
- admit population expansion;
- adjudicate whether an extracted time is true;
- unlock non-public evidence from dialogue.

The useful result is solely whether real generated dialogue can flow through the authoritative interview-session machinery without crossing the game/model authority boundary.


## Executed outcome — 2026-10-28

Authoritative source-bound execution: [Actions 40616799855](https://github.com/techrote/neurounner/actions/runs/40616799855), exact live source `485608478bdffbd7cf7bbe2206761128f7adadc9`.

- exact G02-C3 and G09-C2 adapters plus pinned SmolLM2 base verified and loaded;
- all three bounded generations completed;
- **70 total generated tokens**, zero truncations;
- literal-time annotations: G02 initial = `02:14` statement, G09 = `no_time`, G02 post-evidence = `no_time`;
- final ledger = **1 claim, 0 contradictions**;
- public transfer receipt challenged `C0001` through evidence event `E0003`;
- final session SHA `ae2357a11cc4cee09ee077b8bdd3c193c2b5ecb9103b3a3e588812702df6a85e`;
- canonical case before/after SHA remained identical;
- mandatory public clue routes remained intact;
- independent reconstruction now passes from durable raw evidence;
- zero optimizer steps, zero paid APIs, no relevance-map or learner-final access, no body selection, no population expansion.

The first source-bound model execution actually succeeded; its initial workflow failure was only the standalone verifier's missing repository-path bootstrap. The verifier was fixed without changing any live-source/config/model/session file, and the rerun produced the same session hash and was independently promoted.

See [REPORT_003B.md](REPORT_003B.md) and [datasets/detective-003b](../../datasets/detective-003b/).

This is **infrastructure integration PASS only**. G09's non-answer and G02's non-time follow-up remain raw weak behaviors; the harness does not invent claims for them. C2/C3 policy verdicts and DETECTIVE-001R1 remain unchanged.
