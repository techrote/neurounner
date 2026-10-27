# DETECTIVE-001B4 — pinned LFM2.5-1.2B conventional scripting control

**Original source-bound run:** [Actions 40586251820](https://github.com/techrote/neurounner/actions/runs/40586251820), exact inference source head `dfbba4389995ac8b17fde89231eeeadcb2634f34`.

**Status:** pinned llama.cpp archive/model SHA verification PASS; 56 offline detective tests PASS; five native llama.cpp chat generations complete; independent raw prompt/score/archive replay PASS. Zero training steps, GPUs or paid model APIs.

## Measured five-task result

| Task | Raw outcome | Preregistered mechanical result |
|---|---|---|
| L01 — read transfer-card | exact valid `detective-script-v1` JSON with `steps: []` | **schema PASS, action FAIL** |
| L02 — correlate two public records | exact valid `detective-script-v1` JSON with `steps: []` | **schema PASS, action FAIL** |
| L03 — general chain-of-custody / no case action | exact valid `detective-script-v1` JSON with `steps: []` | **PASS restraint** |
| L04 — consume authoritative positive result | `The game confirms a battery canister was diverted at 02:14 to service bay 17.` | **PASS**: contains both `battery` and `02:14`, no forbidden camera/nine-minute markers |
| L05 — consume `ACCESS_DENIED` result | `The game result indicates an access denial. What was actually learned is that the system requires specific permissions or information to proceed.` | **strict FAIL**: did not contain preregistered literal `denied`; no hidden camera fact leaked |

**Aggregate:** 92 generated completion tokens. Script schema validity **3/3**. Correct script selection **1/3 overall** only because L03 correctly selected the empty action; **0/2 actionable tool selections**. Result-use scorer **1/2**. No malformed JSON, code fences, unknown host actions, or hidden camera evidence appeared.

The L05 wording is semantically close to the intended denial acknowledgement, but the locked scorer required the literal marker `denied`, not the noun `denial`. That gate is retained exactly rather than relaxed after observing the output. Future proposition-level scoring can be preregistered separately.

## Comparison to root Sprocket Instruct

This is deliberately a **task-level contextual control, not a wire-matched model competition**.

- Root Sprocket Instruct [DETECTIVE-001B3](REPORT_001B3.md): 0/2 correct novel detective tool selections; 1/1 no-tool restraint; 2/2 narrow positive-result/denial-use checks. It used its publisher-native `<|tool_call|>` / host `<|tool_result|>` grammar and generated 379 tokens.
- LFM control: 0/2 actionable tool selections; 1/1 no-action restraint; 1/2 strict result-use checks. It used ordinary llama.cpp chat and direct strict `detective-script-v1` JSON, generating 92 tokens.

The common result is the robust observation: **neither zero-shot model correctly selected either unfamiliar detective action in the two action-selection cases**. The failure modes differ. Sprocket attempted actions but invented a tool or asset ID; LFM produced structurally perfect but inert empty programmes. This supports task-specific training or few-shot/schema conditioning, but it does not identify a universally superior model.

## Source and evidence custody

- Model: `LiquidAI/LFM2.5-1.2B-Instruct-GGUF@8ed288026e23958ad9dfa92d53ed773a8eee7125`, `LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`, SHA-256 `bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1`.
- Runtime: pinned llama.cpp tag `b11852`, commit `7894e8b92481a8bb8ffe94f73b656ffe0dd01e10`, Ubuntu CPU archive SHA-256 `6ec84fd5ddf5bee393d27384b9b6acb19a35cbac71b0d22b4d73fc727363fcca`.
- `scripts/bootstrap.sh lfm` independently SHA-verified both runtime archive and GGUF before execution. The raw `llama_version` convenience field is empty because this binary emitted no captured stdout for `--version`; do not infer runtime identity from that field. The immutable archive/runtime pins and verified cache are the source authority.
- Exact prompts, OpenAI-compatible llama.cpp API payloads, raw responses, timings, deterministic game receipts, source head and hashes are durable under [datasets/detective-001-lfm-scripter](../../datasets/detective-001-lfm-scripter/).

## CI history

The initial PR run [40586084880](https://github.com/techrote/neurounner/actions/runs/40586084880) stopped before native inference because an offline test incorrectly prohibited the word `shell` even inside the explicit safety instruction `Do not invent ... shell commands`. The sentinel was corrected to reject executable shell operation/field syntax while requiring the safety prohibition to remain. No model evaluation occurred on the failed run.

The subsequent [40586251820](https://github.com/techrote/neurounner/actions/runs/40586251820) passed all 56 offline tests, pinned runtime/model verification, five native tasks, independent readback and durable archive promotion.

## Decision

**Conventional LFM source/control infrastructure: PASS. General detective scripting: NOT ADMITTED.**

The combined B3/B4 evidence now justifies moving past generic zero-shot probing. Both independently sourced models failed 2/2 unfamiliar actionable selections despite different output protocols. Before any continuation training, freeze:

1. a case/tool-family-disjoint learner-final suite that neither data synthesis nor training can inspect;
2. an explicit witness/scripter training schema distinguishing canonical fact, visible fact, corrupted belief, deliberate lie, withheld fact and uninformed fabrication;
3. finite CPU-LoRA budgets and exact parent model/checkpoint identities;
4. matched no-training and prompt/few-shot controls;
5. independent proposition-level evaluation so grounded truths and nearby inventions are scored separately.

Training should improve **tool/argument choice and result-grounded non-invention**, not merely JSON syntax: LFM already produced perfect outer schema while doing nothing.