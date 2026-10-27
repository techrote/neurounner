# DETECTIVE-001B4 — conventional LFM instruct control

Parent issue: [#109](https://github.com/techrote/neurounner/issues/109), under [#106](https://github.com/techrote/neurounner/issues/106).

This is a **task-level**, not wire-format-matched, control for the Sprocket root-Instruct scripting result.

## Exact source

- LiquidAI/LFM2.5-1.2B-Instruct-GGUF
- revision 8ed288026e23958ad9dfa92d53ed773a8eee7125
- LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf
- SHA-256 bb741ebb106d543e9de114b843a3d3d73d51c74b5801e69da2abde821a0cb3e1
- pinned llama.cpp b11852 / 7894e8b92481a8bb8ffe94f73b656ffe0dd01e10

The existing Neurounner qualification measured this GGUF as a coherent, fast routine compiler; it did not qualify detective scripting.

## Five semantic tasks

L01/L02 require strict detective-script-v1 JSON for reading transfer-card and correlating the two public records. L03 requires the same schema with an empty steps array for a general chain-of-custody question. L04/L05 provide the same authoritative positive transfer-card result and ACCESS_DENIED result used in the Sprocket semantic battery, but as ordinary chat text.

LFM output is inert until strict JSON parses through DETECTIVE-001A and the deterministic fictional authority. Code fences, prose around JSON, unknown operations and extra fields fail. No model-generated host command, file path, network operation or real credential is executable.

## Limits

- five tasks, 96 completion tokens/task / 480 maximum;
- temperature 0, top-p 1, seed 24301;
- 4 CPU threads, 2048 context, 15-minute hosted job;
- zero training and zero paid APIs;
- source runtime/model SHA verification before scoring;
- raw API response, prompts, output, timing and deterministic receipts retained;
- independent standard-library archive replay before persistent promotion.

Interpret results only as a conventional zero-shot reference. Sprocket uses its publisher-native tool wire, so direct model ranking is not justified from five tasks.