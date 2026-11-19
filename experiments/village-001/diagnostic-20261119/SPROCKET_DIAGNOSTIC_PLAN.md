# Sprocket structured-feedback diagnostic

**Status: prepared, not executed.** This is a follow-on diagnostic after Stage E completion. It performs no training and does not read or reuse the final F suite. It does not change the Stage D or E verdicts, adapters, recipes, or quality thresholds.

## Question and current evidence

The recovered Stage D archive contains **0/3,456 schema-valid Sprocket feedback records**. The cloud compiler consumed their retained raw text, as independently verified; this was not successful structured Sprocket criticism. The detailed source-bound audit is [structured-feedback-audit.json](../recovery-20261118/review/structured-feedback-audit.json).

The failure is usually free-form or repetitive output, not nearly valid JSON. There are 147 empty outputs, 2,713 `length` finishes, and 743 `stop` finishes. Exactly 2,721 outputs reach 160 completion tokens, including eight `stop` responses. All 3,456 retained server-log records show no context truncation. A larger output budget alone is not an established fix.

This pilot asks whether a compact schema-first instruction improves syntactic feedback success and whether any improvement remains grounded in the actual response. It separates those two outcomes. It does not establish that Sprocket feedback improves later learning; a future real/omitted/shuffled-feedback control would be needed for that causal claim.

## Frozen cases and workload

The fixed selection spans three existing lineages and all sibling ecologies, with one common T case and one member P probe per selected member:

| Member | Existing common case | Existing member probe |
|---|---|---|
| V-G01-A | T01 | P01 |
| V-G06-B | T12 | P04 |
| V-G12-C | T24 | P08 |

These are previously used diagnostic/training-related inputs, not new held-out evaluation. Selection is frozen before new pilot outputs. The plan does not inspect the Stage E final-suite recipe, rendered suite, responses, or scores.

Each case uses its **three exact archived worker seeds** in both arms. The original seed formula is independently recomputed, and every original chat request must match the archived request exactly before observational capture fields are added. There are **six cases × three seeds × two arms = 36 local generations**, capped at 160 tokens each: at most **5,760 generated tokens**. Arm order alternates across the 18 paired comparisons, giving nine original-first and nine compact-first pairs.

The `original` arm preserves the original system and user messages. The `compact_schema_first` arm changes only the system instruction: a schema-valid example comes first, instructions are shorter, and one keep/mutate item is requested. The example uses `"genericness":"medium"` and a separate instruction to choose low, medium or high, avoiding the original schema-invalid `"low|medium|high"` placeholder. The ancestral temperament and sibling pressure remain present. The situation and villager response stay byte-identical. This tests the combined prompt revision; it does not isolate schema order, example validity and brevity as separate causal factors.

Both arms retain temperature **1.05**, top-p **0.97**, all original seeds, native sampler defaults, and native cache behavior. There is no grammar, JSON response-format constraint, sampler replacement, or model update. Native cache reuse can affect floating-point execution, so this is a fresh paired pilot, not a cross-host byte-determinism claim. The original arm's equality with historical text, token count, and finish reason is explicitly measured.

## Model, template, and runtime qualification

The same frozen model/runtime are required:

- `kandivault/sprocket-500m` revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`.
- `sprocket-500m-chat-q4_k_m.gguf`, SHA-256 `edc2ac323a2513fa6f8e3e54adf120243de0f7a6b8cc11b349b3e6ad61494a09`.
- llama.cpp `b11852`, commit `7894e8b92481a8bb8ffe94f73b656ffe0dd01e10`.
- Original runtime archive SHA-256 `6ec84fd5ddf5bee393d27384b9b6acb19a35cbac71b0d22b4d73fc727363fcca`.
- Four server threads and requested context 4,096; the effective native/model context must be 2,048, matching the historical runtime's cap.

A bounded range read of the pinned model URL located the complete GGUF metadata section ending at byte 1,135,086. The metadata contains a 443-byte Jinja role template, SHA-256 `995184ace6367465228c6ae16dcc5cfbb36e13147ea6bdba956f6b0a077204fa`. It declares system/user/assistant/end token IDs 2/3/4/5, BOS 0 and padding 1. The range inspection does not replace verification of the complete model file; full-file SHA verification is required before execution.

The pinned runtime source enables Jinja by default. Consequently, a missing-template explanation cannot be assumed from the old malformed outputs. The actual runtime must still prove that it uses the expected template and preserves the system/user boundaries.

Before the first stochastic generation, all 36 requests are rendered through `/apply-template`, tokenized with the same special-token settings as the native chat route, and independently checked through its input-token counting endpoint. Rendering must equal the simple system/end/user/end/assistant sequence encoded by the frozen GGUF template. Role IDs, ordering, input counts, and the input-plus-160-token context budget must agree. A mismatch stops the pilot before generation while retaining the audit evidence.

Each generated response uses `verbose=true` and `return_tokens=true`. In the pinned runtime these are observational capture options. The receipt contains the actual native prompt, generated token IDs, effective sampling settings, stop type and full raw response. The actual prompt is retokenized and required to match the preflight token IDs. No detokenized output is substituted for the original raw response.

Primary implementation sources:

- [Pinned server documentation](https://github.com/ggml-org/llama.cpp/blob/7894e8b92481a8bb8ffe94f73b656ffe0dd01e10/tools/server/README.md).
- [Native chat rendering, tokenization and final-response path](https://github.com/ggml-org/llama.cpp/blob/7894e8b92481a8bb8ffe94f73b656ffe0dd01e10/tools/server/server-context.cpp).
- [Observational request options and sampling fields](https://github.com/ggml-org/llama.cpp/blob/7894e8b92481a8bb8ffe94f73b656ffe0dd01e10/tools/server/server-schema.cpp).
- [Verbose native receipt serialization](https://github.com/ggml-org/llama.cpp/blob/7894e8b92481a8bb8ffe94f73b656ffe0dd01e10/tools/server/server-task.cpp).

## Outcomes and grounding review

Report both the historical parser's schema validity and strict whole-string JSON-object validity. Preserve invalid, empty, repetitive, and truncated outputs. Report finish reasons, generated-token counts, instruction echo, and repeated-trigram fractions.

Lexical overlap with the situation/response is a descriptive proxy only. It is not a grounding score. `blinded-grounding-review.json` contains all 36 outputs in opaque order and a fixed 0–2 rubric for response grounding, scenario grounding, and corrective usefulness, plus an unsupported-claims flag, exact source quote, and explanation. A separate private key maps the opaque items to arms and seeds. Ratings remain null until an explicitly identified reviewer completes them. Any later cloud review must have its own bounded request contract and full raw custody; this local script makes zero cloud requests.

Analysis should compare each of the 18 paired cases/seeds, retain disagreements and failures, and avoid a significance or population-generalization claim from this small convenience sample. Syntax improvement without grounding improvement is not a successful feedback repair.

## Execution and custody

Prepare a plan without starting a server or model:

```bash
python3 scripts/village_001f_sprocket_diagnostic.py --out /tmp/sprocket-plan
```

After Stage E has completed, execute with the existing pinned bootstrap:

```bash
bash scripts/bootstrap.sh sprocket
python3 scripts/village_001f_sprocket_diagnostic.py \
  --execute --out "$RUNNER_TEMP/sprocket-diagnostic"
```

Reconstruct a completed run without model files, a server, or network calls:

```bash
python3 scripts/village_001f_sprocket_diagnostic.py \
  --resume "$RUNNER_TEMP/sprocket-diagnostic" \
  --out "$RUNNER_TEMP/sprocket-diagnostic-replay" --offline
```

Resume copies into an empty, disjoint directory, verifies the exact plan/source bytes, and never rewrites prior evidence. Completed responses are reconstructed from archived raw bytes. Native template, tokenization and counting exchanges are reconstructed against their exact requests, including the actual prompt retokenization request. A request without a durable generation response is ambiguous and is never automatically reissued; this preserves the 36-generation ceiling. Completed resumes automatically take the offline reconstruction path even if `--execute` was supplied.

The output includes the fixed plan; verified runtime/GGUF/template metadata; raw local HTTP requests/responses; native rendered prompts and token IDs; all 36 immutable rows; server logs; summary; blinded review packet and private key; and a complete evidence-hash inventory. Preserve the complete artifact even when preflight or generation fails. Weights stay outside ordinary Git.
