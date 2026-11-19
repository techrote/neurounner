# Stage F Sprocket feedback interpretation

**The compact schema-first prompt did not improve structured feedback in this fixed pilot.** Both arms produced **0/18** responses passing either frozen structure measure, and neither arm produced any whole-string JSON value. The archived critic rated **9/18 original-prompt outputs** and **8/18 compact-prompt outputs** as grounded under the prespecified conjunction. Mean corrective usefulness was **0.833/2** for the original prompt and **0.667/2** for the compact prompt. Both structure-required useful-feedback counts are zero. These findings support the frozen classification **no_grounded_actionable_count_gain**; they do not establish an improvement to worker learning. See the [independent numerical recomputation](sprocket-analysis.json) and [native archived result](../results/sprocket-grounding.json).

## Design and accepted evidence

The pilot uses six pre-existing situation/response cases from three members: V-G01-A/T01 and P01, V-G06-B/T12 and P04, and V-G12-C/T24 and P08. Each case has three original Sprocket seeds, with both prompt arms evaluated at each seed. This yields **18 matched case/seed pairs and 36 outputs**. The order alternates between arms, with nine original-first and nine compact-first pairs. The cases and prompts were frozen before the new outputs. Six cases are the task units; the three seeds are repeated observations within each case. See the [frozen configuration](../sprocket-config.json) and [native plan](../inputs/sprocket/plan.json).

Only the system prompt changes between arms. The fixed GGUF, native template, runtime, temperature 1.05, top-p 0.97, and 160-token allowance remain the same. No grammar or response-format constraint was added. The compact prompt requests one keep item, one mutate item, a failure guess, and a genericness category. The supplied villager responses are existing C-stage outputs; these new feedback outputs do not train a worker. The pilot is separate from the worker mask intervention and does not use the Stage E final suite or the Stage F Q reserve.

Both input and final evidence have passed independent acceptance. The input release is at commit **783a73ca7c8e97c3e03ef4e56fb5a5b125d58785**; the final release is at **7ab26c67ad51b5273513e2bb7eeccbc2777d87ae**. The complete final provider archive, metrics, and report reproduce byte-for-byte offline. All **658 published Sprocket files** match the original completed job artifact exactly. Public provenance is retained in the [Sprocket custody receipt](sprocket-custody.json), [input acceptance](input-acceptance.json), [publication comparison](sprocket-input-readback.json), and [final acceptance](final-acceptance.json). Temporary download credentials are not included in this review.

## Structure and generation behavior

| Measurement | Original prompt | Compact schema-first |
|---|---:|---:|
| Responses | 18 | 18 |
| Historical field-validator passes | 0 | 0 |
| Frozen whole-string JSON/schema passes | 0 | 0 |
| Any whole-string JSON value | 0 | 0 |
| Nonempty responses | 18 | 17 |
| Reached the 160-token output limit | 14 | 15 |
| Stopped before that limit | 4 | 3 |
| Mean completion tokens | 151.444 | 147.333 |
| Exact historical raw-text matches | 0/18 | Not a reproduction arm |

Every case has zero structural passes in both arms. None of the 36 outputs begins with an opening brace after leading whitespace. **There are zero whole-string JSON objects on which to evaluate keep/mutate cardinality or extra-field compliance.** Those supplemental checks therefore have denominator zero and are reported as not evaluable, rather than as a compliance percentage. The [per-row numerical record](sprocket-analysis.json) preserves the raw-source hashes and the unchanged native flags.

The name of the frozen **strict_json_schema_valid** flag requires precision: it means that the entire response parses as a JSON object and passes the historical field validator. That validator permits **zero to three nonempty strings** in each keep/mutate array and ignores extra fields. It does not enforce the compact prompt's exact one-item instruction. The supplemental shape counts do not redefine this flag. The implementation is in the [native Sprocket driver](../../../../scripts/village_001f_sprocket_diagnostic.py) and [historical feedback parser](../../../../scripts/village_001d_sprocket_feedback.py).

The 29 output-limit hits do not identify the cause of malformed feedback. All seven early-stop outputs also fail structure. One [compact fan-case output](../inputs/sprocket/rows/003-V-G01-A-T01-W2-compact_schema_first.json) immediately stops with empty content and one EOG token; another [compact records-case output](../inputs/sprocket/rows/030-V-G12-C-T24-W3-compact_schema_first.json) stops normally after 117 tokens of prose. A short output budget alone therefore does not account for every observed structural failure. This pilot does not isolate model capacity, instruction following, sampling, or prompt design as the cause.

The archived runtime records the expected model, runtime, and template hashes. Native prompt lengths are **358–511 tokens**, and the longest prompt plus its 160-token allowance is **671 tokens**, below the effective **2,048-token context**. All 36 actual prompt token sequences match native preflight, and the frozen validator rejects context truncation or changed seeds/sampling settings. This is an offline check of the recorded native exchanges and runtime identity receipts, not a fresh execution of the model. See [runtime-001.json](../inputs/sprocket/runtime-001.json) and the [recomputed summary](sprocket-analysis.json).

## Grounding and usefulness of the prose

The critic evaluated malformed prose as well as empty output; syntax failure did not remove any item from semantic review. The frozen grounded conjunction requires response grounding at least 1/2, scenario grounding at least 1/2, and no flagged unsupported claim. Thus grounded includes weakly grounded ratings and does not mean a fully verified response. The strict-grounded-actionable conjunction additionally requires valid whole-string structure and corrective usefulness at least 1/2. See the [frozen judging rubric](../../../../scripts/village_001f_judge.py), [blind inputs](../results/blinded-inputs.json), and [archived ratings](../results/sprocket-grounding.json).

| Critic-derived measurement | Original prompt | Compact schema-first |
|---|---:|---:|
| Grounded conjunction | 9/18 | 8/18 |
| Unsupported claims flagged | 9/18 | 9/18 |
| Mean response grounding | 22/18 = 1.222/2 | 18/18 = 1.000/2 |
| Mean scenario grounding | 31/18 = 1.722/2 | 27/18 = 1.500/2 |
| Mean corrective usefulness | 15/18 = 0.833/2 | 12/18 = 0.667/2 |
| Strict structure and grounded | 0/18 | 0/18 |
| Strict structure, grounded, and actionable | 0/18 | 0/18 |

The compact-minus-original differences are **−2/9** on both grounding means and **−1/6** on corrective usefulness. They are descriptive differences from this small archived cohort. Case-level behavior varies:

| Case | Grounded, original → compact | Mean usefulness, original → compact |
|---|---:|---:|
| V-G01-A/T01, clicking fan | 3/3 → 1/3 | 1.667 → 0.000 |
| V-G01-A/P01, washer choice | 1/3 → 1/3 | 0.333 → 0.000 |
| V-G06-B/T12, moved object | 1/3 → 2/3 | 1.000 → 1.333 |
| V-G06-B/P04, charger spark | 1/3 → 1/3 | 1.333 → 1.000 |
| V-G12-C/T24, conflicting records | 1/3 → 1/3 | 0.333 → 1.000 |
| V-G12-C/P08, remembered marble | 2/3 → 2/3 | 0.333 → 0.667 |

Several examples clarify why these measurements must remain separate:

- The [compact washer-case output for worker 1](../inputs/sprocket/rows/007-V-G01-A-P01-W1-compact_schema_first.json) was rated 2/2 for both grounding dimensions but 0/2 for usefulness: the critic described it as a restructuring of the choices without a concrete correction. Its review ID is **B-fc74c7a0f9274e**.
- The [compact moved-object output for worker 1](../inputs/sprocket/rows/014-V-G06-B-T12-W1-compact_schema_first.json) was rated 2/2 on all three dimensions and had no unsupported-claim flag. It still fails both structure measures. Its review ID is **B-fbbc622230d7d9**.
- The [compact washer-case output for worker 2](../inputs/sprocket/rows/010-V-G01-A-P01-W2-compact_schema_first.json) restates the situation and response repeatedly, changing possession wording; the critic flagged unsupported claims and assigned usefulness 0/2. Its review ID is **B-82f67020550b6d**.
- The [original marble-case output for worker 3](../inputs/sprocket/rows/036-V-G12-C-P08-W3-original.json) includes the instruction “Return one JSON object only” inside prose. Repeating an instruction is neither JSON structure nor evidence of following it. Its review ID is **B-eb174309810af0**.

These are archived critic judgments, not independently established semantic ground truth. Exact source quotes are checked for the maximum response-grounding score, but that mechanical check cannot establish that every interpretation or correction is sound. The remote backend is not version-pinned. The frozen scores and first accepted ratings are retained without rescoring.

## Reproduction and limits

The fresh original-prompt arm matches **0/18 historical raw outputs**, **11/18 historical completion-token counts**, and **13/18 historical finish reasons**. Historical requests are reconstructed exactly, with observational verbose/token-return fields added. Native cache defaults and paired order are archived; byte equality across historical and fresh execution was measured, not promised. These observations do not identify the cause of non-reproduction. The six-case pilot is not a replay of the full historical 3,456-output feedback population. See the [native plan](../inputs/sprocket/plan.json), [native summary](../inputs/sprocket/summary.json), and [original D evidence](../../preference-evidence/summary.json).

The joint structural outcome is unequivocal for these observed outputs: both arms remain at zero. Grounding and usefulness of the malformed prose show mixed case-level behavior and no aggregate improvement for the compact prompt. Neither result establishes performance on other cases, other sampling settings, a different token allowance, or a grammar-constrained decoder. No feedback from this pilot is used for worker optimization, so it provides no causal estimate of a downstream learning benefit and supports no automatic replacement or rollout.

## Read-only recomputation

The [portable script](recompute_sprocket.py) uses only the standard library and the frozen repository's read-only native validators. It verifies accepted source/config hashes, the complete Sprocket input inventory, native wire reconstruction, the final blind mapping, exact source-bound ratings, and all derived conjunctions and counts. It neither loads model weights nor contacts a provider.

From the repository root, with the accepted review receipts present:

    python experiments/village-001/diagnostic-20261119/review/recompute_sprocket.py \
      --repo . --mode final --out NEW_SPROCKET_RECOMPUTATION.json

The output must be new and must not be inside native inputs or results. Inputs-only mode is also available; [sprocket-input-analysis.json](sprocket-input-analysis.json) preserves that deliberately semantic-pending readback. The [final numerical analysis](sprocket-analysis.json) binds the script, public custody, accepted reports, native sources, and final semantic sources by SHA-256. The preliminary connector and signed-download records remain outside the repository; the durable review contains public IDs, hashes, and relative evidence paths.
