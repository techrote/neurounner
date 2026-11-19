
# Stage E provider completion amendment

This is an explicit operational amendment to complete the frozen Stage E evaluation. The scientific scoring rubrics, catalogs, opaque identities, parsers, models, held-out scenarios, learner responses, individual and population gates remain unchanged. No model is trained and no local response is regenerated.

## Why the original queue stopped

Original judging run [44439359038](https://github.com/techrote/neurounner/actions/runs/44439359038), source `7308271015bf8cc0ff98e0d35ad68314e65db721`, preserved 23 completed ecology judgments and one partial ecology audit. In E-P-G05, the original Super audit supplied valid booleans but an invalid reason; its next repair exhausted its 2,048-token allowance on internal reasoning and returned no usable reason. The first two tournament scenarios then exhausted their 42-call lane limits after repeated valid JSON objects with empty sibling-classification arrays. These were six-token, normal-stop provider outputs, not parser losses or output truncation. The fixed example ID and lack of an explicit required item count are a plausible formatting ambiguity; this record does not claim a proven causal explanation for the provider's behavior.

To retain the original campaign budget, [stop run 44442086494](https://github.com/techrote/neurounner/actions/runs/44442086494) performed normal cancellation of that exact run. The active F03 lane retained its first normalization response and second interrupted request. All 27 produced artifacts remain intact. F04–F24 never acquired a runner and have no steps or artifacts. Their zero-call status is established from the frozen job records, never inferred from a missing archive alone.

## Frozen custody and allocation

`original-run.json`, `original-jobs.json`, and `original-artifacts.json` preserve the terminal GitHub records. `manifest.json` binds their hashes, every original ZIP digest and extracted file, the complete generation archive, and the recovery wrapper and frozen evaluator sources. The 21 unstarted lanes receive only an offline-derived provider identity; no historical ZIP or provider response is invented.

| Lane group | Original logical requests | Effective whole-campaign allocation |
|---|---:|---:|
| 23 completed ecology lanes | 55 | 55 |
| E-P-G05 | 3 | 6 |
| F01 | 42 | 60 |
| F02 | 42 | 60 |
| F03 | 2 | 60 |
| F04–F24 | 0 | 840, or 40 per lane |
| **Total** | **144** | **1,081** |

The original global ceiling of 1,104 logical requests still applies; 23 slots remain unallocated. Allocation depends on execution status, not scores. The wrapper retains the original three transport launches per logical request and six HTTP attempts per launch. F03's interrupted request remains counted; its next transport launch uses the identical original payload. No new schema-repair rounds are added: the original missing-only 3/2/1 schedule remains fixed.

## Two disclosed request amendments

All previously archived requests replay byte-for-byte. For a new logical request absent from the original request inventory:

- A Super missing-only repair receives `max_tokens=8192`. Its original messages, model, temperature, top-p and repair round are unchanged. Internal reasoning is never used as judgment evidence.
- A tournament sibling missing-only repair appends a mechanical JSON-format instruction: the exact opaque IDs and row count to return, and a clarification that the schema example ID is illustrative. The original prompt text, source items, sibling rubric, confidence scale, model and sampling settings remain intact. Accepted classifications are never requested again.

The amendment is a separate recovery identity. It does not rewrite the original provider identity or present amended requests as original ones. The policy is determined from the frozen original inventory, so the new payloads reconstruct exactly during later offline replay.

## Continuation and publication

The operator downloads and independently verifies the original archives, then reconstructs the complete manifest before any new provider call. It requires a new owned-branch execution request and rejects an Actions rerun. A later request must identify the latest prior recovery that entered the provider step; a fresh null resume cannot reset spending. Before calls, the complete inherited 48-lane forest and any earlier recovery ZIP/receipt chain are copied into the output bundle. Partial copies cannot discard inherited provider history. If a previously started tournament remains incomplete, untouched tournament lanes do not begin spending.

Every final lane must replay without a key. `accounting.json` records actual logical and transport counts and confirms first-valid/original-byte preservation. The unchanged strict assembler then computes all 36 individual verdicts, both kinds of confusion matrices, the contemporaneous BASE comparison, and all ecology results. Full raw evidence and the original ZIPs enter the final release TAR; recovery metadata and accounting enter Git. Release notes include this amendment alongside the unchanged scientific report. A completed evaluation may still fail the usable-village gates; such failures are retained and reported.
