
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

## Preserved checkpoint and allocation extension

Recovery run [44443194861](https://github.com/techrote/neurounner/actions/runs/44443194861), at source `abd44e3c21a9bcd63cef1f9310df34e05806f40b`, was normally stopped by the separately guarded [preservation run 44458502603](https://github.com/techrote/neurounner/actions/runs/44458502603). The original workflow had no provider-step deadline before its 180-minute job deadline. The stop preserved completed and partial work before that job deadline; it was not evidence of a provider outage. Its complete 5,208,218-byte recovery ZIP has SHA-256 `4c5f263a836e8a38fc7e8cdeb0143103072745eb8a628d569ebc2eda2727d70d`. All 3,782 extracted files and all original bytes were independently verified.

The saved execution was progressing continuously. It completed 529 new transport launches, each with one recorded HTTP 200 attempt and a normal stop, and began one further launch. No recorded HTTP error, rate-limit response, or retry explains its duration. The 65 completed normalization calls consumed approximately 100 minutes. Cancellation interrupted `F19-normalize-03` after approximately 30 seconds; the saved request remains counted and its next transport launch must use the same payload.

Offline reconstruction completed all 24 ecology lanes and 17 tournament lanes. Those 41 completed lanes used 630 logical requests. F15 had used its entire 40-call allocation, while F19 had used three calls and F20–F24 had not started. The complete preserved campaign therefore contains 673 logical requests and 674 transport launches. The allocation ceiling, rather than a score or provider failure, prevents F15 from continuing under the first recovery manifest.

The separate `allocation-extension.json` retains the original recovery manifest and all old execution identities unchanged. It binds the exact checkpoint ZIP and every selected source file, retires unused allocations from completed lanes, and assigns the remaining lanes explicit limits before further calls:

| Checkpoint lane group | Logical requests already archived | Extended whole-campaign allocation | Remaining logical allowance |
|---|---:|---:|---:|
| 41 completed lanes | 630 | 630 | 0 |
| F15 | 40 | 69 | 29 |
| F19 | 3 | 60 | 57 |
| F20–F24 | 0 | 300, or 60 per lane | 300 |
| **Total** | **673** | **1,059** | **386** |

The original 1,104-call ceiling remains unchanged, with 45 slots unallocated. Completed status and the remaining repair schedule determine this extension; quality scores do not. F15 already has all normalizations, full classifications and raw sibling judgments, plus 21 of 36 behavior sibling judgments. Its current three-row missing-only batch and final twelve-row batch can use at most 29 further requests under the existing 3/2/1 repair schedule. The 69-call limit covers that fixed schedule without adding rounds. The other six incomplete lanes receive the same 60-call provision previously assigned to started tournaments.

The extension adds its own identity and invocation receipts and preserves every earlier original/recovery file. Final accounting must verify the checkpoint boundary, new request deltas, all 48 offline reconstructions, and the global ceiling. An interrupted invocation without a terminal result or native receipt remains explicitly interrupted; its raw request files still count. A new request must resume the latest spending run and cannot reset to the original archives.

The continuation gives the provider step a 145-minute limit within the 180-minute job limit. This leaves 35 minutes minus setup time for the existing always-run uploads. No provider payload, model, accepted judgment, scientific rubric, parser, held-out response, or qualification gate is changed by this allocation extension.
