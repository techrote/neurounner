# VILLAGE-001F: matched mask and feedback diagnostics

Parent [#9](https://github.com/techrote/neurounner/issues/9); execution [#256](https://github.com/techrote/neurounner/issues/256). This protocol is prepared before the diagnostic runs. The operator requires completed Stage E [#14](https://github.com/techrote/neurounner/issues/14), its accepted main commit and exact summary hash before model work begins. Execution results and receipts are separate from these frozen plans.

## Accepted E context and first launch

Stage E is accepted on main at 498cbfb8caed6e962ee4dd7af951aa34281352e6 and issue #14 is closed completed. Its [fresh-main readback](../qualification-20261119/review/main-readback.json) returns closure_ready true and binds summary SHA-256 d49cfb48427718dbc9588d3de34a819f9f4018f4e501431ef070b3e3db4b2479. E's usable-village gate fails: lineage identification passes, conditioned sibling recognition and the required member count fail, while all 36 members meet the frozen raw relevance/genericness minima. D/T and E/F quality means use different cohorts and judging contexts; their numerical difference is not a training improvement. See the [complete interpretation](../qualification-20261119/review/COMPLETION.md).

The first [execution request](request.json) binds that accepted commit and exact summary, with phase generate and revision 1. It retains the plan frozen before the final E outcome: no panel, training data, optimizer setting, Q prompt, judging criterion or budget changes follow from the result. F measures the mask mechanism and feedback prompting; it does not presume that either explains E's recognition failures. Actual runtime results are recorded separately when the input and final publication phases complete. The [independent acceptance tool and source receipt](review/ACCEPTANCE.md) are retained with this launch.

## Questions and controlled comparisons

| Question | Comparison | Fixed workload |
|---|---|---:|
| Does the historical attention-mask construction materially affect preference training? | Historical token-ID mask versus sequence-length mask, from identical C parents and original D pairs | Six members × two arms × 120 steps = 1,440 optimizer steps |
| Does either training arm improve fresh task behavior? | Paired corrected/reproduced D, with BASE, C and original D context | 12 new Q cases; 300 native responses |
| Can a compact feedback prompt make Sprocket's critiques useful? | Historical prompt versus compact schema-first revision, with matched cases and seeds | 36 native local generations |

The pinned [SmolLM2 tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer_config.json) assigns the same ID to padding and genuine end-of-turn tokens. The original D trainer derives attention from token-ID inequality. The matched pilot changes only attention-mask construction; it does not assume that this defect explains all historical quality failures. A 720-sequence native audit and 288 fixed-weight pair forward batches characterize the mechanism. Canonical tensor fingerprints distinguish actual historical reproduction from archive/configuration differences.

Attention masking is the only planned algorithmic intervention. A prospective amendment made before any F execution places both arms in one member job, using two fresh CLI processes from the same verified C parent and original member seed. The six jobs retain twelve 120-step lanes. Historical runs first for V-G01-A, V-G05-C and V-G09-B; corrected runs first for V-G03-B, V-G07-A and V-G11-C. This 3+3 order is balanced across sibling types. The [worker method](WORKER_MASK_PILOT.md#prospective-paired-runner-amendment) records the observed original CPU variation and actual 4m 22s–8m 18s job durations motivating the amendment; the paired job has a 90-minute cap.

Immutable per-invocation records bind the job nonce, run/attempt, runner/boot/host, process, clocks, starting and retained steps, and all inherited records to native trace prefixes. Fresh pairing requires actual shared execution and the declared sequence, rather than matching result labels alone. Completed copies add zero optimizer steps; resumed, interrupted or mixed-host histories remain visible, and any checkpoint/provenance write gap is explicitly unattributed. The report also discloses each original, reproduced and corrected D result's hash-bound CPU model, CPU count, platform, full recorded package metadata and baseline reproduction. Matching metadata does not guarantee numerical equivalence. Observed execution or reproduction differences qualify attribution without excluding members, changing quality criteria or authorizing additional training.

The worker panel is V-G01-A, V-G03-B, V-G05-C, V-G07-A, V-G09-B and V-G11-C: six odd-numbered lineages, cyclic A/B/C, two members per sibling. Selection does not depend on E scores. The new Q reserve has zero exact collisions with 3,407 prior inputs. Q and E-final cases never enter training. The six member means are the replication units; the full 72 paired case differences and every regression are retained. The predeclared working effect threshold is 0.5/4 relevance, at least four positive member means, and no member relevance loss above 0.5. A pilot does not automatically authorize or establish a useful 36-member population.

The [original C baseline audit](C_BASELINE_AUDIT.md) separates confirmed curriculum and validation findings from causal hypotheses and records a proposed seen-example check outside F's workload.

Sprocket retains its original pinned Chat GGUF, native template, runtime, source examples, three seeds and sampling settings. The compact arm is a combined prompt revision, so the experiment does not isolate schema position alone. JSON validity, grounding, usefulness and unsupported claims are reported separately. A syntactically valid critique is not by itself evidence of a training benefit.

The [pinned runtime API source receipt](sprocket-runtime-api-source-receipt.json) verifies six upstream files and the `b11852` tag at commit `7894e8b92481a8bb8ffe94f73b656ffe0dd01e10`. The requested `verbose` and `return_tokens` fields expose the native prompt and token IDs without a `return_prompt` flag; preflight tokenization matches generation, and completion counts include any sampled EOG token. This source review did not execute a model.

Detailed plans: [worker mask pilot](WORKER_MASK_PILOT.md), [Sprocket diagnostic](SPROCKET_DIAGNOSTIC_PLAN.md), [worker plan](worker-plan.json), and [judge contract](judge-config.json).

## Execution and evidence

The owned branch is `work/village-001f-causal-diagnostics-20261119`. The workflow has two explicit phases. Generation runs six paired member jobs containing the 12 matched training lanes, token audit, 25 inference cohorts, 18 fixed-weight score lanes and Sprocket experiment. It verifies native identities and the complete invocation histories before freezing all inputs. The 12 new adapter archives are published outside Git under `village-001f-pilot-inputs-20261119`; source receipts, invocation provenance and textual/token evidence are committed.

Judging begins only from those frozen inputs. Its 50 baseline requests comprise 18 paired worker batches, 26 context batches and six Sprocket batches. The hard logical ceiling is 114 including all bounded missing-only repairs. Each prompt enumerates actual opaque IDs and the exact required row count; examples use an actual requested ID. Empty-text items are retained and graded under the same rubric. Models, rubrics, batch sizes, output-token limits and first-valid preservation remain fixed.

Actions reruns cannot reset budgets. New phase revisions must identify the latest spending run and retain every previously attempted lane. Source ZIPs and their receipts survive recovery; model/checkpoint binaries remain outside Git. The final release `village-001f-mask-sprocket-20261119` contains the report, paired effects, Sprocket grounding results and complete raw evidence. Independent source-bound replay and final readback precede acceptance.

No historical D or E evidence is replaced. Findings, including null effects, failed reproduction, invalid critiques and harmful member changes, determine the next research decision.
