# VILLAGE-001D recovery — 18 November 2026

## Outcome

The blocked VILLAGE-001D campaign is recovered on [Draft/HOLD PR #254](https://github.com/techrote/neurounner/pull/254).
All 36 original preference-trained villagers and all 36 before/after comparisons
are now durably archived. All 36 pass the existing D regression gates. The
independent final tournament and ecology stage has not run; `usable_village`
remains unqualified.

The recovery used **three new NVIDIA critic requests**, **zero new optimizer
steps**, and **zero new local model generations**. The 35 completed comparisons
were reconstructed from their exact archived requests/responses and preserved
byte-for-byte. All original trained adapter identities remain unchanged.

- [Successful recovery run 44417236485](https://github.com/techrote/neurounner/actions/runs/44417236485).
- Recovery execution source: `a8d817e4c7e82088952b8ac72d4867726c9450b8`.
- Durable evidence commit: `5fea627c5f16a1dffe17d12dbb9230522ab7be15`.
- [Isolated recovery prerelease](https://github.com/techrote/neurounner/releases/tag/village-001d-recovery-40613393744-20261118).
- Parent epic: [#9](https://github.com/techrote/neurounner/issues/9); D:
  [#13](https://github.com/techrote/neurounner/issues/13); final E:
  [#14](https://github.com/techrote/neurounner/issues/14).

No merge, Ready transition, auto-merge, main-branch evidence publication, canonical
release replacement, or issue closure was performed.

## What was actually blocked

The last issue comment described D training as queued. The original
[run 40613393744](https://github.com/techrote/neurounner/actions/runs/40613393744)
subsequently completed all 36 training jobs on 28 October. Its 146 jobs comprise
one successful plan, four 36-member stages, and a skipped assembler. Only the
V-G12-B comparison failed; all other feedback, pair, training and comparison
jobs succeeded.

Original source: `388ad349d48102f6266c0c0c837efec0275e38c9`. The accepted
main at continuation start was `40543403f05de7ecacdca84ad7d651e5c5997cea`.
Village inputs, SFT evidence, training recipe and production scripts had not
changed between those sources.

V-G12-B's T10 rationale contained unescaped quotation marks around “other shoe
to drop”. This made the six-review JSON batch invalid. Three historical repairs
repeated the defect, and the original code discarded the whole batch and stopped
before T13–T24.

The corrected parser admits only consecutive, independently valid array elements
before the first malformed element. It does not repair strings, invent scores,
scan for later JSON objects, or select a more favorable later judgment. Nine
original reviews, T01–T09, were safely reused. T10–T24 were completed with three
requests covering 3 + 6 + 6 cases. Original T11/T12 bytes remain archived even
though the conservative parser did not admit them across broken T10.

The new reviews were collected on 18 November; historical judgments were collected
on 28 October. The requested critic model and decoding contract are retained, but
the remote provider backend is unpinned. This is a disclosed recovery cohort,
not a claim that all judgments came from one immutable backend snapshot.

## Engineering changes

Comparison recovery now binds member identity, member contract, exact SFT and
preference result bytes, source cases, configuration, judge model and archived
request payloads. It verifies the archived content against the provider response,
retains the first valid judgment, preserves previous request/response filenames,
and narrows missing-only repairs through batches of 3, 2 and 1. A complete
comparison can be replayed with `--offline` and no provider key.

The source-artifact importer verifies the exact original run, repository, source
commit and all 146 job outcomes. It requires the complete 144-artifact inventory,
checks each original ZIP's byte count and SHA-256, and rejects unsafe ZIP paths,
links and conflicting entries. Previously preserved originals remain verifiable
after Actions retention expires. Transient GET failures have a four-attempt
budget; permanent authorization, missing-artifact and expiry errors fail closed.
GitHub authorization is not forwarded to artifact storage.

Assembly now binds every recovered lane to the current SFT release/result and
exact source responses. Mutually consistent artifacts from an obsolete parent
cannot be silently repromoted. Complete comparison provider directories,
including malformed responses, repairs and recovery receipts, are archived.
The feedback denominator is corrected from 72 to the actual 96 records/member.
An explicit release-tag override supports an isolated recovery prerelease without
changing the frozen training configuration.

The recovery workflow checks the owned branch head and release absence before
provider work, and checks the branch head again before a non-forced branch push.
It never pushes main and never invokes the final E workflow.

## Measured results and their limits

| Measure | Before | After | Interpretation |
|---|---:|---:|---|
| Mean preference accuracy | 34.77% | 99.68% | Evaluation on the training preference pairs |
| Mean preference margin | -0.7802 | +3.1573 nats/token | Same training-pair diagnostic |
| Mean task relevance | 0.9919/4 | 1.0035/4 | Existing T-suite critic diagnostic |
| Mean behavior fit | 0.8275/4 | 0.8958/4 | Existing T-suite critic diagnostic |
| Mean genericness | 0.9086/4 | 0.6887/4 | Lower means less generic |
| Mean sibling prediction accuracy | 31.37% | 38.89% | Diagnostic classification, not final blind tournament |

Every member passes D's existing relevance-regression and raw-style-only
regression checks: **36 qualified, 0 regressed under those gates**. These are
relative regression checks, not absolute quality admission. Individual after
task-relevance means range from **0.75 to 1.25/4**; **0/36 reach 2.5/4** on this
diagnostic. E's 2.5 floor applies to its own fresh final suite, which remains
unmeasured.

The T prompts were used in preference maturation, so neither these T-suite
scores nor the preference-pair accuracy demonstrate final generalization.
The recovery does not relabel them as final held-out results.

Campaign counts remain exact:

| Evidence | Count |
|---|---:|
| Members / completed preference lanes | 36 / 36 |
| Original optimizer steps, across all members | 4,320 |
| New optimizer steps during recovery | 0 |
| Raw Sprocket feedback records | 3,456 |
| Preference pairs | 2,160 |
| Common-scenario / member-probe pairs | 1,296 / 864 |
| Terminal critic-pass / adjudicator-accepted pairs | 2,154 / 6 |
| Original comparisons reused unchanged | 35 |
| New critic requests | 3 |

## Structured Sprocket feedback failed

All **3,456** original feedback records have `valid=false` and `parsed=null`.
Independent reparsing confirms **zero valid structured feedback objects**.
This is a substantive limitation of the historical campaign, not a newly
introduced recovery failure.

- 2,713 outputs ended with `finish_reason=length`; 743 with `finish_reason=stop`.
  A total of 2,721 outputs contain exactly 160 completion tokens, including eight
  stop-terminated responses.
- 147 outputs were empty.
- Only two outputs contain any opening brace; none begins with one.
- Recorded server completions report no context truncation. The maximum recorded
  occupancy is 691 tokens within a 2,048-token slot.

Increasing the token budget alone is not an established fix, because all 743
stop-terminated outputs also failed. The intended pinned Sprocket Chat Q4 model
is identified in the configuration/logs. The archive does not retain the complete
rendered native prompt/token sequence, so it does not prove a template mismatch,
quantization failure or a particular sampling cause.

The malformed text was **not discarded or silently replaced**. An independent
audit decoded 117 compiler/repair requests, 2,185 case inclusions and 6,555 feedback
inclusions. Together these cover all 3,456 distinct source feedback IDs. Embedded
raw text, worker, validity and parsed fields match the archive exactly:
**zero mismatches**. All 2,160 pairs and 36 pair manifests retain the correct
feedback hashes and scenario-specific source IDs.

The defensible description is **cloud-curated preference continuation informed
by raw Sprocket text**. This is not successful structured Sprocket criticism, and
there is no matched no-feedback/shuffled-feedback control establishing a causal
Sprocket contribution. No retrospective gate was invented and no historical
feedback was relabelled.

A separate small diagnostic could inspect and freeze the native prompt rendering,
then compare the original and compact schema-first prompts on six existing
T/P cases and three original seeds (36 local generations, no training). That
diagnostic was not executed during this recovery.

## Permanent custody and readback

The prerelease contains **37 assets, 661,668,069 bytes total**:

- 36 original trained adapter archives, individually identified by size and SHA-256.
- `village-001d-original-artifacts-40613393744.tar`, containing all 144 original
  Actions ZIPs plus exact source metadata and per-file custody hashes.

Actions artifacts are staging only. The raw feedback, pairs, provider records,
training results, comparisons and provenance are also retained in the branch's
durable dataset/evidence trees. No adapter weight files were added to normal Git
history.

Independent readback after publication verified:

- all 37 release names, uploaded states, byte counts and SHA-256 digests;
- **844** preference-evidence file hashes;
- **3,059** dataset file hashes;
- all **36** terminal comparisons reproduced byte-for-byte from archived provider
  evidence with `--offline`, including repaired G12-B, with zero new calls.
- an actual public download of the V-G12-B adapter archive matched its retained
  size/hash and contained the trained safetensors, adapter configuration and README;
- rebuilding the complete original-artifact TAR from the independently downloaded
  ZIPs and committed metadata reproduced the published **334,899,200-byte** bundle
  exactly: SHA-256 `00c6139fc1227dd6f58bc7d2e441cf19033727af6f706174ea468572c1d5dcbb`.

Machine-readable records:

- [Recovery receipt](../experiments/village-001/recovery-20261118/evidence/recovery-result.json).
- [Release inventory](../experiments/village-001/recovery-20261118/evidence/release-assets.json).
- [Original source custody](../experiments/village-001/recovery-20261118/evidence/manifest.json).
- [Published readback](../experiments/village-001/recovery-20261118/review/published-readback.json).
- [Original-bundle reconstruction](../experiments/village-001/recovery-20261118/review/reconstructed-custody-readback.json).
- [Structured-feedback audit](../experiments/village-001/recovery-20261118/review/structured-feedback-audit.json).
- [Stage E readiness audit](../experiments/village-001/recovery-20261118/review/stage-e-readiness-audit.json).

The audit source paths are relative to the recovered original-artifact working
tree described by the custody manifest. Its original ZIPs can be extracted with
the checked recovery helper; they are not expired Actions-only references.

## Validation and execution history

**42 offline regression tests pass**, with optimized-Python checks also passing.
The existing production D compilation and curation checks pass. Independent
review reproduced and corrected an overly permissive JSON-salvage boundary
before live judging.

The exact-source successful run is
[44417236485](https://github.com/techrote/neurounner/actions/runs/44417236485).
Earlier bounded failures are retained in workflow history:

- [44416656314](https://github.com/techrote/neurounner/actions/runs/44416656314):
  runner context used at an unavailable workflow level; rejected before any job.
- [44416908374](https://github.com/techrote/neurounner/actions/runs/44416908374):
  transient GitHub HTTP 502 while reading original job metadata; no provider call.

The fixes moved runner-path initialization into a step and added bounded GET
retries. Neither failure retrained or rejudged the population.

## Remaining work for issue #9

Stage E remains open. It must freeze the exact D summary/release bytes, generate
its separate final suite once, retain that suite unchanged across retries, and
measure the independent final tournament, BASE comparison, ecology and population
gates. No generated final suite was accessed or produced in this continuation.

The current E workflow is not suitable for direct execution from this recovery
branch: it eventually pushes main, can clobber the canonical final release, and
one ecology download hardcodes the canonical D tag. Its provider stages also
delete output and lack resumable replay. An isolated continuation should fix those
publication and recovery paths before committing the full campaign.

This is substantial work: the inspected frozen design contains 111 jobs,
1,068 local model generations and 468 baseline provider requests, with additional
bounded audits, schema repairs and transport retries. The independent readiness
audit records the exact bounds. The low historical T-suite relevance is a quality
risk to report, not permission to weaken E's gates or declare E failed without
running it.

PR #254 remains **Draft/HOLD**. Issues #9, #13 and #14 remain open pending their
appropriate evidence/acceptance decisions.

## Stage D acceptance and integration — 19 November 2026

A separate acceptance review checked all seven requirements in issue #13 against the complete recovered archive. All seven pass under the frozen D contract. The machine-readable review is `experiments/village-001/recovery-20261118/review/stage-d-acceptance.json`. Formal closure additionally requires independently reading the merged evidence back from `main`, as specified in the original issue #13 handoff. That readback and the merge commit are recorded in the issue when complete.

The review reconfirmed 96 feedback records and 60 validated preference pairs per member, all 36 terminal 120-step training records, exact parent/feedback/provider lineage, all 36 comparison gates, all 36 retrievable adapter assets, 844 evidence hashes and 3,059 dataset hashes. All 42 recovery tests pass normally and under optimized Python.

Passing the D gates does not mean every behavior improved: 10 members have a negative behavior-fit delta. None combines a positive raw lineage-fit delta with a negative behavior-fit delta, so none qualifies on a raw-only improvement. Their individual declines remain explicit in the new acceptance record. The structured-feedback failure and weak absolute relevance remain unchanged scientific findings.

Before integration, the historical Stage E workflow's automatic and manual live triggers were removed; its pull-request contract checks remain. This prevents merging D evidence from launching the obsolete workflow that publishes to main and the canonical E release. The requested Stage E continuation will use the separately reviewed isolated, resumable execution path. No D adapter, summary, release-manifest byte, training recipe or gate was changed by this acceptance step.
