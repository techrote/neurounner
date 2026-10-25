# STORY-001 progress checkpoint

**Recorded:** 2026-10-25 21:56 UTC  
**Issue:** #43  
**PR:** #44  
**Branch:** `story-001-autofactory-kickoff`  
**Branch head at checkpoint:** `a9bc8214a4fe1385f281e05d6822b17765aa57be`

## Milestones

| Milestone | Status | Evidence |
|---|---|---|
| M-1 repository / archival contract inspection | complete | issue #43 kickoff plus repository RAG/dataset authority |
| M0 exact runner/runtime/model preflight | complete | repeated successful preflight jobs; current local-only contract uses pinned llama.cpp and SHA-pinned GGUFs |
| M1 Sprocket memory elicitation | complete | three independent pinned Sprocket responses preserved with exact requests/raw responses |
| M2 structural storyboard | complete | `storyboard.md` recovered from run 40228097488; SHA-256 `5e40dbb886d15d49ce619bfc21d252b3a8fb7d4e449283eb5184a7922d9c4dfa` |
| M3 draft | complete / qualified for length | recovered eight-section draft reduced by exact-sentence deletion from 3,417 to 3,288 words; recovered draft file SHA-256 `6406b47465381874d7e15056780bfe9d4609d126cd9434939bc97f0a532946d3`; prune record preserves before/after hashes and every applied/rejected sentence |
| M4 independent review | complete | `review.md` exists in run 40228097488 evidence; 730 words, review score 3/10 / REVISE; exact criticized draft hash recorded in review header |
| M5 revision/edit | in progress | current pipeline is testing LFM2.5-8B-A1B as edit planner with Qwen3.5-4B as prose renderer |
| M6 durable archive promotion / qualification | blocked on M5 | promotion remains fail-closed until final revised story passes length and archive-readback gates |

## Local model stack

Cloud Nemotron was superseded. Current story pipeline is GitHub-hosted CPU inference only.

- **Sprocket-500M chat Q4_K_M** — memory/voice source.
- **QwenStoryteller Q4_K_M** — upstream creative prose source.
- **Qwen3.5-4B Q4_K_M** — structural normalization, review and prose rendering.
- **LFM2.5-8B-A1B Q4_K_M** — current M5 edit-planning role.
- **llama.cpp b11852 / commit 7894e8b92481a8bb8ffe94f73b656ffe0dd01e10** — local runtime.

All local candidate downloads are SHA-256 verified before inference. No NVIDIA API credential is required for the current production path.

## Important execution evidence

### Run 40219258883
- local-only Qwen3.5/QwenStoryteller run;
- Sprocket and both large GGUF downloads passed;
- preserved 119-file / 172,142-byte evidence artifact;
- failed at M3 because aggregate draft remained 3,417 words after three model rewrite rebalances.

### M3 recovery
The failed M3 evidence was recovered rather than regenerated.

Recovered source state:
- aggregate: 3,417 words;
- section counts: 422 / 328 / 487 / 450 / 407 / 433 / 368 / 522;
- source aggregate SHA-256: `3601ecb3d6acacd0a528c013b1e8b398dcb97c147da5fcd189154398e117c7bf`.

Delete-only prune:
- Qwen selected candidate sentences, but the orchestrator applied only exact existing sentence matches;
- 75-word and 54-word sentences were removed from section 2;
- aggregate after prune: **3,288 words**;
- prune after-hash: `0a6a9701a173cd254de79efef877c94542bbb43dc69753cdf746c160121a3a66`;
- rejected non-matching candidate deletions are retained in `m3-resume-delete-prune-1.json`.

This made M3 pass the 2,700–3,300 acceptance gate without silently rewriting text.

### Run 40228097488
- recovered M3 and completed M4;
- evidence artifact: `story001-complete-evidence`, artifact ID **12612167359**, 231,056 bytes;
- M4 review exists and explicitly returns **REVISE, 3/10**;
- run failed in M5 when `m5-edit-01` exhausted its response token budget;
- this run is the authoritative preserved source for current M3/M4 resume work.

### Run 40229671732
- latest run at checkpoint head `a9bc8214a4fe1385f281e05d6822b17765aa57be`;
- recovered M3/M4 from run 40228097488;
- downloaded and SHA-verified **LFM2.5-8B-A1B Q4_K_M**;
- evidence artifact ID **12611803738**, 231,539 bytes;
- failed immediately on the planner→renderer split because Qwen3.5 was required for rendering but `STORY001_QWEN35_4B` was not provisioned in this resume workflow.

## Current blocker

M3 and M4 no longer need regeneration.

The remaining blocker is mechanical M5 orchestration: the resume workflow provisions LFM2.5-8B-A1B for edit planning but must also provision the already-pinned Qwen3.5-4B GGUF for the rendering half of the planner/renderer split.

No final story or M6 archive is qualified yet.

## Next action

Provision both local M5 models in the resume job, preserve LFM reasoning as the edit plan, render bounded section edits with Qwen3.5-4B, then run:

1. final 2,700–3,300 word gate;
2. exact draft/review/final SHA-256 capture;
3. independent archive read-back;
4. durable promotion under `datasets/story-001/`;
5. PR exact-head CI and final issue milestone update.

Do not reopen M0–M4 unless their preserved evidence fails read-back.
