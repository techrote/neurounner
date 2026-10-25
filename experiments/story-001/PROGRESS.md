# STORY-001 final progress record

**Completed:** 2026-10-25  
**Issue:** #43  
**PR:** #44  
**Production run:** [40242589104](https://github.com/techrote/neurounner/actions/runs/40242589104)  
**Archive commit:** `3a4faf7b105b5c5ab4b27d27de4dc8d59fd3dd56`

## Final milestone state

| Milestone | Status | Durable evidence |
|---|---|---|
| M-1 repository / archival contract inspection | complete | issue #43 kickoff; `RAG.md`; `docs/DATASET_FILING_SYSTEM.md` |
| M0 exact runner/runtime/model preflight | qualified | production run preflight; SHA-pinned llama.cpp and local GGUF identities |
| M1 Sprocket memory elicitation | qualified | three exact pinned Sprocket requests/raw responses under `datasets/story-001/raw/sprocket/` |
| M2 structural storyboard | qualified | `datasets/story-001/compiled/storyboard.md`; SHA-256 `5e40dbb886d15d49ce619bfc21d252b3a8fb7d4e449283eb5184a7922d9c4dfa` |
| M3 long-form draft | qualified | eight-section source/normalization evidence, recovered and length-qualified from preserved Actions evidence |
| M4 independent review | qualified | `datasets/story-001/compiled/review.md`; SHA-256 `c11b8662c69068dc436a4ced10272e5795bf3a1f914202775c3996a4951fdb2b` |
| M5 revision/edit | qualified | LFM2.5-8B-A1B edit plans + Qwen3.5-4B renders + exact-sentence sanitation records under `datasets/story-001/local-models/` |
| M6 promotion/read-back | qualified | manifest, provenance, SHA-256 inventory and independent archive read-back committed under `datasets/story-001/` |

## Final artifact

- Title: **The Shift That Kept Turning**
- Kind: fictional first-person Sprocket memory
- Final length: **2,706 words**
- Accepted range: 2,700–3,300 words
- Final story SHA-256: `95bad9e1c53443eafa4c680ff4a99170e441058fc5abc72ab62a6c0fd01d15b5`
- Durable story: `datasets/story-001/compiled/story.md`
- Manifest: `datasets/story-001/manifest.json`
- Provenance: `datasets/story-001/provenance/source-artifacts.json`
- Full hash inventory: `datasets/story-001/provenance/hashes.sha256`

The incident is creative fiction. Sprocket generations establish source voice/material; they are not evidence that the narrated incident historically occurred.

## Executed local model stack

The initial cloud-Nemotron design was superseded after the user requested using the larger models already qualified on neurounner Actions runners.

- **Sprocket-500M chat Q4_K_M** — memory / voice elicitation.
- **Qwen3.5-4B Q4_K_M** — storyboard, normalization, independent M4 review and final prose rendering.
- **QwenStoryteller Q4_K_M** — upstream M3 creative prose source.
- **LFM2.5-8B-A1B Q4_K_M** — M5 edit planning.
- **llama.cpp b11852**, commit `7894e8b92481a8bb8ffe94f73b656ffe0dd01e10` — CPU runtime.

Every production GGUF was SHA-256 verified on `ubuntu-24.04` before inference. The final production path used **no remote model provider**.

## Recovery / audit trail

STORY-001 deliberately preserved failed and partial work instead of regenerating blindly.

- Run `40219258883`: local M3 reached 3,417 words; 119-file evidence artifact preserved.
- M3 recovery: exact-sentence deletion reduced the recovered draft to 3,288 words while retaining before/after hashes and rejected deletion candidates.
- Run `40228097488`: completed M4 review; M5 initially hit an output budget.
- Subsequent M5 resume runs progressively preserved section plans/renders and exposed bounded length/content guards.
- Final M5 used LFM8B planning, Qwen3.5 rendering, and recorded exact-sentence deletion for predeclared scene/lore violations.
- Run `40242589104`: M0–M6 PASS; final story 2,706 words; archive committed as `3a4faf7b105b5c5ab4b27d27de4dc8d59fd3dd56`.

Actions artifacts are retained as provenance/staging evidence, but the committed `datasets/story-001/` tree is authoritative.
