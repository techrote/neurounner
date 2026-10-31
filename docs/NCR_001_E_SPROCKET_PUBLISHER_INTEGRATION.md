# PR #102 historical Sprocket publisher-wire integration — 31 October 2026

This document explains **source-only and archival integration**, not new independent
NCR-001 #72 admission. The original source/evidence PR #102 commit
`3592841802d85c442ca2521479b5d8795bee7233` remains the second parent of the main-composed integration
commit. The immediately preceding authoritative main was `1247571ae619c244b7cb65ace9e9c0aee56b7f24`,
which already contained historical dependent #97 and #100 merges.

## Source and negative evidence

The original #102 immutable Sprocket **root Instruct** implementation pins the public
publisher SFT role, tool-result and memory masks, guardrail/serializer source, and
separate frozen-model diagnostic. Its entire original dataset of 21 responses
(seven synthetic cases across disabled, zero and manually activated *untrained*
NCA) remains intact, including invalid tool/memory generations and exhausted
token budgets. **No** new model run, behavior-scoring, hosted physical witness,
training, held-out evaluation, or learned adapter claim occurs in this integration.

The original PR explicitly noted that the complete Actions ZIP and binary
source tar were **not** durably archived in #102 itself; [dependent draft
#197](https://github.com/techrote/neurounner/pull/197) is their separate
original-byte custody and must be independently reconciled, not silently
assumed merged with #102. [Original #113](https://github.com/techrote/neurounner/pull/113)
is a separate reviewer-authored model/mask diagnostic and is likewise not
scientific or integration approval by virtue of this source composition.

## Exact-head composition and trigger isolation

- **All 20 newly added source, test, documentation and archive Git blobs**
  remain byte-identical to #102 original commit. The three added workflow YAML
  blobs are integration-only changes: both physical-model workflows are now
  `workflow_dispatch` **only**, and the receipt auditor runs exact-head
  source/custody verification and the four no-model regression suites.
- The original #102 `RAG.md` modification is retained in immutable second-parent
  history, **not replayed into current main**. Its 2026-10-27 version was an
  obsolete short retrieval map; changing modern main `RAG.md` would
  automatically trigger existing Chat/root physical workflows with unrelated
  stale-base assumptions. These source references remain accessible through
  this document and the issue/PR evidence links, with no model-triggering
  RAG modification.
- The one new diagnostic dataset row is appended to current main's dataset
  index while preserving all recent story and village entries.
- The original frozen four-model protocol and source-pin Git blobs are unchanged
  (`14ce5fbaa38f8806c8c8738f19c01b91385eebaf` and
  `59fff1aed872b5563aa6898171983973e0cdb02c`).
- The source-only test workflow checks original second-parent ancestry and
  **literal blob equality** for the complete preserved source/evidence set,
  plus the absence of automatic physical event triggers.

Explicit boundary: `train_authorized=false`,
`full_72_conformance_pass=false`,
`actual_compute_budget_qualified=false`; final per-model and four-lane
conformance still blocked. A same-account COMMENTED source review is not a
distinct-person scientific admission or GitHub APPROVED review.

Reference [PR #102](https://github.com/techrote/neurounner/pull/102),
[issue #72](https://github.com/techrote/neurounner/issues/72)
and [integration ownership](https://github.com/techrote/neurounner/issues/213).
