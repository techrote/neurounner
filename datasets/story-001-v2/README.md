# STORY-001-V2 durable archive

**The Shift That Kept Turning** is the corrective STORY-001 artifact: one continuous fictional first-person Sprocket memory, replacing the earlier stitched-eight-variant narrative as the accepted story.

The earlier `datasets/story-001/` archive is preserved unchanged as superseded evidence.

## Final qualification

- words: **3057**
- standalone `I`: **64** (2.094%)
- variant separator lines: **0**
- duplicate paragraphs: **0**
- final SHA-256: `e9f010b40d81ab55e0a65c2b331284e5245c272f4142e4bb9558f467135d570f`
- provider: NVIDIA API (remote backend identity remains unpinned)
- synthesis/review: `nvidia/nemotron-3.5-bolt-e4b`
- draft/edit: `nvidia/nemotron-3.5-super-vl-120b-a12b` / `nvidia/nemotron-3.5-super-vl-120b-a12b`

The story is creative fiction, not historical Sprocket canon.

**Hash correction:** review found that the earlier recorded digest `ee163eb4ec14d0bb64edf6cd14c96edc143a583f19683d7f038ccf5d7dbb432a` did not match the committed `compiled/story.md` bytes. The canonical file SHA-256 is `e9f010b40d81ab55e0a65c2b331284e5245c272f4142e4bb9558f467135d570f`. Story text was not changed by this correction.

## Post-qualification semantic audit

The NVIDIA provider output was 3,322 words and passed the original mechanical gates, but a semantic read-back found one near-verbatim repeat of the central rescue sequence. That provider output is preserved as `compiled/story-pre-semantic-audit.md`. The accepted `compiled/story.md` removes only those three repeated paragraphs; no replacement prose was added. The accepted story is 3,057 words, contains 64 standalone `I` tokens, and has no remaining paragraph pair at Jaccard similarity ≥ 0.28. See `provenance/post-qualification-audit.json`.

## Evidence

`provider/requests/`, `provider/raw/` and `provider/calls/` retain exact provider requests, raw responses, retries and usage. `compiled/synthesis-brief.md`, `compiled/review.md` and `compiled/story.md` are the principal human-readable artifacts. `provenance/` records source hashes and the workflow run.

Actions artifacts were staging only; this committed tree is authoritative.
