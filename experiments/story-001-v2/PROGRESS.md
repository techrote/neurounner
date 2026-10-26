# STORY-001-V2 final progress record

**Corrective objective:** replace the rejected stitched-eight-variant STORY-001 narrative with one continuous NVIDIA-produced short story of at least 3,000 words, using the user's preferred burnt-copper / wet-wool failed-rescue passage as the primary tonal anchor and substantially reducing standalone `I` usage.

**Issue:** #43  
**PR:** #44  
**Successful NVIDIA production run:** [40247262042](https://github.com/techrote/neurounner/actions/runs/40247262042)  
**Provider archive commit:** `059e9cc72763fae8643d0c5faaee325e5238e0cc`  
**Accepted post-audit story SHA-256:** `ee163eb4ec14d0bb64edf6cd14c96edc143a583f19683d7f038ccf5d7dbb432a`

## Corrective pipeline

1. Preserve `datasets/story-001/` unchanged as superseded v1 evidence.
2. Record the user's preferred passage and anti-stitch constraints in `experiments/story-001-v2/ANCHOR.md`.
3. NVIDIA Nemotron 3.5 Bolt E4B synthesizes the strongest unique material from the v1 variants into one chronological/spatial brief.
4. NVIDIA Nemotron 3.5 Super VL 120B-A12B writes five sequential, non-overlapping continuations of the same incident.
5. Bolt independently reviews the assembled draft for cohesion, repetition, continuity, Sprocket voice and pronoun rhythm.
6. Super performs bounded segment-wise editing while preserving one continuous arc.
7. Mechanical qualification enforces >=3,000 words, <=80 standalone `I` tokens, <=2.5% standalone-`I` rate, zero variant separators, no exact duplicate paragraphs and at most one anchor reset.
8. A separate semantic read-back checks for near-duplicate paragraphs. The provider-qualified output still repeated the central rescue once; the exact repeated three-paragraph block was deleted without adding replacement prose.
9. The original 3,322-word provider output remains preserved as `compiled/story-pre-semantic-audit.md`; the accepted story is `compiled/story.md`.

## Final accepted artifact

- title: **The Shift That Kept Turning**
- artifact: `datasets/story-001-v2/compiled/story.md`
- length: **3,057 words**
- standalone `I`: **64**
- standalone-`I` rate: **2.094%**
- variant separator lines: **0**
- anchor reset phrase count: **0**
- exact duplicate paragraphs: **0**
- paragraph pairs at Jaccard similarity >= 0.28 after semantic audit: **0**
- final SHA-256: `ee163eb4ec14d0bb64edf6cd14c96edc143a583f19683d7f038ccf5d7dbb432a`

## NVIDIA identities

- synthesis: `nvidia/nemotron-3.5-bolt-e4b`
- independent review: `nvidia/nemotron-3.5-bolt-e4b`
- sequential drafting: `nvidia/nemotron-3.5-super-vl-120b-a12b`
- editing/repair: `nvidia/nemotron-3.5-super-vl-120b-a12b`
- provider: NVIDIA API, remote backend identity unpinned
- Super prose calls used direct final-output mode (`reasoning_effort: none`) after an earlier failed run showed the default reasoning mode could exhaust the token budget before emitting prose.

## Evidence and failed runs

- `40246785924`: Bolt/Super preflight passed; Super exhausted draft budget in reasoning-only mode and emitted no final prose. Evidence preserved.
- `40246971066`: direct-output Super emitted prose, but a monolithic generation/repair path remained too short and duplicated paragraphs. Evidence preserved.
- `40247262042`: segmented continuation pipeline succeeded and promoted 3,322-word provider output.
- Post-qualification semantic audit detected a near-verbatim rescue repetition (maximum paragraph-pair Jaccard 0.966) that the exact-duplicate gate missed.
- `provenance/post-qualification-audit.json` records the exact three deleted paragraphs, before/after hashes and similarity evidence.
- No new prose was introduced during the semantic correction.

The incident remains fictional character memory, not historical Sprocket canon.
