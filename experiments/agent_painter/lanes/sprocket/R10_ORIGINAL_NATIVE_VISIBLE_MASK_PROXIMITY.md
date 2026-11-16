# SPROCKET R10 — original-native material-mask proximity to visible partitions

**Evidence date:** 2026-11-15 UTC. **Status:** REVIEW ONLY / GLOBAL TRAIN HOLD.  
**Independent research lane:** [PR #576](https://github.com/techrote/cybersand/pull/576) — OPEN/Draft/HOLD, unmerged.

## Source and exact-host provenance

- Pinned *root Instruct* model `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`; **this audit did not run the model or optimizer**.
- Existing original positive-native corpus source `633920373a3d406a2ede61ce4137c324f76be23b`, [run 43506373969, artifact 14076703326](https://github.com/techrote/cybersand/actions/runs/43506373969/artifacts/14076703326), ZIP SHA-256 `50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39`, manifest SHA-256 `2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`.
- Separate original **hand-authored** R10 native research producer source `ab101910921a7cf7f30b3dc83d150d214aaa060d`, [run 43582224721, artifact 14109121755](https://github.com/techrote/cybersand/actions/runs/43582224721/artifacts/14109121755), ZIP SHA-256 `f7e673c414487668598eb26dc618daa019b2fc83bc70ed373cd574049f20e53c`, canonical R10 program SHA-256 `0f52d4f6b252d17f9f335e8c8a65a44f78626e79cd26298881a78446f3f7b745`.
- Executable comparison source **`756b5f07574c3be6650a7625041fa0098d7a2109`**. GitHub-hosted ordinary CPU [run 43665041403 / job 132662425424](https://github.com/techrote/cybersand/actions/runs/43665041403/job/132662425424) **SUCCESS**: 6/6 positive/adversarial tests, source/host checks and policy PASS, both original ZIP SHA-256 checks PASS. Independently downloaded [output artifact 14141986903](https://github.com/techrote/cybersand/actions/runs/43665041403/artifacts/14141986903) ZIP SHA-256 **`6388a3313c9eb71aecfe99a6ad85f8192ebcdbb3cf7e2f1ec8a08808ccf76071`**, 3/3 ZIP CRC PASS, `source-head.txt` exactly matches tested SHA.
- Earlier [run 43664964225](https://github.com/techrote/cybersand/actions/runs/43664964225) **FAILED 5/6 source tests** due to an *incorrect synthetic test assertion* that x95,y0 is Chebyshev-distance 1 from x0,y1 (it is not). No original-data comparison receipt published by that failed job. Corrected the fixture rather than changing the metric.

## Measured native material58 comparisons

The existing R10 native composition is a **single** 10-node, 12-edge, three-cycle, five-cell-diameter orthogonal three-bay ladder, observed under two original host seeds. For each matching seed, the source-fenced script compares all native material58 pixels (96×96 grid) with the original positive-native visible Pipework reference. Each original reference readback, receipt and display/material PNG hash and material pixels are checked. **TRAIN and DEVELOPMENT only; FINAL targets/images were not opened.**

Metrics: IoU is exact original-coordinate full-material intersection/union. `C→R near3` is the fraction of R10 painted cells within Chebyshev distance ≤3 of an original reference cell; `R→C near3` reverses the direction. These are deliberately **not** graph-isomorphism or semantic-equivalence assertions. Fractions are rounded to 6 decimals.

| Visible group | Seed | IoU | C→R near3 | R→C near3 |
| --- | ---: | ---: | ---: | ---: |
| TRAIN two elbow route | 104729 | 0.062802 | 0.316705 | 0.781609 |
| TRAIN two elbow route | 130363 | 0.075556 | 0.266241 | 0.619253 |
| TRAIN tee distribution | 104729 | 0.062643 | 0.201276 | 0.525000 |
| TRAIN tee distribution | 130363 | 0.058770 | 0.205336 | 0.590000 |
| DEVELOPMENT three-branch manifold | 104729 | 0.130631 | 0.361369 | 0.622137 |
| DEVELOPMENT three-branch manifold | 130363 | 0.043659 | 0.261021 | 0.553435 |
| DEVELOPMENT closed loop with tap | 104729 | **0.175734** | **0.602088** | **0.948468** |
| DEVELOPMENT closed loop with tap | 130363 | **0.178003** | **0.497680** | **0.845404** |

**Denominator:** 4 independent visible Pipework structural groups (2 TRAIN + 2 DEVELOPMENT), 8 original seed-paired material comparisons. Each candidate seed represents the *same one authored R10*, not additional TRAIN diversity. Candidate and references are not image-aligned or shift-normalized.

## Visual inspection, risk, disposition

Original images were inspected in their actual pixel geometry: R10 has a long upper rail, long lower return and three interior vertical bays. It is distinguishable in fine topology and has no exact graph isomorphism to the four visible references or the five previously pinned VOID peer graphs, but its **rectangular silhouette is materially close to the held-out closed-loop-with-tap** under spatial near-neighbour measurement. In seed 104729, **94.8468% of the reference pixels lie within three pixels of *some* R10 pixel**; this does **not** mean 94.8468% identical pixels or demonstrate plagiarism/duplication. Exact IoU is only 17.5734%.

This is a meaningful **whole-shape split-risk flag for independent reviewer and shepherd**; do not self-certify R10 as a new globally disjoint TRAIN structure solely because graph topology differs. Do not redesign R10 by tuning to held-out coordinates or reuse DEVELOPMENT examples as TRAIN. Preserve the original proposal, all failing/passing receipts and source hashes.

**Next decision:** reviewer/shepherd inspect full peer registry plus original visible artwork and classify R10 (YES-for-admission-review / REVISE / NO). Only the shepherd/shared foundation owner may allocate a global structural ID and source-build a *separately* positive native shared TRAIN admission. The frozen common 60-request DEVELOPMENT suite is still outstanding. This report admitted **0** new TRAIN groups, launched **0** optimizer updates, **0** model outputs, **0** fresh native applications and froze **0** useful painter versions. PR remains Draft/HOLD.
