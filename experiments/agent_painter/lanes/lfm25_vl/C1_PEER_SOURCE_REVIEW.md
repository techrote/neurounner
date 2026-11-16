# LFM C1 versus four pinned peer Greebles — source-box comparison

**14 November 2026; research-only, Draft/HOLD.** This is a bounded,
machine-checked **program-panel-box** comparison, not a native image similarity
test, model evaluation, frozen TRAIN admission or independent reviewer decision.

## Original sources and completed evidence

The lane's `c1_peer_review.py` reads only the following immutable **source
bytes**, enforcing their Git blob IDs before parsing. SPROCKET's `STUDIES`
dictionary is decoded with Python `ast.literal_eval`, never imported or run.
All three source files are covered by SHA-256 in the resulting receipt:

| Source | Commit | Exact Git blob |
| --- | --- | --- |
| LFM C1 `contrastive_proposals_v1.json` | `5fd5e33aad38296155c7bb47cf37a07f0ee07946` | `226ce959c68a9e2649e88fe15c10713437cf82be` |
| VOID G01/G02 `v3_research_proposals.json` | `ea056e395295f1fd3f8ed64c039b746555060d61` | `0e594de0b683af2547ca3c09e978cf61a349392e` |
| SPROCKET G08/G09 `threefamily_v3.py` | `917425fee7c2b4a5ede2d9d8bbf085d4b4091d63` | `b3ac98a8cd06158b39a2f58776fab845046f245f` |

After repairing an **invalid adversarial fixture**, the exact-source hosted
[run 43602322934](https://github.com/techrote/cybersand/actions/runs/43602322934)
at `80f552df9d26d7ba4706e4475b796fe7ed8c3147` completed SUCCESS:
**8/8** model-free tests, pinned raw peer downloads, Git blob checks and
four-peer comparisons passed. The independently re-downloaded
[artifact 14116338874](https://github.com/techrote/cybersand/actions/runs/43602322934/artifacts/14116338874)
is **1,013 bytes**, complete ZIP SHA-256
`9cda21cc9edf332d849fbdc94d076640ee28991a2cbdf53719a5af03def9ffe2`
**MATCH** GitHub, single ZIP member CRC PASS. Its receipt independently
reports the source file SHA-256s and all values below. Initial
[run 43602238097](https://github.com/techrote/cybersand/actions/runs/43602238097)
failed 7/8 tests because a deliberately dislocated fixture was moved to
y=78 with height 20, exceeding the 96-cell canvas; it remains FAILED.
The corrected y=75 remains inside bounds and detects real tier misalignment.
No native or optimization job was retried as part of this repair.

## Spatial differences and nearest peer

Metrics operate on the **union of the 96×96 rectangular panel bounding boxes**
from the exact authored programs, NOT on rendered foreground/native pixels.
The central chase is the entire horizontal band x=40..55 inclusive across
full height; a proposal has a clear chase only if **no panel box overlaps it**.
Three paired tiers means two separated three-panel vertical banks aligned
within eight cells of pair-centre y. Source-box IoU uses
intersection / union of the two rectangle-union masks.

| Program | Boxes | Cross-chase boxes | Full-height chase | Three paired tiers | Box IoU with C1 |
| --- | ---: | ---: | --- | --- | ---: |
| **LFM C1** | **6** | **0** | **yes** | **yes** | 1.000000 |
| VOID G01 | 5 | 1 | no | no | 0.332478 |
| VOID G02 | 4 | 1 | no | no | 0.331944 |
| SPROCKET G08 | 5 | 3 | no | no | **0.402281** |
| SPROCKET G09 | 3 | 3 | no | no | 0.309895 |

**Source-only interpretation:** C1's six cassette pairs with a complete
central chase distinguish it from VOID G01's 3+2 bank whose top centre
cassette enters that band; VOID G02's diagonal stagger; SPROCKET G08's
five-cassette stepped layout (the **nearest** tested peer by box IoU);
and SPROCKET G09's three long horizontal bands. This is a **specific
visible motif difference**, not a proof that the six detailed cassette
contents/styles are semantically independent, that new global curriculum
groups should be counted, or that low coordinate overlap implies novelty.

The previously inspected **original** C1 teacher images and readbacks arose
from the source-built [failed-original native run 43587733967](https://github.com/techrote/cybersand/actions/runs/43587733967).
Its post-run custody-field spelling defect was separately corrected in a
**retrospective** [audit 43589453856](https://github.com/techrote/cybersand/actions/runs/43589453856)
with 148/148 checks. It remains a hand-authored, native-executed teacher
design under review, **not trained LFM output**. Neither previous original
ZIP nor original failed job was changed here.

## Explicit handoff and limits

Request independent reviewer/shepherd inspect C1's original PNG against
**VOID G01/G02 and SPROCKET G08/G09** with the literal paired-bank/chase
witness above, as well as the original visible **TRAIN/DEVELOPMENT** and
remaining peer proposals. The current comparison reads four pinned peer
source programs, not complete global dataset galleries. No FINAL
target/image/split was read; no shared curriculum registry or model
checkpoint was changed. C2 still needs separate globally coordinated
Pipework review; C3 remains P2 Architecture near-alias QUARANTINED.

**Disposition:** source-level C1 distinction supported; global
independence/admission **PENDING**. Newly admitted TRAIN groups **0**,
new optimizer updates **0**, new native executions **0**, model calls
**0**, new useful vN freezes **0**. Do not start another LFM optimization
campaign until shepherd/reviewer establish positively native-admitted
structurally new TRAIN and a frozen shared 60-request DEVELOPMENT suite.
