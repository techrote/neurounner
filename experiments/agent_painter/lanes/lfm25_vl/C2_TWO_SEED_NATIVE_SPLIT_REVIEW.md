# LFM C2 dual-cross Pipework — original two-seed native split review

**2026-11-14 UTC · reviewer handoff · RESEARCH ONLY · TRAIN HOLD.** This review resolves a specifically requested two-seed, original-byte comparison of the LFM C2 teacher proposal with the **visible TRAIN and DEVELOPMENT** Pipework structures. It does **not** declare global split safety, change any corpus, inspect FINAL, retrain a model, or reclassify the original failed native workflow.

## Immutable inputs and custody

- LFM original authored C2 and original native research source: [run 43587733967, artifact 14111581483](https://github.com/techrote/cybersand/actions/runs/43587733967/artifacts/14111581483), producer `abef683ceaa0fb2782e2d4b65db2c4c3c475e52d`; ZIP SHA-256 `e90014c8883bb89e1caaebfdb00cf28f2078604553a6b8de803e6caf9305aebf`, **62/62 ZIP members CRC valid**. Original workflow conclusion is **FAILED** due to its post-run custody key; [retrospective audit 43589453856](https://github.com/techrote/cybersand/actions/runs/43589453856) independently confirmed physical execution without changing the producer's outcome.
- Original genuinely native-admitted pilot: [run 43506373969, artifact 14076703326](https://github.com/techrote/cybersand/actions/runs/43506373969/artifacts/14076703326), source `633920373a3d406a2ede61ce4137c324f76be23b`, ZIP SHA-256 `50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39`, **286/286 ZIP member CRC valid**, original `corpus/admitted-corpus.json` SHA-256 `2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`. Only the explicitly enumerated `corpus/train.jsonl` (`5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1`) and `corpus/development.jsonl` (`8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572`) were opened for membership and original source graph comparison. **`corpus/final.jsonl` and FINAL images were not opened.**
- The C2 program remains exactly `proposals[1].program` in `contrastive_proposals_v1.json` original blob `226ce959c68a9e2649e88fe15c10713437cf82be`; no coordinate, graph or style mutation. Material ID **58**, canvas **96×96**, host seeds **104729 and 130363**.

Both ZIPs were freshly retrieved and SHA-256/CRC checked for this review. For **10/10 selected original seed views** (C2 ×2 and four visible pilot Pipework groups ×2), each original little-endian signed-32 material crop exactly matched its receipt's SHA-256, source-admitted cell counts and legal material IDs `{0,58}`. Grayscale material-ID PNGs exactly matched **92,160/92,160** material cells; original colour PNG foreground/background exactly matched native occupied/empty cells, also **92,160/92,160** pixels. This independently confirms the selected *existing original readback*, not a new Godot execution. C2's two original views each acknowledge exactly **701** changed cells, with zero unexpected/outside cells. Seed variants represent **one** structural programme.

## Original native-image comparison

Compute **exact material-mask IoU** on same-seed, unscaled 96×96 crops. `C2 within 3` is directional coverage: fraction of C2 occupied cells at Chebyshev distance ≤3 from any reference occupied cell (7×7 maximum filter). These are descriptive measures, **not automatic admission thresholds**.

| Reference | Split | C2 IoU seed 104729 | C2 within 3 seed 104729 | C2 IoU seed 130363 | C2 within 3 seed 130363 |
|---|---|---:|---:|---:|---:|
| `pipework/tee_distribution/v1` | TRAIN | 3.34% | 13.12% | 2.36% | 21.68% |
| `pipework/two_elbow_route/v1` | TRAIN | 2.65% | 11.84% | 5.28% | 18.26% |
| `pipework/three_branch_manifold/v1` | **DEVELOPMENT** | **9.10%** | **50.07%** | **5.69%** | **60.63%** |
| `pipework/closed_loop_with_tap/v1` | DEVELOPMENT | 3.96% | 26.82% | 5.82% | 32.38% |

The same-structure seed change does *not* create an independent test design. The strong **50.07–60.63%** directional proximity to the three-riser DEVELOPMENT design warrants a **semantic near-neighbour warning**, even though exact pixel IoU is low. Comparing only the 104729 view would understate this range.

## Source-graph and visual-semantic contrast

| Witness | LFM C2 teacher | Original held-out three-branch manifold |
|---|---|---|
| Node degrees (sorted) | `[1,1,1,1,1,1,4,4]` | `[1,1,1,1,1,3,3,3]` |
| Endpoint terminals | **6** | **5** |
| Four-way crosses / three-way tees | **2 crosses / 0 tees** | **0 crosses / 3 tees** |
| Vertical terminal branches | **2 up + 2 down** from two crosses | **3 up + 0 down** from trunk tees |
| Graph cycles | 0 | 0 |
| Constant diameter | 3 | 3 |
| Overall silhouette | a central horizontal connector between two full-height vertical cross-arms | a low horizontal trunk with three upward-only risers |

The original full-resolution native PNGs exhibit that distinction: C2 looks like **two vertical crossing pipe columns** joined by a trunk, with substantial empty space between and around the lower and upper ends. The original DEVELOPMENT example looks like a **comb with three upward branches above a lower trunk**. Both share the more generic visual motif “horizontal pipe with attached vertical arms”; calling them identical or wholly independent on either degree counts or pixel score alone would be unjustified.

The original TRAIN tee and elbow have diameter **5** and different topology; neither isomorphic to C2. The held-out one-cycle tap has diameter 3 but **one loop**, rather than C2's acyclic two-cross network. All source graphs came from the original approved TRAIN/DEVELOPMENT rows and C2's unchanged authored source; there is no reconstructed/guessed model output.

## Reviewer decision requested — not self-admission

**Disposition: C2 GLOBAL REVIEW HOLD**, preserved. Independent reviewer: decide **YES for one split-safe review candidate / REVISE / QUARANTINE** after evaluating these two original views and the shared whole-structure peer registry, especially prior SPROCKET cyclic networks and SMOL nested fan. If accepted *for review*, the shepherd must allocate the global canonical group ID, approve an unambiguous **dual-cross + upward/downward** training request, provide a shared-source write lease and admit it through a **new successful source-built native common TRAIN run**. The original failed producer and retrospective audit cannot substitute for that new admission. A new frozen common DEVELOPMENT suite is separately required before another LFM optimization campaign.

No new model-generated views, optimizer updates, common TRAIN groups, source modifications to peer/shared files, FINAL access, promotion or merge. Historical LFM native model request conformity remains **0/4**; useful vN freezes remain **0**.

### Reproduction (read-only)

Retrieve the two immutable ZIPs above and verify SHA-256, then CRC all members. Decode only the named TRAIN/DEVELOPMENT JSONL rows and the ten selected `*-s104729/s130363` original `*.receipt.json`, `*.materials.i32le`, `*.materials.png` and `*.png` views; assert each crop SHA-256 and 96×96 material alignment. Calculate mask `(native_i32le == 58)`, IoU as intersection/union and directional within-three coverage by 7×7 dilation. Infer graph degrees, endpoints, vertical terminal orientation, cycles and diameter directly from the original `program.components[0]` node/edge arrays, *not* from filename labels. Do not access the FINAL split.
