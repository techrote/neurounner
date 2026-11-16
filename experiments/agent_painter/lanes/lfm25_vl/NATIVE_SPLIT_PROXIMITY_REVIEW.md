# LFM C1/C2 — original native-pixel TRAIN/DEVELOPMENT proximity review

**2026-11-14 UTC; LFM lane; Draft/HOLD; research-only.** This is an independent, read-only comparison of **original executed native material/PNG bytes** against the original visible TRAIN/DEVELOPMENT gallery and selected independently executed peer research images. It supplements, rather than replaces, the earlier program-box and graph-only studies. It does **not** establish global novelty, change a split, admit TRAIN, qualify a trained model, or evaluate FINAL.

## Custody and scope

Fresh original GitHub-hosted artifact downloads, recomputed **outer ZIP SHA-256**, and complete ZIP CRC validation:

| Artifact | Original run and artifact | Verified whole ZIP SHA-256 |
| --- | --- | --- |
| Original pilot **TRAIN + DEVELOPMENT only** native evidence | [43506373969 / 14076703326](https://github.com/techrote/cybersand/actions/runs/43506373969/artifacts/14076703326) | `50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39` |
| LFM C1/C2 original physically executed research (original post-run verification **FAILED**) | [43587733967 / 14111581483](https://github.com/techrote/cybersand/actions/runs/43587733967/artifacts/14111581483) | `e90014c8883bb89e1caaebfdb00cf28f2078604553a6b8de803e6caf9305aebf` |
| SMOL G06/P01 original successful native research | [43601281142 / 14117110667](https://github.com/techrote/cybersand/actions/runs/43601281142/artifacts/14117110667) | `71ddc40478e06e3350f05eecd5e16ed77c082300f066be97bbccf8afb2370ff1` |
| SPROCKET G08/G09 original successful native research | [43587600137 / 14111800549](https://github.com/techrote/cybersand/actions/runs/43587600137/artifacts/14111800549) | `3a7397883b146a28da990f0a37b891897285c2c7c81cd9dd2aaa3895d0d196dd` |

All four ZIPs passed **member CRC checks**. Only **14 named original seed-104729 material/PNG/receipt views** were extracted and interpreted: C1, C2; **8** original visible TRAIN/DEVELOPMENT structural groups (**2 TRAIN + 2 DEVELOPMENT per family**); and **4** peer research designs. For **14/14** views, the complete **96 × 96 signed i32le native grid** SHA-256 matched its unchanged receipt's `crop_i32le_sha256`; the reported filled-cell count matched the authoritative number of material-58 cells; and **9,216/9,216** original PNG foreground pixels per view matched native material occupancy, **129,024/129,024** in total. Every selected grid contained only material IDs 0 and 58. The original pilot's `gallery/manifest.json` supplied authoritative TRAIN/DEVELOPMENT membership, not guessed filename labels. Original corpus source `633920373a3d406a2ede61ce4137c324f76be23b`, original manifest SHA256 `2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`. **No FINAL source, target, image, or label was accessed.**

LFM C1 original crop SHA256 `869a9b6abc0ff87d703cda28d8f0aa867cc71b08c5644c9a0b5fa759010bf420`, 1,021 material cells, PNG SHA256 `ec83a408c6b207ea8ea871f9cfb5a1a42e574044f6be40993a6f371b4c245445`. C2 original crop SHA256 `9bc7413fbeb3927611c94b2b7f44b9b8644e7a9e646fd89593affe2085dace0c`, 701 material cells, PNG SHA256 `b5d899e02425311fffb830be28566bd44b48db53b450cf66cec394fd63861df5`. The earlier independent [retrospective correction run 43589453856](https://github.com/techrote/cybersand/actions/runs/43589453856) proved six physical executions under a single temporary custody-key correction; the original [43587733967](https://github.com/techrote/cybersand/actions/runs/43587733967) remains **FAILED**, not relabelled as a successful producer job. These are **hand-authored teacher programs**, NOT LFM inference.

## Native material-mask comparison

For each aligned 96×96 authoritative material-58 occupancy grid:

- **Exact IoU** = occupied pixel intersection / occupied pixel union, expressed as a percent.
- **C1/C2 within 3** = fraction of candidate occupied pixels lying within **Chebyshev distance ≤3 cells** of *any* reference occupied pixel. Constructed by a 7×7 binary maximum-filter dilation of the reference. It is **directional**; the reverse coverage is different. This deliberately tests near-placement even where exact masks do not intersect.
- Exact PNG/native readback checks are **not** proof of semantically similar compositions. A dilation-tolerant statistic can become large when many rectangular edges share a broad grid. Do not turn any figure into an admission threshold without reviewer agreement.

### Greebles C1, original native result

| Reference | Split/status | Exact mask IoU | C1 pixels within 3 of reference |
| --- | --- | ---: | ---: |
| `greebles-switchboard_stack` | **TRAIN** | **12.58%** | **98.43%** |
| `greebles-service_cluster` | **TRAIN** | 8.04% | 80.02% |
| `greebles-spine_assembly` | **DEVELOPMENT** | 8.29% | 86.39% |
| `greebles-open_center_modules` | **DEVELOPMENT** | 8.20% | 64.54% |
| SPROCKET G08 | peer research | 9.50% | 69.44% |
| SPROCKET G09 | peer research | 4.01% | 58.86% |
| SMOL G06 | peer research | 7.32% | 31.24% |

**Material finding:** C1's genuinely six-paired-tier/clear-centre composition has a **strong spatial placement proximity to the original TRAIN switchboard_stack image**, despite only 12.58% exact material overlap. This is stronger caution than prior **source panel-box comparisons with peers alone** provided. The motif is still visually distinguishable in panel count, proportions, arrangements and internal detail. Treat this as **C1 REVIEW HOLD / POSSIBLE TRAIN SPATIAL NEAR-ALIAS**, *not* either a proved duplicate or a new approved TRAIN group. Independent reviewer should examine both **original full-resolution images**, not select on box IoU or degree statistics. If rejected, author a genuinely different C1 replacement rather than merely shifting the same six frames by one or two pixels.

### Pipework C2, original native result

| Reference | Split/status | Exact mask IoU | C2 pixels within 3 of reference |
| --- | --- | ---: | ---: |
| `pipework-three_branch_manifold` | **DEVELOPMENT** | **9.10%** | **50.07%** |
| `pipework-closed_loop_with_tap` | **DEVELOPMENT** | 3.96% | 26.82% |
| `pipework-tee_distribution` | **TRAIN** | 3.34% | 13.12% |
| `pipework-two_elbow_route` | **TRAIN** | 2.65% | 11.84% |
| SMOL P01 | peer research | 5.36% | 28.10% |

**Material finding:** C2 remains comparatively differentiated from original TRAIN raster shapes, but its horizontal trunk plus upward branches resembles the original **held-out DEVELOPMENT three_branch_manifold at the motif level**. The two-degree-four-cross/six-terminal *source graph* (two cross junctions, no cycles) differs topologically from the held-out manifold. Pixel similarity alone cannot exclude prompt-semantic leakage or certify a globally distinct structure. **C2 REVIEW HOLD**, with the held-out manifold explicitly prioritized for native visual/semantic review before any shared TRAIN admission. C3 remains P2 Architecture near-alias **QUARANTINED**.

## Reproduction and limits

One can independently recompute this report without model execution: download the **four immutable original ZIPs** above and verify their digests; select `lfm-contrastive-native/lfm-c1-six-bay-service-chase-s104729` and `lfm-c2-dual-cross-six-terminal-network-s104729`, original pilot `gallery/manifest.json` and the matching visible TRAIN/DEVELOPMENT `*-s104729` files; and named SMOL/SPROCKET native peer files. Parse each `*.materials.i32le` as **9,216 little-endian signed 32-bit cells**, verify SHA against `*.receipt.json`, compare **material 58** with corresponding original `*.png` foreground (RGB **232,226,205**; background RGB **13,21,28**); then compute exact binary IoU and directional dilation coverage as defined. No rescaling, image search, prompt-generation fallback or synthetic replacing of original PNGs was used in the numeric comparisons. The displayed galleries for manual review were enlarged separately by nearest-neighbour only and were **not** used to calculate metrics.

This independent container-side analysis **is not a new GitHub-hosted native or model job** and introduces **no training, shared corpus mutation, new native execution, model checkpoint, or useful vN freeze**. Source-built original research-native evidence is preserved, but only LFM's retrospective audit can clear its malformed original post-run receipt. The model's historical DEVELOPMENT remains **1/6 strict compiler accepted per 48- and 120-step arm; native 4/4 physically painted but 0/4 satisfied the requested Pipework graph**. Shared reviewer/shepherd authority over global identity, TRAIN admission and the frozen 60-request DEVELOPMENT suite remains unchanged.
