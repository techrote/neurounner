# SPROCKET G09 vs SMOL G06 — original native-material comparison (14 November 2026 UTC)

**Status:** `PAIRWISE_NATIVE_GEOMETRY_DIFFERENT / GLOBAL_SPLIT_HOLD / TRAIN=0`. Independent read-only research. This is **not** a new native execution, model inference, new optimization run, useful painter, frozen DEVELOPMENT benchmark, or shared TRAIN admission.

## Immutable original custody

I downloaded two complete *original* public GitHub-hosted Art research archives afresh and independently opened their original 96×96 native material grids, PNGs and receipts:

| Case producer | Original run / artifact | Whole original ZIP SHA-256 | ZIP entries / CRC |
| --- | --- | --- | --- |
| SPROCKET v3 G09 | [43587600137](https://github.com/techrote/cybersand/actions/runs/43587600137) / [14111800549](https://github.com/techrote/cybersand/actions/runs/43587600137/artifacts/14111800549); producer `63ee8b2b561068a72e0652e54b6e4193cb47d5e6` | `3a7397883b146a28da990f0a37b891897285c2c7c81cd9dd2aaa3895d0d196dd` | 65/65 |
| SMOL v2 G06 | [43601281142](https://github.com/techrote/cybersand/actions/runs/43601281142) / [14117110667](https://github.com/techrote/cybersand/actions/runs/43601281142/artifacts/14117110667); producer `6e412178c9b71cb0196d8e35408470077b50f884` | `71ddc40478e06e3350f05eecd5e16ed77c082300f066be97bbccf8afb2370ff1` | 88/88 |

Archive digests match GitHub's original artifact digests. All original case readbacks (`.materials.i32le`), final PNG and receipt bytes were independently SHA256-checked against the *original* native verifier case entries. Each raw grid had exactly 9,216 little-endian signed 32-bit cells, IDs **only 0 and 58**; all 36,864 rendered foreground pixels and all 36,864 grayscale material-PNG pixels match the raw material grid. Native research receipts were positive and deterministic, with no new Art replay triggered by this comparison.

## Actual native material-mask contrast

Both candidates are represented by the **same two host seeds**, but a seed variation does not create a second independent structure.

| Host seed | SPROCKET G09 material58 cells | SMOL G06 material58 cells | Shared foreground cells | Union | Native-mask IoU | Dice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 104729 | 847 | 547 | 32 | 1,362 | **0.02349486** | 0.04591105 |
| 130363 | 900 | 535 | 41 | 1,394 | **0.02941176** | 0.05714286 |

Native material readback demonstrates **three unusually wide horizontal, descending, separated bands** in G09 against **three compact diagonally ascending cassettes** in G06. Both original seed images were inspected side by side. The low native-mask overlap is additional pairwise evidence beyond source-panel box IoU **0.1182667** at [the previous pinned source check](https://github.com/techrote/cybersand/actions/runs/43607349533). Distinct native masks alone do **not** establish semantic independence: a translation, change of scale, random detail, or paint density might also produce low IoU.

### Exact limits and requested global decision

- This comparison reuses **four existing positive native seed views / two structural candidates**; **zero** new native executions, model-generated images, optimizer updates, benchmark requests, or shared TRAIN admissions.
- Original source633 corpus source remains `633920373a3d406a2ede61ce4137c324f76be23b`, manifest SHA-256 `2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`. No original FINAL target/image opened.
- The visual and adjacency difference is concrete for **G09 vs G06 only**. It is **not** a global originality certificate or proof of high artistic value; the examples are still relatively sparse hand-authored Greebles.
- **Shepherd/reviewer:** assess G09 against the original visible TRAIN/DEVELOPMENT composition semantics and the other lanes' original native designs. If suitable, assign a globally unique structural ID and decide split membership via a separate versioned positive-admission receipt. Preserve both original research producers. Otherwise quarantine with a specific collision witness.
- Keep sole [SPROCKET Draft PR #576](https://github.com/techrote/cybersand/pull/576) **Draft/HOLD**. Do not restart optimization on the unchanged six-structure corpus or freeze/promote the existing unqualified 96-step model.

**Reproduction:** Open both original ZIP archives listed above; compare `sprocket-v3-native/sprocket-G09-s{seed}.materials.i32le` with `smol-v2-native/smol_g06_stepped_absence-s{seed}.materials.i32le`. Decode each as exactly 96×96 little-endian `int32`; treat cells equal to 58 as a foreground set; compute intersection, union, IoU and Dice. Validate case SHA256 against each archive's `native-verification.json`, and the 96×96 PNG RGB foreground palette (background 13/21/28, wall 232/226/205) and grayscale material PNG against each grid. Local standard-library/Pillow verification plus 8/8 model-free adversarial tests passed; this comparison was **not** independently rerun on GitHub Actions at this documentation checkpoint.


## Cross-check against the *original visible* TRAIN and DEVELOPMENT Greebles (same inspection)

An additional fresh download of the **original** source633 native corpus [run 43506373969, artifact 14076703326](https://github.com/techrote/cybersand/actions/runs/43506373969/artifacts/14076703326) established ZIP SHA256 `50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39`, all **286/286 CRC PASS**, TRAIN file SHA256 `5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1`, DEVELOPMENT file SHA256 `8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572`. Only the visible `train.jsonl`, `development.jsonl`, and their referenced **Greebles** native PNG/readback/receipt files were opened. The sealed `final.jsonl` and all FINAL targets/images were **not** opened.

The four *independent* original Greebles groups each supply two host seeds (**4 TRAIN + 4 DEVELOPMENT seed views**). All 24/24 original image/readback/receipt SHA256 matches, and **73,728/73,728 visible original rendered pixels** match authoritative material-ID grids. Reusing G09's corresponding native material grids gave these strictly *geometric* foreground intersections:

| Original structural group | Original split | Seed 104729 native IoU | Seed 130363 native IoU | Native structural observation |
| --- | --- | ---: | ---: | --- |
| `greebles/service_cluster/v1` | TRAIN | 0.051448 | 0.063875 | Four unequal clusters, including a large upper-left cassette; not three full-width descending bands |
| `greebles/switchboard_stack/v1` | TRAIN | 0.060658 | **0.119286** | Vertical tall rail plus right-side stack; still different hierarchy and adjacency |
| `greebles/open_center_modules/v1` | DEVELOPMENT | 0.068027 | 0.110029 | Four modules around an internal void; G09 has two exterior-spanning service gaps |
| `greebles/spine_assembly/v1` | DEVELOPMENT | 0.073869 | 0.087790 | Central vertical spine and unequal side panels; no G09-like horizontal band order |

All five named original native Greebles compositions (G09 plus these four) were viewed side by side at both host seeds. **No obvious same-composition copy in these four original public Greebles groups** was observed. The lowest overlap figures alone do not prove semantic novelty. The closest *pixel-space* comparison is original TRAIN `switchboard_stack/v1` seed 130363, IoU 0.119286, but its vertical rail/right-stack configuration is qualitatively different from G09's descending horizontal bands.

This narrows the remaining review to **other peer-lane candidate semantics and a global shared split decision**, not an uninspected original public Greebles group. This evidence is *not* a new native corpus admission, does not introduce a new benchmark or elevate G09 above the explicit global TRAIN HOLD gate.
