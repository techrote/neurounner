# M8 — local novelty/style distillation on fresh obstacle topologies (16 Nov 2026)

**Disposition: Research HOLD.** Synthetic 96×96 host only; no native CyberSand, FINAL, Workbench, gameplay, default-provider or paid-GPU work.

## Real executed experiment

- Three actual newly optimized decision heads: 20,549-parameter novelty MLP; 20,549-parameter style MLP; 53,605-parameter style CNN. **600 AdamW optimizer updates** (200/head). No pretrained backbone weights or LoRA were changed.
- TRAIN 21000–21035 (19,980 paired labels from local host states), DEV 22000–22007 (4,440); dedicated new M8 EVAL world seeds 34000–34011. TRAIN/DEV/EVAL seed groups disjoint; 6 new obstacle generators, 3 family labels/world, 10 matched policies, 64 unrestricted legal decisions.
- The local observation is the 162-float 7×7 local W0-derived vector. All actions belong to host-supplied `UP DOWN LEFT RIGHT STOP` masks. Protected-cell traps teach STOP when it is the sole legal action. No privileged global reachable map is passed to policies.
- **360 original PNG/JSON rollouts**, 23,040 independently recomputed full neural/action decisions and host state transitions, zero applied illegal actions, 727/727 original manifest members and 5/5 tests PASS. Two complete CPU runs reproduced **728/728 source-generated evidence files byte-for-byte**.

## Out-of-curriculum matched 64-action results

| Policy | Mean distinct cells | Mean branch nodes | Mean turns |
| --- | ---: | ---: | ---: |
| Novelty deterministic | **57.417** | 0.417 | 3.500 |
| Teacher local novelty | 55.917 | 1.417 | 29.000 |
| Existing M5 convolutional | 55.056 | 1.028 | 20.306 |
| **New M8 novelty MLP** | **53.167** | **2.583** | 17.778 |
| M8 style CNN | 52.472 | 0.972 | 22.639 |
| Existing M5 MLP | 51.944 | 0.889 | 20.639 |
| M8 style MLP | 47.611 | 1.139 | 27.972 |
| Random legal | 22.694 | 6.889 | 42.639 |

Paired differences (12 seed-cluster bootstraps): M8 novelty MLP vs M5 conv **−1.889 cells** [−9.861,+4.222] and **+1.556 branch nodes** [+0.611,+2.500]. Increased branch count is measurable on these test seeds, but does not prove aesthetic value. The style MLP makes +7.333 more turns than M5 MLP [1.472,13.778] while losing −4.333 cells [−11.694,+2.861].

## Critical negative finding

Novelty MLP remains largely **family-insensitive**: cross-family mean footprint IoU **0.917**, with 22/36 cross-family action pairs identical, despite higher branches. The style MLP lowers cross-family IoU to **0.349**, only 4/36 identical pairs, but at a coverage cost. Novelty-vs-style supervision is a real tradeoff, not a winner. No candidate passes native, artistic or promotion gates.

Single-observation local CPU forward-only median: M8 novelty MLP **0.0305 ms**, style MLP **0.0305 ms**, style CNN **0.0718 ms**; p95 0.0629/0.0545/0.1604 ms. Not end-to-end per-turn CPU/host/GPU latency.

## Exact provenance and qualification boundary

- New executed source SHA-256: `m8_style_distill.py` `b9cd88ee56be301006256be711c41246750cc209de18ae9ee9d125b6202bbe8a`; `audit_m8_style.py` `92aadf5944ebec7cf45454970665671473b521dccbbb084620dcce1f7d9f398e`; `test_m8_style.py` `1f87a117fd66fc2844f854d78943e613cebb911ddafcf8836050064c9b70040a`.
- Complete local reproducibility ZIP SHA-256 `f195b73bf448d59eae88f644d33aabf6197d71a78510d6f9f3ae6920c1211803`. Original PNGs, training checkpoints, 360 receipts, all original/executed sources, local M5 baselines and full audit are in the ZIP supplied in the associated ChatGPT conversation; **the ZIP is not a GitHub Actions artifact**.
- **Provenance gap:** While the executed M8 scripts are included verbatim in that ZIP, they are **not yet committed to this branch**. The archived `core.py` and `m5_topology_train.py` used during execution have Git blob differences against current repository HEAD; therefore **not exact-HEAD source-qualified**. Preserve the existing W0 historical evidence; do not silently relabel this as native or immutable hosted training.

**Next:** source-match M8's four imported dependencies, commit the complete executed scripts under this lane, run fresh hosted CPU source-locked qualification (with checkpoints and PNG/JSON receipts in immutable Actions artifacts), then test new geometry families with user-level visual/style scores. Leave PR Draft/HOLD and unmerged.
