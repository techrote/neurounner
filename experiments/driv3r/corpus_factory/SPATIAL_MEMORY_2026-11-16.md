# CyberSand Game Corpus Factory — spatial-memory v6

**Status:** research-only; Draft/HOLD PR #638, not merged. 2026-11-16.

## Question and execution

Do host-observable long-history features help a small trained action-ranking policy escape synthetic navigation loops, beyond the ordinary local 5×5 material/visit observation?

Two binary SGD action rankers were fit for **24 actual CPU epochs each**, from precisely the same TRAIN state/action pairs and the same 19,847 host-legal negative action candidates. They share optimizer settings, world-group samples, legality masking, action candidates and labels. One receives only ordinary local features; the other additionally receives counts, recency, recent-window visit frequencies, oriented-visit counts and two-step-return indicators computed from the **already executed route**. Neither sees privileged BFS distances, held-out world geometry, protected-cell shortcuts, or future outcomes. The behavioral sampler is a separately trained 24-epoch TRAIN-only CPU SGD policy.

The run created 5,296 deduplicated TRAIN decisions, 25,143 legal action candidate training instances, 19,847 hard-negative candidates and **72 complete host-applied TRAIN trajectories** with per-action legal menus, positions and source-world fingerprints. The 12 behavior-training source worlds and 24 corrective-collection TRAIN worlds are distinct. Four synthetic world-generator families appear across partitioned TRAIN/DEV/held-out evaluation; the unseen vertical-slalom generator is deliberately absent from TRAIN and DEV.

## Measured closed-loop performance

Each evaluation used a maximum of 192 applied actions, with each world evaluated across three **correlated family replicas** under each trained model. World-level success is the independent observational unit.

| Synthetic evaluation domain | Worlds | Local-only solved | Long-history solved | Local-only two-step returns | Long-history two-step returns |
| --- | ---: | ---: | ---: | ---: | ---: |
| DEV original | 4 | 1 | 1 | 57.083 | 70.667 |
| DEV horizontal barriers | 4 | 4 | 4 | 0.333 | 0.167 |
| Held-out original | 4 | 2 | 2 | 4.167 | 8.333 |
| Held-out horizontal barriers | 4 | 2 | 2 | 36.417 | 51.333 |
| Held-out vertical slalom (unseen geometry) | 12 | **0** | **1** | 116.306 | 97.972 |

On held-out slalom, the memory model averaged **85.056 unique cells** versus 75.833 for the local-only model. On held-out horizontal barriers, memory hurt coverage (61.583 versus 94.167 unique cells). This is a mixed preliminary result, not an assertion of general superiority or reliable goal solving. The long-history arm also has more features (1,000 versus 821), an intrinsic capacity difference in this representation ablation. Larger, independently generated families and a separate training/architecture control would be needed to isolate causal benefits.

## Partition discipline and provenance

- **TRAIN source:** seeds 1000–1011 (original generator).
- **TRAIN corrective trajectories:** seeds 2600–2607 (original); 7600–7607 (horizontal); 8600–8607 (independent vertical-comb generator).
- **DEV:** 4600–4603 original; 6600–6603 horizontal. No optimizer training on DEV.
- **Held-out synthetic:** 4700–4703 original, 6700–6703 horizontal, 9700–9711 vertical-slalom. No optimizer training or hyperparameter selection on held-out data.

World geometry SHA-256 and source IDs are recorded for all partitions. The TRAIN ZIP includes message rows, per-row provenance, legal hard negatives, complete trajectory receipts and a SHA-256 manifest. Near-duplicate observations are excluded by hashing normalized observation plus host-observed action memory, ignoring family and absolute tick; 4,783 repeated states were discarded. Each split has an explicitly labelled **synthetic-only** identity; neither the DEV nor held-out sets is an official CyberSand benchmark. Synthetic difficulty bands are coarse deterministic route-length/obstacle-density strata, **not calibrated gameplay difficulty**.

## Source and artifact qualification

Pinned parent source Git blobs:

| Source | Blob |
| --- | --- |
| `recovery_curriculum.py` | `fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4` |
| `closed_loop_transfer_probe.py` | `928537a3f02d227c51e36ba61815315d4a2c4ff7` |
| `learnability_probe_scoped.py` | `78d8c899e73d692ffe0540f6ac24dd0e173c38e0` |
| `on_policy_recovery_v2.py` | `3804065af7432b57a52b524de4f4140c54f49e92` |
| `domain_recovery_v3.py` | `c284f09de420f05f95746b141e5d990dddc22cc1` |
| `transfer_orientation_v5.py` | `499f5eb99f187e01ae08697897923faf07ea5b00` |

v6 exact executed code blob: `4c66cd7803603a42789a20fd671cb4518db9cafb`.

- Complete JSON evaluation SHA-256: `862b2f33f933593bbea5466e631456b291e7a359e8f059a93d4d363978b41476`.
- TRAIN-only ZIP SHA-256: `98645c23aff73f294adc99757bf288215bed3338517719581c8238acd6279c94`.
- Two complete independent local executions reproduced **both artifacts byte-identically**.
- `python -m unittest -v test_spatial_memory_v6.py`: **10/10 pass**. Checks cover partition collisions, route budget bounds, parent source pins, executable host action semantics, protection, memory derivation from actual paths, split export rejection and sparse inference equivalence.

Exact machine-readable reports and original trajectory PNGs are kept in the accompanying local evidence archive. The exported corpus is not yet an **immutable GitHub Actions artifact**, so model trainers should not silently treat the local file as source-published production data. No checkpoint binaries are committed. All work remains under `experiments/driv3r/corpus_factory/**`.

## Research interpretation and next step

The result supports investigating history-aware policies, but it is only a **1/12 versus 0/12** slalom success signal and contains regressions elsewhere. Do not promote the memory model. Use this matched TRAIN dataset to test stronger explicit state-space memory or learned persistent-state policies, with model capacity controlled and a newly sealed generator family. Trainer handoff should keep goal completion, unique painting coverage, cycles, immutable-cell safety, and artistic quality as separate metrics.

**Boundaries:** no native/Godot/Workbench/game modifications, no paid external compute, no production changes, no Ready transition, no merge; synthetic solver only.
