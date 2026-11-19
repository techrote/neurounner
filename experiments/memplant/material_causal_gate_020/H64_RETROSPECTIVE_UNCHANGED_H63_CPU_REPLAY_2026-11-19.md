# DG020 H64 — unchanged H63 trained-neural CPU replay and action-portability audit

**19 November 2026 · techrote/neurounner · existing Draft/HOLD PR #253 only**

**H64 retrospective replay: PASS at action level. Physical cross-CPU portability: NOT QUALIFIED. Prospective causal-selectivity gate: NOT RETESTED.** The original preregistered H63 restricted 45/45 PASS remains the scientific outcome. This audit neither selects a receiver nor introduces new holdouts.

## Provenance and method

- Recovered the original H63 self-contained ZIP, checked ZIP integrity, validated **53/53 SHA-256 manifest entries**, and unpacked nested H62/H56/H52 original checkpoints and scripts.
- Verified exact H63 source SHA-256 e2ed326dc96e2cc4f08c6513f0011ca20f578ec833d8ffe802dc68be33c8f0c0, H62 evaluator d4f61eb0…, H56 evaluator f8c059ee…, H62 receiver cac5a05f…, H52 receiver b1662812…, and original receiver acec1cc9… against original archived bytes.
- Ran the **unchanged preregistered H63 evaluator**, including fresh inference from the eight genuine trained donor checkpoints and three frozen 209-parameter receivers, using the original 2,400 worlds/45 cells. No training, fitting, source/threshold changes, law families, model-size or topology expansion.
- Execution environment: Linux 6.18.57 on virtualized AMD EPYC 9V74; Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5; constrained to single CPU inference thread per original H56/H63 evaluator. This is a fresh CPU execution, **not proven to be a different physical CPU**: the H63 original-run physical host identity was not preserved.
- Compared all raw receipt fields under exact receiver_seed, family, bucket, index alignment, including world hashes, oracle decisions, controls and genuine donor estimates. Only neural_route_margins differed numerically. Both H63 and H64 executed the same 17,280-source-interval invariant-cost calculation with zero failed intervals.

## Recorded result

| Quantity | H64 |
|---|---:|
| Worlds / receiver-world receipts | 2,400 / 7,200 |
| Frozen trained-neural endpoint decisions | **57,600** |
| Changed neural actions | **0** |
| Changed float32 numeric margin values | **24,917** |
| Maximum absolute numerical margin drift | **4.57763671875e-05** |
| Mean absolute margin drift | **3.4526735544e-06** |
| Smallest correct replay signed margin | **+1.9073486328125e-06** |
| Original H63 decisions inside historical H48 drift window | **2,604** (all receiver 26011641) |
| Restricted acceptance cells | **45/45**, unchanged |
| Incorrect continuous invariant-cost intervals | **0/17,280**, unchanged |

By receiver, the changed margin values were 11,889 (26011639), 11,100 (26011640) and 1,928 (26011641), with **zero action differences for all three**. The numerical drift is real: the replay's entire raw JSONL SHA differs from the original even though every categorical decision, true oracle, control, trained-donor estimate and world matches.

Original H63 18,848,697-byte raw receipt SHA-256: 48170223daf0b906b5a3d4c44cf70c05398aedafd9a9ac4ba45ba549b5ee6c48.

H64 newly inferred raw receipt SHA-256: fdb8b8d1bdc17fb16aaa1be9b59883e83f5a2dd5574cdf3442e3bd7477069928.

Exact executed concise comparator SHA-256: 98e789024addf5130b29d51d829a19119445df6be353ceab7be82869b7977bb9 (Git blob b0d51e365c80789116c02b80927cbee55db35970). Machine-readable comparison JSON SHA-256: 7f1167b69c43fb20cc2f3e56839ff461c8c45770519d67203cdad1dbc3c3d219.

## Scientific interpretation and next minimal intervention

**H64 action replay PASS, not a new selective-inheritance PASS.** No H63 neural endpoint actions or invariant-cost interval verdicts were overturned. Yet the smallest action margin is much narrower than the largest drift observed on other examples: one CPU run cannot bound all alternative backends. Physical host separation from the original H63 run remains unverified, and the new raw receipt bytes are deliberately preserved rather than rewritten to match old records.

The next justified infrastructure step is an **exact-HEAD pinned, CPU-only H63 neural replay** that stores both source/destination hardware identifiers, source/checkpoint hashes and full raw decision receipts as retained artifacts. Ensure original binary evidence is placed under reproducible GitHub custody first. If genuinely separate hardware ever changes an action, retain the failing case and restrict the next model intervention to training-only margin stabilization of receiver 26011641, followed by an entirely new preregistered held-out scientific gate. Do not tune against already revealed H63 worlds.

Historical negatives (H46, H54 INVALID, H58 FAIL, H59a FAIL/INVALID, H60 NOT EXECUTED) and the additive-routing restriction remain unchanged. No general material-law memory transplantation, spatial-planning or CyberSand integration claim.

**Governance:** PR #253 is open Draft/HOLD, unmerged and not Ready. No new PRs, issues, branches, paid compute, or changes to CyberSand/DRIV3R.

## Reproduction

From the original H63 owner ZIP root, unpack the nested H62 ZIP into h62_import, then nested H56 ZIP into the root. Put the original H62 candidate at h52/h61_training_only_global_knot_lp_candidate.pt. Copy the archived H62 source, H61 interval helper and H51 interval verifier to the root; copy the H55 and H56 source files from h54/ to root. Then run:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python h63_family_disjoint_selectivity.py evaluate --out h64_local_output
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python h64_h63_action_replay_comparator.py

The comparator requires the original h63_output/ receipts and full original H63 evidence chain; it never manufactures original evidence. Preserve both raw JSONL files and their distinct SHA-256s. The H64 owner supplement includes executed logs, raw replay receipts, exact comparator, output, and hash manifest.
