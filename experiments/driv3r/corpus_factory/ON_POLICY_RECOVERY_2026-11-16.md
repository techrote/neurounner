# GAME CORPUS FACTORY — On-policy recovery corrections v2

16 November 2026. **Research-only / PR #638 Draft/HOLD.** Only `experiments/driv3r/corpus_factory/**` may change in Git.

## Execution: trained student drives the host into its own failure states

Earlier closed-loop experiments demonstrated that the privileged-geodesic synthetic teacher could train a goal-oriented student, but four world seeds caused indefinitely repeated action/retrace cycles. This follow-up **does not use those previously evaluated world seeds as training data**. It fits a genuine 24-epoch CPU logistic SGD student on **80 original TRAIN worlds** (seeds `1000..1079`), executes that student on **32 further TRAIN worlds** (`2200..2231`) with a host-applied legal action menu, then requests the privileged oriented-geodesic teacher's corrective labels at the states actually visited by the student. No privileged distance field is present in model-visible observations or the exported chat prompts. The resulting correction corpus is deduplicated against the original source rows and against itself (family excluded from state identity). A second 24-epoch SGD student trains on the same source plus twice-weighted corrections. The comparison is **fitted vs fitted**, not model vs unfitted heuristic.

| Training receipt | Count |
|---|---:|
| Original TRAIN teacher decisions, 80 seeded worlds | 21,735 |
| Student-host-applied on-policy decisions, 32 TRAIN worlds × 3 families | 8,843 |
| Student/teacher disagreements | 1,718 |
| Repeated oriented cells in student host trajectories | 2,666 |
| Normalized source/correction duplicates discarded | 5,736 |
| New, trainer-safe correction examples | **3,107** |
| Weighted final model training rows | 27,949 |
| CPU optimizer epochs: base / corrected | 24 / 24 |
| CPU samples seen: base / corrected | 521,641 / 670,777 |

Host action validity and protected-cell exclusions are enforced at application. The corrective-label corpus is separate from model evaluation and carries a TRAIN-only per-row provenance receipt, original observation fingerprint, action and hard-negative candidate ledger. It is **synthetic host-applied training data**, not W0/native records, artwork scoring, DEV data or production provider changes.

## Two independent, unseen evaluation groups

Both groups are disjoint from the 80 source and 32 correction TRAIN world groups, have **24 independent worlds each**, and are run under the same 192-decision budget with three correlated family replicas per world per policy. Episodes terminate on goal arrival or STOP, so looping statistics are measured **before goal arrival**, not artificially inflated by post-goal movement.

| Evaluation generator / metric | Original student | On-policy-corrected student |
|---|---:|---:|
| Original-style generator: independent worlds reaching goal | 21/24 | **24/24** |
| Original-style generator: correlated family rollouts reaching goal | 61/72 | **72/72** |
| Original-style generator: mean two-step returns before goal | 22.861 | **0.319** |
| Original-style generator: mean steps to goal/STOP | 86.083 | **69.181** |
| New **staggered-gate/obstacle-barrier** generator: independent worlds reaching goal | 19/24 | **20/24** |
| New barrier generator: correlated family rollouts reaching goal | 57/72 | **60/72** |
| New barrier generator: mean two-step returns before goal | 33.056 | **20.153** |
| New barrier generator: mean steps to goal/STOP | 96.958 | **92.278** |

**Qualified finding:** adding supervised labels on the student's own failure trajectories dramatically reduced synthetic two-step loops on the original generator and improved original-generator goal arrivals. **Cross-generator transfer was weaker**: one net additional independent world solved, but four barrier worlds remain unsolved (6004, 6013, 6021, 6023), including a regression on 6021. These are 24 independent *world* samples per generator, not 72 independent samples; do not extrapolate the result to the actual game or Sprocket/LFM.

## Reproducibility, source contracts and boundaries

- Exact source imports are Git-blob-pinned: `recovery_curriculum.py` `fcb1c1f5edfd5724e7cbe73fffdd9365e918b0d4`, `closed_loop_transfer_probe.py` `928537a3f02d227c51e36ba61815315d4a2c4ff7`, `learnability_probe_scoped.py` `78d8c899e73d692ffe0540f6ac24dd0e173c38e0`.
- Two complete deterministic executions produced **byte-identical** full result JSONs and correction ZIPs. `report.json` SHA-256 `f7df486fb63ae4053be0b2d23b8445888686a5e53ba105f31f4ace5971f80761`; correction TRAIN ZIP SHA-256 `a3fbb8cfcab70c1e818686aa358b673b09bdc7d057d3515c290282c062fe3231`.
- Focused local regressions: **7/7 PASS** including scorer parity with exact scikit-learn inference, legal host application, independent synthetic barrier geography/reachability, split overlap refusal, teacher/negative provenance, cross-source dedup and complete byte-identical miniature rerun.
- Runtime: Python 3, NumPy/scikit-learn CPU only. No paid API, external GPU, binary checkpoint in Git, primary-model optimizer duplication or production promotion.
- **Local execution only.** Hosted exact-HEAD Actions evaluation, immutable Actions artifact and primary Sprocket/LFM checkpoint transfer remain unqualified. W0 reference data remains EVAL-ONLY and never enters TRAIN.

Reproduce from the corpus directory (with scikit-learn installed):

```bash
python3 -m unittest -v test_on_policy_recovery_v2.py
python3 on_policy_recovery_v2.py --out /tmp/cybersand-onpolicy-v2-new
```

Read-only research input is allowed; **no files outside this corpus directory may be modified**. No merge, Ready promotion or owner physical acceptance claim.
