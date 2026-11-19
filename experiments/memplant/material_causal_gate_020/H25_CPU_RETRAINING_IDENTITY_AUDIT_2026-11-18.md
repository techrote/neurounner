# DG020 H25 — independent seeded CPU retraining and checkpoint identity audit (2026-11-18)

**Disposition: scientific H25 original restricted gate PASS (unchanged); byte-for-byte training reproducibility FAIL; exposed-receipt action-level regression PASS.** This is a **post-hoc** verification of an already exposed acceptance set, not a prospective cohort, independent empirical confirmation, or H25 exact-HEAD CI replay.

## Inputs and custody

- Existing Draft/HOLD [PR #253](https://github.com/techrote/neurounner/pull/253); starting HEAD `1779fcce2929cc449103fbe83184b7b3dd0debe8`.
- Original owner-supplied `DG020-H25-prospective-causal-selectivity-PASS-evidence-20261118.zip`; original H25 genuine checkpoints and 4,800 raw matched receipts were **read-only**. Original H22 genuine-donor capsule was extracted separately.
- Re-executed pinned `h25_selectivity_repair_trial.py` (SHA-256 `b1a385dc4c9bceea4aa5b3bf2f8f22382db1e9f0af7b646b755f631815a00853`) with `h23_paired_margin_trial.py` (SHA-256 `41475d535ce4005e1f94c57385891b2f90b87f01e5f265d4ee0eec482bfb8358`), source donor checkpoints, original seeds, original 1,100-step training objective, PyTorch `2.10.0+cpu`, 2 Torch CPU threads.
- No new samples added to accepted H25 results; no receiver-size/law/topology change. Same 209-parameter architecture and genuinely trained H21R donors.
- New checkpoint files are **retraining replicates**, never substituted for the locked, original H25 checkpoints.

## Outcome by original receiver seed

| Seed | Original SHA-256 (prefix) | Retrained SHA-256 (prefix) | Max absolute parameter change | Primary/foreign action mismatches over 1,600 paired receipts | Exposed-cell gate |
| --- | --- | --- | ---: | ---: | --- |
| 26011639 | `49a74ad64fbf` | `d601560e3aa9` | 0.000001371 | 0 / 0 | 10/10 PASS |
| 26011640 | `21df8ec5e5a1` | `3d7376d0be6d` | 0.000002056 | 0 / 0 | 10/10 PASS |
| 26011641 | `093750d8e0f9` | `f1e613927f3b` | 0.000009537 | 0 / 0 | 10/10 PASS |

Each of the three training-observation hash files reproduced **byte-for-byte** (2,560 observations/seed). Retrained weights were numerically close but **not byte-identical**, even under the same local PyTorch version and source; this is a real provenance limitation, not permission to rewrite any historical checkpoint hash.

Replaying all 4,800 original exposed matched worlds with the three newly trained checkpoints and both genuine trained donor groups gave **0/9,600 primary action mismatches**, **0/9,600 foreign-source action mismatches**, and **0/4,800 paired-verdict mismatches** against the frozen originals. All 30 original cell criteria still passed on the exposed worlds. Maximum observed primary-logit absolute drift across seeds was `4.19617e-5` and foreign-logit drift `4.38690e-5`; the worst new equal-wet p99 score delta was `0.001991445` against H25 limit `0.010`.

**Interpretation:** decision-level behavior is robust to this small CPU retraining drift on the **already exposed H25 set**. Exact-checkpoint provenance requires preserving and replaying the **original** checkpoint bytes, not regenerating them and assuming they will hash identically. Avoid any CI workflow that expects fresh training to reconstruct the historical checkpoint SHA-256. Use the sealed originals for pinned neural replay, and report regenerated-training stability separately.

## Strict next actions

1. Put the exact original H25 evidence/checkpoints/source into durable GitHub custody; retain every original receipt and negative historical H23/H24 result. Preserve SHA-256 identities before any recompression or chunking.
2. Run **genuine frozen-original-checkpoint** H25 replay in exact-HEAD GitHub Actions, using original bytes and source-hash validation. Existing 12/12 green workflows at prior HEAD were **H22/older** qualification, not H25 CI.
3. Keep stronger donor-control and additive-invariance methodological review separate from the 30/30 historical H25 restricted PASS. Do not extrapolate to arbitrary world mechanics or CyberSand gameplay; avoid new test families/models until the causal selectivity gate and custody are independently qualified.

No new GitHub PR, branch, issue; no merge/Ready; no paid API/GPU and no CyberSand/DRIV3R writes.
