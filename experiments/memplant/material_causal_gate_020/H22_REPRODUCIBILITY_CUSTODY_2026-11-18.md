# DG020 H22 — independent receipt-contract and local frozen-checkpoint verification (18 November 2026)

**Disposition:** Reproduction/forensic audit **PASS**; **NOT** a new prospective scientific acceptance cohort. H22's original restricted 30/30 preregistered scientific PASS remains the governing result. PR #253 stays **Draft/HOLD**.

## Exactly what was checked

1. Reassembled the original H22 evidence overlay with the unmodified H21R parent archive in a clean local scratch directory; verified the original H22 locked source has SHA-256 `a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d`.
2. Ran the archived independent PyTorch frozen-checkpoint replayer, using the genuine H21R donor checkpoints and frozen H19 receivers, on all **4,800** sealed counterfactual pairs. Python CPU environment: PyTorch **2.10.0+cpu**, NumPy **2.3.5**. Result: **PASS**, **30/30 cells**, **31/31** original evidence manifest hashes, and **zero** neural logit or action disagreements.
3. Separately authored and ran `h22_receipt_contract_audit.py` (committed beside this document), which requires only Python's standard library and recomputes from raw observations: float32 observation hashes, independent per-law route-cost oracle, legal-mask and route shape, both-direction correctness, changed/invariant decision flags, recorded logit argmax consistency, sham/placebo/shuffled/zero/source-blind action accounting, all 30 threshold verdicts and all 31 archived manifest digests. It rejects duplicate observations and inconsistent negative indices.
4. Independent receipt-contract result: **PASS**, **30/30 cells**, **4,800 unique paired observations**, **1,920/1,920 necessary divergent flips**, **2,877/2,880 invariant both-correct**, **3 inappropriate invariant flips**, **0** foreign same-law changes, **0** dry-only placebo changes, **496/1,920** shuffled-donor divergent both-correct.

## Retained negatives

All three are seed `26011641` in **equal-wet invariant** worlds. They are not suppressed or relabelled:

| Family | Index | Observation SHA-256 |
| --- | ---: | --- |
| thin_center_tail_22 | 14 | `25880f1571179b91941bd573b11504130b1e574985df225ac089d695e0c9e74d` |
| thick_center_tail_22 | 5 | `d2d6ded877657bccfe4714176d26d4093a22e8e26d6c8af97daa032d2abb826e` |
| thick_center_tail_22 | 152 | `135d0a2fdc7400ee12b97ebb19ea32c784949077cdff7e0cc3298435f89c89c9` |

## Custody manifest for archived handoff files

These are original owner-handoff archives, **not** GitHub-hosted workflow artifacts. SHA-256:

- `DG020-H22-wet-only-causal-selectivity-PASS-20261117.zip`: `5a60ca1119a22916bda18ed46c0c6acecf77b4d891f5b1b841854a81533d1ca0`
- `DG020-H21R-fresh-donor-seed-confirmation-evidence-20261117.zip`: `9a7895ea8c149e755a875fb5b765d9a81559e7ff63b62bc7de565a08234c9c3a`
- `DG020-H22-posthoc-methods-ablation-20261118.zip`: `f673f45cd03f9ace7875eac841f61b75a61e8ced48bc8eb226623b5d10fd392c`

The H22 outer ZIP includes the byte-identical nested H21R ZIP and an overlay of sealed H22 source/receipts. Reproduction: unpack the outer archive, unpack its nested H21R ZIP to an empty directory, copy `H22-overlay/` contents over it, and run `python h22_independent_receipt_replay.py`. Then run the repository script `python h22_receipt_contract_audit.py <extracted>/source/h22-evidence`. Do not overwrite the sealed acceptance directory.

## Scientific claim and limitations

This is **only** a second independent-style local verification path over the **already exposed H22 receipt cohort**, not a new pre-registered independent confirmation. The standard-library auditor does not load genuine donor/receiver neural weights and cannot independently verify neural inference: that separate property was verified by the frozen-checkpoint replayer. The archived neural replayer is local and not yet pinned GitHub Actions exact-HEAD-qualified.

In H22 the inherited intervention was a **learned wet-cost forecast coordinate**, with dry forecast and receiver weights held fixed. The material-specific quadrature probe is engineered; the research does not establish a law edit inside one donor's weights or a generalized game-law transfer. The post-hoc exact-cost diagnostic had **zero action differences** on 9,600 arms, limiting the mechanism claim to scalar-cost rule conditioning.

**Outstanding:** GitHub-hosted complete locked H19/H21R/H22 source/checkpoint/raw receipt custody, pinned exact-HEAD CPU GitHub Actions reproduction, and independent external methodology review. These are prerequisites for qualification beyond the current restricted PASS. No new law families, topology, receiver sizes, PRs, or CyberSand/DRIV3R modifications are justified.
