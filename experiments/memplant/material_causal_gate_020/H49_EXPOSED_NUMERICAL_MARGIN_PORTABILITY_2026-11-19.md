# DREAMGRAFT DG020 H49 — exposed-only numerical margin and portability audit (19 Nov 2026)

**Disposition:** H49 retrospective margin inventory **PASS**; cross-runtime *bit-exactness* and an expanded mechanistic scientific claim **NOT ESTABLISHED**. H40 remains a historically preregistered restricted **30/30 PASS**, not rerun on fresh worlds. H45 remains **off-grid FAIL**. PR #253 remains Draft/HOLD.

## Questions and fixed inputs

After H48 found one synthetic action difference in 7,526,400 route margins across CPU environments, test how many archived H40 and H45 endpoint/interpolation margins are close enough to zero that the *empirically observed* H48 cross-runtime difference could matter. This is explicitly **post-hoc on already exposed H40/H45 data**, with no optimizer, new holdout, synthetic ground-truth claim, law change, receiver size change or model promotion.

The authentic inputs are the unchanged H40 raw counterfactual receipts (4,800 scenes; SHA-256 `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`), H47's original float32 full-network margin archive (SHA-256 `77e96494144c2015dab45ad4823dfbc17056b1c064cdd2e5a793e1c324da7bee`), 8 genuine trained donors, 3 frozen H40 209-parameter receivers, and 3 exposed H45 candidate receivers. H40 receipt-scoped invariants were checked: fixed paired geometry/goals/legal masks, 3 receiver seeds, 1,600 worlds per seed, 640 divergent and 960 invariant scenes per seed.

The comparison threshold is **7.62939453125e-5**, H48's actual maximum observed absolute margin deviation on one distinct CPU replay. It is an **empirical stress window, not a proven floating-point error bound**; it must not be interpreted as safety certification across all possible hardware or compiler paths. Low margins are *possible numerical hazards*, not automatically wrong actions.

## Results

| Archived full-network condition | Genuine endpoint margins | Incorrect original actions | Margins at or below empirical H48 drift | Minimum absolute margin |
| --- | ---: | ---: | ---: | ---: |
| H40 original genuine donor cost | 38,400 | 0 | **0** | 0.00013256072998046875 |
| H40 source-aligned donor cost | 38,400 | 0 | **0** | 0.00014209747314453125 |
| H45 exposed hidden-knee candidate, original cost | 38,400 | 0 | **1,390** | 0.00000762939453125 |
| H45 exposed hidden-knee candidate, aligned cost | 38,400 | 0 | **1,392** | 0.000010013580322265625 |

H45's archived **synthetic** off-grid scores contain **102,943/7,372,800** values at or below the same empirical drift window (11 exact zeros). These remain *synthetic intermediate donor-feature* evaluations, not genuine independently trained intermediate-law donor outcomes, and the H46 non-identifiability obstruction still applies. The majority of low margins are on receiver seed 26011641. The H40 baseline has a genuine-endpoint minimum 1.74 times the observed H48 cross-runtime maximum drift; that ratio is descriptive, not a universal portability guarantee.

**Independent live full-network corroboration in the current CPU environment:** reexecuted the H48 authentic checkpoint/donor audit on the unchanged H48 archive with PyTorch 2.10.0+cpu and NumPy 2.3.5, once at 2 threads and once at 1 thread. Both replays checked **7,526,400/7,526,400** route margins with **zero bit disagreements, zero action disagreements and 0.0 max drift**, verifying reproducibility *on this machine*. The prior independent H48 run on a different CPU recorded **3,406,554 bit disagreements and one synthetic off-grid action disagreement** (H45 seed 26011641, source 29002179-rep0, world index 255, alpha 255/256, +4.76837158203125e-7 vs 0.0). The two environmental observations are retained together, not averaged away.

The H49 script is [`h49_numerical_gate_audit.py`](h49_numerical_gate_audit.py), with original SHA-256 `97edfc6f25b5499f825cfd24e3051a38232d1164f7febb42cf5511b350c4e238` and Git blob `361b7846b6b85a8754ef063475c0b68d1c0463b2`. It requires the **original H48 owner evidence archive**, verifies the receipt and margin-array hashes, checks legal-mask and category counts, independently scores stored original neural margins against oracle actions and inventories their numerical distance from zero. This script is an audit of stored neural outputs; the separately executed H48 runner performs the independent *live network* replay.

Run, after unpacking the H48 archive:

```sh
python experiments/memplant/material_causal_gate_020/h49_numerical_gate_audit.py DG020-H48-EVIDENCE --output h49_numerical_result.json
python DG020-H48-EVIDENCE/h48_cross_runtime_replay.py DG020-H48-EVIDENCE --out your-output-directory
```

The first command verifies H40/H45 margins without inference. The second invokes authentic neural inference but can display CPU-specific float32 bit differences; its historical strict-bit verdict is intentionally **not** silently weakened.

## What follows scientifically

1. **H49 inventory PASS**: reproduced all recorded input hashes and per-seed/aggregate low-margin counts. H40's genuine trained-law endpoints were not near the observed H48 drift window; H45's repaired endpoints and synthetic interior were much more numerically exposed.
2. **Cross-runtime bit-level qualification remains FAIL** because H48's independent CPU reproduced one action differently on a synthetic intermediate example. A fresh exact-HEAD GitHub run of the full H48 evidence is not established.
3. **Prospective causal-selectivity gate NOT RETESTED**. H40's restricted 30/30 result is unchanged, but H41's spatial permutation-invariance and H46's conflicting synthetic label/feature collisions remain serious scientific boundaries. The pilot does not establish generic material-law inheritance or a CyberSand policy.

**Smallest next mechanism/provenance move:** retain H40's frozen endpoint behavior and use an *action-level genuine-law CPU replay with explicit near-boundary exceptions* as a qualification layer; never turn the observed drift into a universal tolerance or excuse a real invariant error. Prioritize complete Git-backed custody of genuine checkpoints and raw receipts, then pinned full-authentic replay and independent methodological review. Do not further train against the H46 contradictory synthetic-interpolation oracle.

**Governance:** only this existing techrote/neurounner Draft/HOLD lane modified. No new branch/PR/issue, merge/Ready, paid compute, or CyberSand/DRIV3R edits.
