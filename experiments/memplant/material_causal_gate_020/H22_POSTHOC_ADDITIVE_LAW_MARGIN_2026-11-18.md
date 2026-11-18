# DG020 H22 — additive law-margin residual diagnostic (18 November 2026)

**Disposition:** H22 original preregistered **30/30 restricted PASS remains unchanged**. New **post-hoc additive-score-invariance diagnostic FAIL**. This is not an H23 cohort, not external review, and not a new scientific gate. No training, topology, law family, receipt, or checkpoint was changed.

## Exact evidence and procedure

- Source: original sealed **4,240,821-byte** H22 compact genuine-neural capsule, SHA-256 `47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667`; original prospective H22 evaluator source SHA-256 `a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d`.
- Two genuine donor networks per law, independently frozen at seed 29002117, re-probed using the original 5×5 texture quadrature. Three frozen H19 receivers (209 trainable parameters apiece). The script verifies all capsule manifest hashes, donor and receiver checkpoint identities, 4,800 original raw float32 observation hashes, oracle actions, every original neural action and both-arm logits. Max neural-logit replay drift: **0.0**.
- Defined signed route advantage `A_l = score(route1,l) - score(route0,l)` where scores are negative learned route costs. Exact additive physical-law prediction for the two **existing** law arms is `A_1 - A_0 = (wet_count(route0) - wet_count(route1)) × (2.30 - 0.35)`. In equal-wet worlds this must be zero, **even if the action remains unchanged**.
- Only previously exposed H22 worlds and two existing law endpoints were tested. No new world family, law, model, checkpoint or prospective test condition. `1e-4` is an exploratory numerical cutoff for nonzero score drift, **not a preregistered threshold**.

## Numerical results

| Post-hoc check | Result |
|---|---:|
| Original sealed world pairs / cells | 4,800 / 30 |
| Original H22 negative decisions reproduced | 3 / 2,880 invariant pairs |
| Wrong decisions after oracle-known exact cost-feature substitution (diagnostic only) | **0 differences** across 9,600 individual actions |
| Equal-wet pairs exhibiting absolute spurious law-induced margin shift > `1e-4` | **317 / 960** |
| Equal-wet maximum absolute spurious law-induced margin shift | **0.0333786** |
| Equal-wet 95th / 99th percentile absolute shift | **0.0128301 / 0.0226279** |
| Exact-cost diagnostic: equal-wet shifts > `1e-4` | **318 / 960** |
| Across all 4,800 pairs, maximum absolute error in expected law-response of route advantage | **0.0333786** |
| Across 9,600 arm-observations, maximum absolute route-advantage error from oracle | **0.0357694** |

The original three inappropriate flips all have **equal wet-cell counts**, so the *true* law effect on the route advantage is exactly zero. The frozen receiver instead exhibits spurious law-induced shifts:

| Receiver seed | Fee family | Receipt index | True route-advantage delta | Neural route-advantage delta | Original oracle actions / neural actions |
|---|---|---:|---:|---:|---|
| 26011641 | thick_center_tail_22 | 5 | 0 | **+0.0148249** | `[0,0]` / `[0,1]` |
| 26011641 | thick_center_tail_22 | 152 | 0 | **-0.0333481** | `[1,1]` / `[1,0]` |
| 26011641 | thin_center_tail_22 | 14 | 0 | **+0.0214567** | `[0,0]` / `[0,1]` |

Retained original SHA-256 identifiers, respectively: `d2d6ded877657bccfe4714176d26d4093a22e8e26d6c8af97daa032d2abb826e`, `135d0a2fdc7400ee12b97ebb19ea32c784949077cdff7e0cc3298435f89c89c9`, `25880f1571179b91941bd5733b11504130b1e574985df225ac089d695e0c9e74d`.

## Mechanism and minimum bounded correction

The receiver's learned *tile cost aggregator* is nearly but not perfectly additive in inherited wet cost, material and tile fee. These interactions can change the **relative route margin** under a law swap even when route wet counts agree. Exact scalar material costs yield the same 9,600 actions and comparable spurious margin drift, so the three retained errors are **not resolved by improving donor cost accuracy**. They are receiver scoring/conditionality defects.

**Smallest proposed mechanism change (not executed):** keep the 209-parameter receiver, existing two laws and same 2×12 task, and add a *training-only* additive-law-advantage consistency loss on paired worlds:

`L_delta = mean(((A_expensive - A_cheap) - (n_wet_route0 - n_wet_route1)*(wet_expensive - wet_cheap))**2)`

On equal-wet invariant pairs this directly penalizes any route-margin movement due solely to law change; on divergent pairs it calibrates the sign and magnitude of the necessary law response. Continue supervised action/cost loss, use both divergent and invariant counterfactual training pairs, preserve same-law/sham controls, and preregister a **fresh** holdout before retraining or evaluation. Evaluate real donor-to-receiver conditioning, not oracle-cost substitution, for any eventual PASS. No unapproved training or new acceptance run is claimed here.

**Reproduce:** With the exact Git-tracked H22 capsule reassembled as in the existing H22 genuine-neural CI workflow, run `python -B experiments/memplant/material_causal_gate_020/h22_posthoc_additive_margin_audit.py --capsule /path/to/h22-capsule.zip --out /tmp/h22-additive-margin-audit.json`. The script refuses archives with incorrect SHA-256, verifies all internal manifest entries and checkpoint identities, independently replays the original neural predictions, and retains the three negative receipts. A nonzero numerical margin drift is an expected **diagnostic FAIL**; the process returns success if the evidence is faithfully replayed.

**Governance:** existing neurounner PR #253 Draft/HOLD only, never merge/Ready. CyberSand/DRIV3R read-only; no paid compute.
