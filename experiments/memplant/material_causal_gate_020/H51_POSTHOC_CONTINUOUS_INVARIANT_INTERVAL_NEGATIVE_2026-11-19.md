# DG020 H51 — complete ReLU-knot invariant interval audit (19 Nov 2026)

**Disposition: exposed-only H51 interval-robustness FAIL. Scientific prospective gate: NOT RETESTED.** H40's preregistered restricted 30/30 PASS remains unchanged. H45's discrete sampled-grid result was valid only at sampled coordinates. Do not promote either as continuous-law generalization.

## Question and immutable evidence

The H42/H45 interpolation studies sampled donor-feature positions. Can an invariant-world receiver still choose the **wrong** action in extremely narrow unsampled intervals between the two genuine trained-donor endpoint costs?

H51 uses the already-exposed, **unmodified** H40 held-out 4,800 world receipts (SHA-256 `5a57c26474d4254275bd1e01b0d7e62c03182ec685906a39750d9e8694eade0d`), eight real H34 trained donor checkpoints representing four source/replicate profiles under two laws, three original H40 trained 209-parameter receivers, and three *negative* H45 same-size knee-repaired candidates. No donor/receiver retraining or checkpoint alteration, no new holdout, laws, topology or paid compute. Original geometry, goal, legal mask, wetness and tile fees remain frozen. No oracle/family/source label is supplied to inference.

For each invariant H40 world (960 per receiver seed) and each genuine donor-source profile, use its fixed two trained-donor estimated wet-cost endpoints to bound a **hypothetical intermediate scalar cost**. The receiver constructs per-tile `(wet, fee, cost)`: wet cost `c+fee`, dry cost `1+fee`, and sums actual frozen 3→16→8→1 ReLU tile-network scores across both 12-tile routes. The input observations are rounded to original float32 before being promoted to float64 for analytical evaluation.

Each receiver's route-margin function of `c` is piecewise affine. Exhaustively enumerate the crossings of the 16 first-layer ReLUs for all wet tiles; within each such interval enumerate the eight second-layer ReLU crossings. Union all route/tile knots, then evaluate the margin at both ends of each affine segment and analytically solve its zero crossing. A negative signed margin over **positive width** witnesses an inappropriate invariant-world action change. This is a real-arithmetic float64 ReLU analysis, not a universal floating-point or physical-law guarantee.

## Full exposed-world results

| Receiver condition | Invariant world × donor-profile intervals with an incorrect interior action | Unique invariant worlds affected | Piecewise-affine intervals checked |
|---|---:|---:|---:|
| H40 original genuine frozen receivers | **568 / 11,520** | **142 / 2,880** | 1,084,896 |
| H45 negative knee-adjusted candidates | **20 / 11,520** | **5 / 2,880** | 1,126,112 |

H40 unique failures by seed: `26011639` 11, `26011640` 56, `26011641` 75. H45: only receiver `26011640` fails, in five `wet_poor_invariant` worlds (four `interleaved_braid`, one `reverse_bifurcation`). All four real donor-source endpoint intervals cover these five narrow cost pockets. Total distinct wrong-interior H45 observations: **5**, not 20 independent world examples.

H45's strongest witness: seed `26011640`, `interleaved_braid`, world index 87, world seed `26011929`. At scalar cost `0.351164215033142`, ideal float64 signed neural margin ≈ **−0.00020485**, with incorrect-action cost interval total width ≈ `0.00010248`. Both actual genuine trained-donor endpoint actions remain correct. These off-grid failures explain why 65 fitted plus 192 later sampled positions cannot certify continuous behavior.

**Independent full-network PyTorch audit:** a second script did no ReLU-knot enumeration, independently loaded the authentic checkpoint tensors and world receipts, submitted all **588** recorded negative scalar witnesses to the full network, and reproduced **588/588 incorrect actions under both float32 and float64 CPU inference**, with **1,176/1,176** matching genuine endpoint checks correct. The maximum absolute signed float64 witness-margin difference against the analytical evaluator was ≈ `1.025e-7`. The two independent scripts, complete original source H50 evidence archive (including genuine checkpoint bytes), full 588 raw witnesses, H51 result JSON, replay JSON/logs and SHA-256 manifest have been preserved in the owner handoff ZIP.

## Claim boundary and next mechanism

H51 is **post-hoc, exposed-only, with synthetic intermediate costs**. No independently trained intermediate-law donor was used. In particular H46's identical-input/opposite-physical-label witnesses make the universal synthetic-alpha *divergent-world* training objective ill-defined; H51 does **not** revive it. Correct endpoint decisions and historical H40 30/30 restricted preregistered PASS remain intact, along with H41's permutation-invariant placement limitation and H48's cross-CPU bitwise drift.

**Smallest defensible next mechanism:** treat the five H45 invariant-world intervals as exposed, hard *invariance-only* regression constraints while also hard-preserving genuine trained-law endpoint correctness, within the same 209-parameter architecture. Do not optimize contradictory intermediate divergent oracle labels. If such constrained fitting is infeasible, record FAIL rather than growing model size. Only an untouched preregistered unseen-family cohort with authentic trained donors, two-direction flips, invariant selectivity, informative sham/zero/shuffled controls and independent review can yield a new scientific PASS.

The sealed H22 exact-HEAD GitHub Actions replay is **not** H51 CI. H51-specific exact-source CI and complete GitHub binary custody remain outstanding.

**Governance:** Existing PR #253 **open Draft/HOLD; never merge or mark Ready**. Only `techrote/neurounner` writable; `techrote/cybersand` and `techrote/DRIV3R` read-only. No new branches/PRs/issues or paid API/GPU usage.
