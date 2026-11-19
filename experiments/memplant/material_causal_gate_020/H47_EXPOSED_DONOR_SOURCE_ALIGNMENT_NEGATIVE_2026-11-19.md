# DREAMGRAFT DG020 — H47 donor-source alignment (19 Nov 2026)

**Verdict: exposed-only H47 strict mechanism FAIL.** Original H40 **restricted** prospective 30/30 cell PASS remains historical; H41 spatial/controls objections, H42 intermediate-conditioning failure and H46 exact-input identifiability obstruction remain in scope. No new prospective cohort, no new model training, no general physical-law inheritance claim.

**Preregistration:** [H47 protocol](https://github.com/techrote/neurounner/pull/253#issuecomment-7007133960), [syntax-only correction before inference](https://github.com/techrote/neurounner/pull/253#issuecomment-7007144392), and [post-execution H47a rounding discrepancy disclosure/H47b source freeze](https://github.com/techrote/neurounner/pull/253#issuecomment-7007171697). Full evidence and executable sources in owner ZIP; only report is stored in repository pending full binary custody.

## Mechanism and scientific limitation

Eight previously trained H40 genuine donors supply four two-law source/replicate pairs. Compute source-specific predicted donor wet-route costs using the frozen 5×5 material probes, then align each source's two *predicted* cost endpoints to the pooled **unlabelled** mean of the four source profiles:

`aligned(c,s) = pooled_low + (c - source_low) * (pooled_high - pooled_low)/(source_high-source_low)`.

Neither physical cost target 0.35/2.30, nor world/oracle/answer label, fee profile, family ID or receiver action enters calibration; the cost signal alone is passed to the frozen receiver. **Caveat:** fitting this mapping uses *both* already-known low/high donor endpoints and a source-specific transform. Source agreement after alignment is partly tautological and is not evidence that a single model acquired material-law competence.

Test 4,800 **already exposed H40** matched world pairs, three genuine H40 209-parameter frozen receivers, three older H45 adjusted 209-parameter candidate receivers, four genuine trained donor profiles, all original low/high endpoints and 192 artificial interpolated off-grid positions. The intermediate coordinates are **not genuine trained intermediate-law donors**; H46 previously proved that uncalibrated nominal-alpha target is mathematically non-identifiable from some identical receiver inputs. Strict diagnostic: no genuine endpoint regressions, no invariant flips, no same-law source disagreement, and **zero** synthetic off-grid errors. One violation means FAIL.

## Full-network results

| Metric | Raw | Source-aligned H47b |
|---|---:|---:|
| H40 frozen receiver genuine endpoint errors, 38,400 individual actions | 0 | 0 |
| H45 adjusted receiver genuine endpoint errors, 38,400 individual actions | 0 | 0 |
| Incorrect invariant endpoint changes (both sets) | 0 | 0 |
| Same-law donor-source endpoint disagreements (both sets) | 0 | 0 |
| H45 artificial off-grid errors, 3,686,400 neural decisions | **5,173** fresh local replay | **5,060** |
| H45 artificial invariant off-grid errors | 0 | 0 |
| H45 off-grid donor-source disagreements | 487 | 0 |
| Strict no-regression + zero-off-grid diagnostic gate | FAIL | **FAIL** |

By receiver seed, H45 raw→aligned off-grid errors: `26011639` 757→656; `26011640` 2573→2564; `26011641` 1843→1840. **All 5,060 aligned errors remain divergent-world failures**, so eliminating donor-source variation is insufficient to repair the remaining absolute threshold/cost calibration problem.

### Numerical replay discrepancy — retained, not concealed

The original archived H45 off-grid report counted **5,172** errors (757/2573/1842). Rerunning its **unchanged** original source now produces **5,173** (757/2573/1843), exactly reproduced by the H47b raw endpoint convention. This one-decision float32/environment variation is **not** reconciled bit-exactly across runs. H47a used prematurely float32-rounded donor endpoints and measured 5,171 raw/5,060 aligned; its negative report is preserved. H47b was a disclosed *post-hoc correction on exposed worlds*, not new preregistered evidence.

### Genuine neural control test

Zero-cost **literal zero tensor** and law-invariant neutral (`1+fee`) controls both give zero divergent both-correct outcomes. Swapping the genuine donor-law cost before inference gives **0/15,360** divergent both-correct versus **15,360/15,360** original genuine-pair cases. Deterministic-seed pre-inference random donor-law selection gives **3,874/15,360** divergent both-correct (H40 1,927/7,680; H45 1,947/7,680), near random 25% and not independently persuasive as a sham. Genuine endpoint same-law substitutions and invariant decisions show no regressions.

## Independent audit and immutable evidence

Independently implemented genuine neural donor quadrature and frozen receiver layer-by-layer replay checked **7,526,400 archived float32 route margins**, finding **0 bit differences, 0 action differences, maximum drift 0.0**. Full source+all eight donor and six receiver checkpoint bytes, original H40 raw observations, H47a and H47b negative outputs, both scripts' inference receipts, raw neural logits and control decisions are in the verified ZIP/35-file SHA-256 manifest. This GitHub report **does not claim** complete binary GitHub custody or an H47 pinned Actions run.

Source SHA-256: H47b `1d07d74e5b0ce6eeca7fdbbb1272f846fe54b915dc37e47d4571721ddd31e19a`; independent auditor `a1d31f03fc5961c0ea14d04ea3d3516de6ad1c182e335c47633a552fe19787b`; full controls `974c5a4448979c79333abb4925f7229abcf29abdc1edc5e9a3dc38b362f6326c`. Archived raw margin NPZ SHA-256 `77e96494144c2015dab45ad4823dfbc17056b1c064cdd2e5a793e1c324da7bee`.

## Next bounded step

**Stop fitting a universal nominal-alpha law label to arbitrary synthetic features.** Before any new prospective claim, complete H40 full binary checkpoint/decision custody, pinned exact-HEAD CPU reproducibility and independent methods review of the engineered material-specific donor probe, negative-control power and spatial permutation-invariance. Any subsequent authentic material-law counterfactual must be independently preregistered and preserve matched geometry, legal masks, actual trained donor evidence and invariant selectivity. Do not enlarge receiver, number of laws or world families merely to escape the blocker.

**Research classification: H47 FAIL (training/exposed-only), H40 historical restricted prospective PASS, all general claims HOLD. Existing PR #253 remains Draft/HOLD; never merge/Ready.** No new PR/issue/branch, paid compute or CyberSand/DRIV3R writes.
