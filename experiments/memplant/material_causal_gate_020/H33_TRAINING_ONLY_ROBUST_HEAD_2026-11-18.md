# DG020 — H33/H33b training-only robust output-head calibration (2026-11-18)

**Disposition: H33/H33b training-only screens PASS; scientific causal-selectivity gate NOT PASSED.** The latest stringent preregistered prospective cohort remains **H27 FAIL (28/30 cells)**. Do not merge or mark this Draft/HOLD PR ready.

## Research question and fixed constraints

The H32b fully trained 209-parameter receiver had a donor-cluster plateau initialization but H32c's least-squares head recalibration left five errors and one same-law genuine donor-source disagreement among 7,680 previously exposed H27 compact-margin training observations. Can the receiver's eight final-layer weights be calibrated to respect **both** material-law interventions and **all four** genuine trained donor source/replicate pairs simultaneously, without changing the hidden network or world/agent configuration?

H33 uses a linear program over the last eight output weights, minimizing L1 departure from H32c with a positive signed route-margin floor of 0.00001 across the historical H27 training observations × four genuine donor variant pairs × two laws. H33b fixes an L1 departure budget of 0.001 and maximizes the minimum signed training margin. The output bias and all preceding 200 parameters are frozen. Runtime receiver inputs remain only tile wetness, tile fee, and the corresponding *genuine neural donor's predicted local material cost*; no law labels, source IDs, family IDs, oracle costs, oracle actions, or answer/distance information is exposed at inference. Training-only labels use historical oracle supervision, as permitted for supervised training.

Three trained receivers, seeds `26011639`, `26011640`, `26011641`; four genuine trained H21R donor source/replicate pairs with two original material laws; original 2×12 world geometry and legal actions. Every matched counterfactual pair reuses exactly the same observation and receiver state, changing only which trained-law donor feature is supplied.

## Replayed training-only results

| Measurement across 7,680 matched training worlds | H32c | H33 | H33b |
|---|---:|---:|---:|
| Both law arms correct for all genuine donor variants | 7,675 | **7,680** | **7,680** |
| Divergent pairs correctly flip, both actions correct | 3,067/3,072 | **3,072/3,072** | **3,072/3,072** |
| Invariant pairs correct under both laws | 4,608/4,608 | **4,608/4,608** | **4,608/4,608** |
| Same-law donor-source action disagreements | 1 | **0** | **0** |
| Wrong invariant-law action flips | 0 | **0** | **0** |
| Wrong-law swapped donor, divergent pairs both-correct | 0/3,072 | 0/3,072 | 0/3,072 |
| Zero donor, divergent pairs both-correct | 0/3,072 | 0/3,072 | 0/3,072 |
| Illegal actions | 0 | 0 | 0 |
| Worst signed action margin in frozen float32 neural replay | -0.0002956 | **+0.00000954** | **+0.0000191** |

An independent CPU replay (not importing H33 optimization code) reconstructed all 7,680 original H27 training observation hashes, evaluated genuine frozen donor and receiver weights, and retained 7,680 JSONL receipts with every receiver variant's action and logits for each of the four trained donor profiles and both law arms. Replay matched the optimizer's verdict and preserved original training seeds, original source/donor checkpoints and failures.

H33 adjusted the seed `26011641` output-head weights by L1 **0.000233779**; the other two seeds already passed H32c and required no appreciable change. H33b's fixed 0.001 L1 budget could raise seed `26011641`'s worst margin only to about **0.00002012**; the other two seeds reached about 0.000506–0.000508. Thus the limiting seed remains dangerously close to its decision boundary despite zero historical-training errors.

## Scientific limitations / negative interpretation

- **Not an unseen test.** The H27 worlds were exposed, and the oracle labels are part of the training constraints. A training-only PASS is not a prospective scientific PASS. H27's preregistered 28/30 FAIL is not overturned.
- **Not an independent donor-sham generalization test.** All four genuine donor source/replicate pairs are now used in fitting the head. Even though zero/swapped controls fail as expected, these are gross dependence checks, not previously-unseen genuinely trained donors.
- **Very narrow residual margins.** The seed `26011641` robust signed margin is only 1.9e-5 in float32. This is small compared with donor-estimation variation in prior H30 analysis; new donor sources or stronger near-tie tests could fail.
- **Engineered precursor.** The H32b network descends from H32's manually structured donor-cluster plateau teacher. No spontaneous causal rule acquisition or generalizable transplant claim follows.
- The experiment does not test more laws, different receiver sizes, actual CyberSand gameplay, or additional topology. No CyberSand or DRIV3R changes, no paid GPU/API work.

**Next smallest change:** improve limiting seed `26011641`'s hidden activation-knee stability, or train independent genuinely trained donor source checkpoints held completely out of H33 fitting. Establish strong worst-replicate route margins on training-only worlds first. If progressing, preregister a disjoint new world-family holdout and thresholds (both-direction, invariant, new genuine donor sham, zero/shuffled, illegal actions, absolute margin) before generating/evaluating it. Never reuse H27 exposed worlds as prospective acceptance.

## Exact local evidence custody and replay

The complete self-contained archive accompanying this handoff is `DG020-H33-trainingonly-robust-head-20261118.zip` (38 manifest-verified files). It includes: three exact executed source scripts, frozen H32b/H32c and H33/H33b PyTorch checkpoints, original genuine H21R trained donors, original H27 training-world hashes, an independent genuine-neural replayer, all 7,680 raw per-world receipts and logits, optimizer reports, prior H32 archive, and full SHA-256 manifest. Clean extraction reproduced all three versions' results and validated 38/38 source/evidence entries. This document records its provenance; **the archive itself is not in durable GitHub custody**, and H33-specific exact-HEAD GitHub CI replay is **not qualified**.

Executed script SHA-256:
- H33 solver: `71ac10fff72ae1d2b045ec4b78c21c50ce9047724258e16b0b65f775fad23fcf`
- H33b bounded max-margin: `1845123d8c1da46ae5fa236a77f80ab890fb2becd7e965c9f48d28071b1a977d`
- Independent replay: `f8fd45361c6e53f0cf91e61c06d9de99164af9683267616d16878503874dc651`

Receiver H33 checkpoint SHA-256 by ascending receiver seed: `bffb343b328059ee757ec432cdbf4f90576af2622e5c58d14178254fb312a151`, `1d6575a99ffecd47eded71f58c6d6ed4ee99e62a3b1073ab4d29677a77d8509b`, `858644f6e167eefd56532e0153db071755b771568ddc2b5c6bc7d8f5c964c6d8`.

Receiver H33b checkpoint SHA-256 by ascending receiver seed: `dfe5af62e09924dd6556d80dd92c3554f011e6462acb1438b4533d5b01bbc610`, `001114244f2b90adf310c11193df2a9a04e19228db9dcd6108da050088d7e801`, `08e134b68072bac8ce0b551601313a24e3807f88cdfca7cf3c848fac32e7dd63`.

**Research lane:** only `techrote/neurounner` PR #253. Remain **open Draft/HOLD**; no Ready/merge, no new PR/issue/branch.
