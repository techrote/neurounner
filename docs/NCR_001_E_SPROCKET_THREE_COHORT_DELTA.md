# NCR-001E — three distinct original Sprocket conformance cohorts, no training

**30 October 2026. Source-only bounded #72 scientific delta, not model execution or final admission.** This draft is stacked on [PR #207](https://github.com/techrote/neurounner/pull/207), exact head cb016f6dceb79e8c7c5fb2c032479c4be5f461e7; it reads literal independently validated [PR #204](https://github.com/techrote/neurounner/pull/204), head 16f8610d856cccba27a7b35e5a99b96526652b50, in a separate checkout. No earlier PR is modified. Historical formal Chat #124 and Instruct #122 reports remain unchanged.

## Original provenance and exact scope

| Cohort | Original physical source and Actions run | What was observed | Permanent evidence |
|---|---|---|---|
| 19-unit synthetic NCA-only diagnostic | [#159](https://github.com/techrote/neurounner/pull/159) df8479aae34550cba014d910a73e30e59cd493e2, [40710171622](https://github.com/techrote/neurounner/actions/runs/40710171622) | One *diagnostic* in-memory NCA-only update per frozen variant, finite synthetic loss and nonzero postwarmup gradients. **Not** matched NCA/MLP learned improvement. | Original #160/#195 raw custody and [#194](https://github.com/techrote/neurounner/pull/194) exact-head source-only reconciliation, e5ac7fb0659bd2e640a3007e8cf798820069fc2d. |
| B=2 2048-position synthetic left/right masked inputs | [#202](https://github.com/techrote/neurounner/pull/202) b52af229dbfd0220d8d89f0d18e58733a2c3b56b, [40939386209](https://github.com/techrote/neurounner/actions/runs/40939386209) | **Exactly four** genuine one-token KV increments per row against full 2048; separate Python interpreter loaded same frozen model/adapter on the **same host** and reproduced final-four scores exactly. Short row 1536 real tokens, right gap synthetic **not** publisher wire. | [#203](https://github.com/techrote/neurounner/pull/203) original whole three ZIPs and three complete decoded logs, original numeric/source audit. [#204](https://github.com/techrote/neurounner/pull/204) independently reconciled with learning diagnostic; immutable published source report Git blob 09d8bc81b4568931c964a3c1b070997ae9cba00d. |
| B=1 unpadded ordinary-role synthetic 2048 real tokens | [#205](https://github.com/techrote/neurounner/pull/205) 0c866f5b65380ef6629a5bf084d017e886b91e52, [40987371706](https://github.com/techrote/neurounner/actions/runs/40987371706) | **All 2048** genuine incremental one-token KV logits versus full 2048, **exactly ten** independently uncached full-prefix checks, not 2048 separate uncached prefixes. **No** separate-process cold replay in this cohort. | [#207](https://github.com/techrote/neurounner/pull/207) original whole three ZIPs and three complete decoded logs, pinned immutable manifest and original-archive source-only auditor. |

The unchanged four-model protocol is Git blob 14ce5fbaa38f8806c8c8738f19c01b91385eebaf; source pins are 59fff1aed872b5563aa6898171983973e0cdb02c. Frozen kandivault/sprocket-500m revision eb157d4ddddc4619e661d4a2426bb2dd000fc668: Chat chat/model.safetensors SHA-256 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97; separate root Instruct model.safetensors SHA-256 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864. Shared architecture/revision does not imply shared weights or behavioral competence.

## Actual numeric observations

| Measurement | Chat | Root Instruct |
|---|---:|---:|
| B2 masked-left final-four max absolute logit error | 0.000022232532501220703 | 0.000015735626220703125 |
| B2 synthetic-right-gap final-four max absolute error | 0.000022232532501220703 | 0.000022649765014648438 |
| B1 all-2048 genuine cached-token max absolute error | 0.00026702880859375 | 0.000110626220703125 |

Fixed original atol=rtol=0.0005. B1 maxima both occurred at token position zero. Ten B1 independently uncached prefix positions: 1, 2, 16, 64, 128, 256, 512, 1024, 1536, 2048; exact earlier-position suffix mutation drift zero. B2 2048 is physical row length, not evidence that its shorter row has 2048 real tokens. The three original cohorts are separate: **they do not establish a single physical run exercising all B1, B2, cold-process and learning conditions together**.

## Read-only fail-closed source audit

The source-only new auditor at scripts/ncr_001e_sprocket_three_cohort_delta.py rechecks the complete original #203 and #207 ZIP/log Git-blob and SHA-256 custody through their respective original primary auditors. It reads a separate literal #204 checkout, requires fixed commit, fixed published report Git blob, current unchanged four-model protocol, and preserves precise variant-specific numeric/scientific limitations. Sixteen adversarial tests refuse lane swaps, extra fictional cached steps, false 2048 uncached checks, changed physical errors, incorrectly declaring synthetic padding as source wire, absent cold replay or postwarmup gradient, altered original run, false scientific pass and reopened training. The derived [per-model decision](../experiments/ncr-001/adjudication/sprocket-three-cohort-source-delta-20261030.json) is committed in Git and regenerated/compared at exact-head CI.

The new workflow does **not** repeat #204's older diagnostic-report regeneration from original #194; that was verified by [Actions 40946386044](https://github.com/techrote/neurounner/actions/runs/40946386044). This workflow independently reruns the primary B2 ZIP/log custody auditor and B1 ZIP/log custody auditor and verifies the #204 published report's Git blob. No new physical model inference, optimizer, HF download, training comparison, heldout access or original evidence rewrite. CI log/artifact is temporary supplemental evidence; original immutable complete Actions ZIP and decoded job-log bytes remain permanently Git-custodied in both parent branches. Previous original failures, cancelled jobs and corrected bootstrap failures remain retained in their original PRs.

Original root-Instruct [#161](https://github.com/techrote/neurounner/pull/161) generated an **incomplete six-token tool-call prefix**: this is negative evidence, no useful generated tool/result/memory competence. void.1 corrected true EOS integer **258** has historical TRAIN-002b effect **UNMEASURED**. SmolLM2 and void.1 have separate #72 gates; matched #73 learned NCA/MLP behavior and actual compute-budget parity remain unqualified.

The owner-accepted same-account COMMENTED technical reviews are administrative, not GitHub APPROVED, another independent human, final scientific #72 admission or merge/training permission. No draft dependency merge is authorized.

**Decision:** BLOCKED_FULL_72_SCIENTIFIC_CONFORMANCE__THREE_DISTINCT_SOURCE_COHORTS.

train_authorized=false; full_72_conformance_pass=false; actual_compute_budget_qualified=false.
