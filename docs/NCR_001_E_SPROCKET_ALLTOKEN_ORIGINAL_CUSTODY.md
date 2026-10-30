# NCR-001E — original PR #205 all-2048-token Sprocket evidence custody

**30 October 2026. Permanent original raw evidence, not a new model run.**

Source-bound [physical PR #205](https://github.com/techrote/neurounner/pull/205) has immutable head **0c866f5b65380ef6629a5bf084d017e886b91e52** from integrated main **b085d0cec41473b33d844acbf644b57304526779**. The original postlease source, four-model protocol and model pins are Git blobs **66e6b4d97eea5e917339b8e7bfb5d1c191db4a2d**, **14ce5fbaa38f8806c8c8738f19c01b91385eebaf**, and **59fff1aed872b5563aa6898171983973e0cdb02c**, unchanged. This archival branch cannot change the original source, preregistered fixtures/tolerances or historical failures.

## Original physical result, not a custody model rerun

[Original Actions #40987371706](https://github.com/techrote/neurounner/actions/runs/40987371706) **SUCCESS**, with source-only job **123841492749** (12/12 tests), original frozen **Chat** job **123841738132** (chat/model.safetensors SHA-256 551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97) and distinct root **Instruct** job **123841738103** (model.safetensors SHA-256 1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864). Both original frozen weight files contain 1,002,207,984 bytes but are not interchangeable.

Each source-formatted B=1 unpadded synthetic ordinary-role run compared *all* **2048 genuine incremental one-token KV-cache** logits against one full 2048-forward sequence, frozen 5e-4 absolute/relative tolerance. Chat maximum absolute difference **0.00026702880859375**, root Instruct **0.000110626220703125**, both at position zero. Each ran exactly **ten** independent uncached prefix checkpoints at positions 1, 2, 16, 64, 128, 256, 512, 1024, 1536 and 2048, plus exact earlier-2047 invariance under a final-token mutation, no-op parity and before/after frozen backbone identity. The readout was manually activated, **not trained**; optimizer steps zero. Reports explicitly assert **all_prefixes_run_without_cache=false**. This test is distinct from the B=2 masked final-four KV and separate-interpreter cold adapter results of [#202/#203](https://github.com/techrote/neurounner/pull/203).

## Whole original ZIP and decoded job-log bytes

All six original complete raw streams and a source-bound manifest now reside permanently in Git under **experiments/ncr-001/evidence-review/pr205-alltoken2048-original/**. The three whole original compressed Actions ZIPs are:

| Evidence | Artifact ID | Original whole ZIP SHA-256 |
|---|---:|---|
| Source 12 tests | 12947388112 | 0d5b5a8c99d701a1b6c1df6234a36912c979fd52939b29332542509ed4170c69 |
| Chat physical | 12947358773 | 05d4b24909d612f08abf6dd65f498f71f31daab53b4c3cc3cfc8d0d881a40f36 |
| Root Instruct physical | 12947986971 | 6d519ff309f81dfd805cfc108c926d22eafaebd12a2854fc8fba4377ca199b32 |

Also preserved are full original GitHub **API-decoded job logs** from jobs **123841492749**, **123841738132** and **123841738103** (not mislabeled as raw HTTP compressed logs). The original fixed manifest Git blob is **7d5bb5343198b4d303039a98ecb02cf52fec8264**; six independent immutable ZIP/log blob identities are constants in the primary auditor. The original Actions assets expire **29 November 2026**, while Git custody is durable.

Original first edited-event [#40987308916](https://github.com/techrote/neurounner/actions/runs/40987308916) was **CANCELLED**, not a physical model failure. First archive bootstrap [#40993353927](https://github.com/techrote/neurounner/actions/runs/40993353927) **FAILED** with HTTP 401 due to an erroneous literal backslash before its GitHub token; it did not modify raw evidence. The corrected [#40993581991](https://github.com/techrote/neurounner/actions/runs/40993581991) **SUCCEEDED**: original run/job/artifact identities checked; three original ZIPs and three original decoded jobs downloaded and committed byte-for-byte. The one-shot write-capable workflow was then **deleted**; final audit is strictly read-only.

## Audit and limitations

Execute the source-only primary **scripts/ncr_001e_alltoken2048_original_custody.py** and twelve adversarial tests in **tests/test_ncr_001e_alltoken2048_original_custody.py**. They pin original manifest and six raw Git blobs, independently check raw ZIP SHA-256/member integrity, source and weight identity, complete logs, numerical scope, frozen integrity and no-training status, refusing corruption, missing/swap/replacement and forged manifest/training permissions. The final read-only Actions also verifies original GitHub run/job/artifact identities and fresh byte-identical ZIP/log re-downloads. A green custody check is not new frozen-model inference.

Historical original #161 produced only an **incomplete six-token generated tool-call prefix**: no useful generated tool/memory competence has been admitted. void.1 real EOS integer ID **258** remains corrected only for future labels; historical TRAIN-002b impact is **UNMEASURED**. This archival PR is **draft** pending exact-head technical review and explicit source-integration authorization. Same-account COMMENTED review is accepted for the narrow administrative gate under [owner #72 decision](https://github.com/techrote/neurounner/issues/72#issuecomment-6506010783), but must not be called GitHub APPROVED or scientific admission.

**train_authorized=false · full_72_conformance_pass=false · actual_compute_budget_qualified=false**.
