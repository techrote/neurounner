# DG020 H62 — frozen evaluator dependency custody (before H62 neural holdout inference)

**Status at record creation: NO H62 prospective neural inference.** This record supplements, but does not alter, the pre-existing H62 [preregistration](H62_PREREGISTRATION_BEFORE_NEURAL_INFERENCE_2026-11-19.json). All thresholds, cohorts, model bytes and frozen evaluator source are unchanged.

## Verified frozen inputs

- H62 evaluator Python, decoded SHA-256 `d4f61eb0ba61f034df55caa30e92f95e27ad0a1b241e2b004ee000e701722b51` (Git custody `h62_preregistered_causal_selectivity.py.gz.b64`).
- H62 genuine trained receiver seed 26011639, decoded checkpoint SHA-256 `cac5a05ff4d173bedaaa211f2b41697021ab59195d51d44ebd4fd90683779ccd` (Git custody `h62_candidate_26011639.pt.gz.b64`).
- Unmodified H56 engine SHA-256 `f8c059eec92215af7cbbe4752a15733667ca75dbadf2147b6ed6a8810572b871`; unmodified H55 engine `2909ee6ca0e604f866721a6081dec55fd0007f0e8c82a4cd65e2d2c57cbe2815`.
- The frozen H62 script imported `fast_margin` from `h61_training_only_lp_interval_audit`, but the historic H61 helper module was not present in this branch or owner evidence. Before any holdout inference, a **new, declared compatibility shim** was committed under that import name, with **exact source SHA-256 `b38190bbf7ca51485d0f96f524f0eef613608f71a7c9d00f4384e540818d34d0`**. It delegates only to the earlier H51 analytic float64 ReLU-margin implementation `margins`, without labels, fitting, or access to acceptance data. It is not asserted to be a recovered original H61 training script.
- On three **H40 exposed-world** observations, comparison of independent H51 float64 route margins against the unchanged PyTorch full-network float32 model at five fixed cost inputs each gave maximum absolute discrepancies of `7.2988e-6`, `8.2084e-6`, and `7.7951e-6`; all corresponding action signs agreed. These are only dependency checks on exposed worlds.

## Pre-inference geometry-only preflight

Running the exact frozen H62 Python with the shim imported reproduced **2,400/2,400** newly generated worlds and the exact registered world fingerprint `17fb1be3cef0342454a83450d8593fb2b4dcb2bd3a85a868cbf122d465a6b879`. The collision check excludes H40/H54/H55/H56 and H59/H60 preflight world hashes. This did NOT invoke neural inference on any H62 holdout.

## Qualification discipline

The first H62 neural holdout run must use the frozen H62 source and checkpoint, unchanged H55/H56 engines, same donor checkpoints and the declared shim above. No post-hoc evaluator edits or threshold changes are permitted. If the shim or numerical path creates a failure or makes interpretation ambiguous, record that limitation explicitly; do not replace a failing held-out cohort.

H60 remains **NOT EXECUTED** due to missing registered artifacts. H58 and H59a negatives remain unchanged. The only valid prospective scientific claim is a restricted additive-routing one if **all 45 nonpooled endpoint cells AND all 17,280 invariant-cost intervals** pass and independent float64 qualification supports the neural decisions. Cross-hardware portability and general game-policy inheritance remain separate, unqualified questions. PR #253 remains Draft/HOLD. No paid compute or other repository modifications.
