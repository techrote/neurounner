# Independent Stage F acceptance

The [read-only acceptance tool](independent_stage_f_acceptance.py) is the exact independently reviewed implementation, SHA-256 8066e05169f3fb10048945b6d9158ecffd1c3992360d1d058d688ee823702050. It does not fetch Git objects or API data, load models, tokenize, train, generate or call a provider. The caller supplies a clean fetched checkout, saved API snapshots and downloaded assets. Output reports must be new paths; earlier reports are retained.

The immutable [source-only receipt](source-check-78a72238.json) verifies the protocol frozen at 78a72238a8403c4cddb9e8651242a89e17ee21be: 14 executable dependencies, four contracts, workflow and reconstructed worker/Sprocket plans. Its SHA-256 is 8be9f12235fe6189b5db5cabc4e58faf1913714289d232ac1965b3b261309ce0. Actual input and final acceptance are not inferred from that source check.

## Supplied evidence

- Input release directory: all 13 baseline assets (12 new adapter TARs and complete-input TAR), plus any declared recovery assets.
- Final release directory: all five baseline assets (four JSON reports and complete-evidence TAR), plus any declared judgment-recovery assets.
- Historical references: the 12 C and original-D TARs named in the frozen worker plan.
- Unmodified GitHub API JSON for the input/final releases, complete workflow-run inventory, and each referenced run's run, all-attempt jobs and artifacts endpoints.

The saved Actions directory contains workflow-runs.json and runs, jobs and artifacts subdirectories keyed by run ID. Fetch the release tags and referenced source commits locally before auditing. API inventories must be complete and bounded; missing or truncated history fails validation.

## Invocation

Run the script with --help for all arguments. Source-only mode requires --repo, --expected-head, --mode source and --out. Actual input mode adds --mode inputs, --input-target, --input-release-dir, --input-release-json, --reference-assets-dir and --history-api-dir. Final mode uses --mode final and additionally --final-target, --final-release-dir and --final-release-json. A final main readback requires --require-main and an independently refreshed origin/main reference.

The tool verifies exact release/API digests and sizes, tag commits and publication parents, complete TAR-to-Git agreement, raw custody and recovery history. It checks all 12 native training histories, 300 Q responses, 36 Sprocket outputs, 720 token-audit rows and 288 pair-forward batches. A separate safetensors reader independently checks all new/historical adapter bytes and canonical named-tensor fingerprints without loading a model.

Native invocation records establish the retained optimizer hosts and fresh same-runner/process/order claims. Resumed, mixed-host, interrupted and zero-step copy histories remain visible. Final mode independently recomputes integer-delta effects, six-member working-effect criteria, context and feedback metrics, and replays provider outputs offline within the 88/26/114 logical ceilings. Negative or null scientific findings can pass evidence acceptance. Shared-runner execution does not establish numerical determinism, and computation lost before a durable interruption record cannot be reconstructed.
