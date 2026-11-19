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

## Accepted input publication — 19 November 2026

The [actual input acceptance](input-acceptance.json), SHA-256 `c6f882d2755f293cfd236452619ff039f161843e0b99aabdf44ab5958fbeefc3`, passes at input bot/tag `783a73ca7c8e97c3e03ef4e56fb5a5b125d58785`. It verifies all thirteen [published input assets](https://github.com/techrote/neurounner/releases/tag/village-001f-pilot-inputs-20261119), all 24 adapter fingerprints from actual bytes, every native workload count, and complete execution history. All six pairs are `fresh_same_job_same_runner`: twelve uninterrupted 0–120 optimizer trajectories, twelve invocations, 72 provenance sidecars, and no copies or unattributed retained steps.

Generation [run 44467827716](https://github.com/techrote/neurounner/actions/runs/44467827716) completed successfully from isolated request revision 2 at `eaeddb22142339c0cd838e588bc09daaa980f652`. The [actual-source receipt](source-check-eaeddb22.json), [release and download records](input-execution/release.json), [complete Actions snapshot](input-execution/actions/capture-receipt.json), and [Sprocket artifact/publication equality](sprocket-input-readback.json) are retained. All 658 Sprocket files match the preliminary native artifact exactly.

Request revision 3 enters the separate judge phase from these accepted inputs. No additional training or generation is requested. At input acceptance, final scientific results, provider replay, final publication acceptance and main readback remained pending. Input acceptance alone does not establish a working mask effect or useful feedback.

Native invocation records establish the retained optimizer hosts and fresh same-runner/process/order claims. Resumed, mixed-host, interrupted and zero-step copy histories remain visible. Final mode independently recomputes integer-delta effects, six-member working-effect criteria, context and feedback metrics, and replays provider outputs offline within the 88/26/114 logical ceilings. Negative or null scientific findings can pass evidence acceptance. Shared-runner execution does not establish numerical determinism, and computation lost before a durable interruption record cannot be reconstructed.

## Accepted final publication — 19 November 2026

The [final publication acceptance](final-acceptance.json), SHA-256 `c47336c618f9666fabe582623107281928fd6aff058ca684088e073f2cb62186`, passes at bot/tag `7ab26c67ad51b5273513e2bb7eeccbc2777d87ae`. It verifies all five final release assets, both complete release/TAR readbacks, all native and adapter checks, the complete two-run Actions history, byte-identical offline provider reconstruction and independent scientific arithmetic. Provider accounting is 53 logical requests, 53 transports and 53 HTTP-attempt records, within the unchanged 114 ceiling. No recovery or new execution occurred during acceptance.

The actual findings are negative: exact worker relevance gain zero, three positive member means, worst change −0.25; Sprocket structural and structure-required grounded/actionable gains zero. See [COMPLETION.md](COMPLETION.md) and the complete raw results. This is a publication readback; its `closure_ready:false` records that main integration is a separate subsequent check. The independent post-merge main readback and closure will be recorded after integration in [issue #256](https://github.com/techrote/neurounner/issues/256).
