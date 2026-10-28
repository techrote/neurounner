# NCR-001E — SmolLM2 original full-context and cold-interpreter results (28 October 2026)

**Research:** [NCR-001 #67](https://github.com/techrote/neurounner/issues/67), [conformance #72](https://github.com/techrote/neurounner/issues/72); [draft PR #150](https://github.com/techrote/neurounner/pull/150), strictly stacked on unchanged [draft PR #89](https://github.com/techrote/neurounner/pull/89). No comparative training, optimizer, gradients, held-out dataset, or PR merge. `train_authorized=false` and SmolLM2's full `per_model_gate=false` remain unchanged.

## Original physical source run, not archive-only replay

The original [exact-head Actions run 40651696185](https://github.com/techrote/neurounner/actions/runs/40651696185) at immutable test head `b452d3e5b112507b39be57bbfcac7f0ab721b609` **completed successfully**. Original model job `122693027277` separately invoked Python processes **2707** (producer) and **2758** (consumer) on the **same** GitHub-hosted Ubuntu runner `GitHub Actions 1000046335` / AMD EPYC 7763; 4 CPU threads, Torch 2.14.1+cpu, Transformers 4.57.6. Each independently loaded the pinned **269,060,552-byte** SmolLM2 original safetensors checkpoint with verified SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`, verified pinned original config/tokenizer source files and invoked original reviewed SmolLM2 hook `ec0d4a80e15f203a06c26ee9c23858f78976370b`. No variant substitution.

The producer's manual readout `0.03` is **not trained**. It changed logits (max absolute `0.006818294525146484`); initial zero-readout output remained **exactly equal** to the same frozen source reference at 512/1024/2048 synthetic diagnostic units.

| Frozen 3-tap source/cache check | Original measured maximum absolute error |
|---|---:|
| 512 full vs 496 prefill + 16 true one-token cached | `4.7206878662109375e-05` |
| 1024 full vs 1008 prefill + 16 true one-token cached | `4.76837158203125e-05` |
| 2048 full vs 2032 prefill + 16 true one-token cached | `6.890296936035156e-05` |
| Real HF `GenerationMixin.generate()` manually active, cache off/on processed-score difference | `3.528594970703125e-05` |
| HF generate zero vs disabled original scores (both cache modes) | `0`, exact |
| 256-unit future suffix earlier-logit drift / A→B→A request replay | `0` / exact |
| Different Python consumer, reloaded full original source+adapter vs producer logits | `0`, exact |

All three incremental results remain within the **unchanged 0.0005 absolute/relative cache tolerance**; original HF `generate()` produced the same four diagnostic output token IDs across six original disabled/zero/active × cache on/off modes. That is a **transport consistency observation**, not tool competence or measured natural-text quality. Actual frozen model tensor SHA-256 before/after matched; incorrect adapter revision was rejected. Peak producer/consumer RSS: **3,713,924 / 1,128,312 KiB**; process wall times **86.137 / 1.934 seconds** respectively (not matched throughput benchmarks). No gradients, updates, metrics over held-out samples or new model checkpoints.

## Raw evidence custody and reproducibility

The original GitHub Actions ZIP artifact [#12800615190](https://github.com/techrote/neurounner/actions/runs/40651696185/artifacts/12800615190), **551,739 bytes**, SHA-256 `ae0b26e835f451a513527b7bad3a52e783248300a7a677b1f075a19d658b0f09`, contains the two original complete phase JSONs, one original manually perturbed *untrained* adapter state, its metadata and the small reference logits. Its ZIP bytes and binary PyTorch members remain temporary Actions staging; their hashes/lengths are preserved, not misrepresented as binary assets stored in Git.

The original complete **5,708-byte** [producer.json](../experiments/ncr-001/evidence-review/smolllm2-2048/producer.json) has SHA-256 `958dccd26c6ab9fa0f89f59f0a72c2ad33b7ca5ccd00396c97c026dfd02ca9aa`; the original complete **2,560-byte** [consumer.json](../experiments/ncr-001/evidence-review/smolllm2-2048/consumer.json) has SHA-256 `1d879bcc88a26b7b436766f7c1b583977756fa762e9b685ab30370903310c2ea`. Both have been **losslessly promoted** into the PR branch. A second read via GitHub confirmed both Git blob SHA-1s exactly equal those calculated directly over the original Actions ZIP member bytes, independently of text appearance. The [provenance manifest](../experiments/ncr-001/evidence-review/smolllm2-2048/provenance.json) records all five member SHA-256s, ZIP digest, exact original head/runner/PIDs, source pins, numerical results, negative boundaries and what was not committed.

The independent standard-library [auditor](../scripts/ncr_001e_smol_2048_archive_audit.py) and [negative tests](../tests/test_ncr_001e_smol_2048_archive_audit.py) verify complete original raw SHA/bytes, Git object digests, source/protocol/variant, original cache/no-op/HF score values, process separation and fail-closed qualification. This is **source/readback-only** verification and must not count as an additional physical model invocation. See [audit workflow](../.github/workflows/ncr-001e-smol-2048-archive.yml) for exact-head CI. If archive audit has not passed yet, archive qualification remains pending despite the original physical success.

## Hard remaining admission blockers

**SmolLM2 #72 final gate remains BLOCKED**, exactly like the three other source/runtime-adjudicated model identities. Required `base_weight_unchanged_after_step`, `nonzero_adapter_gradient_after_warmup` and supervised `finite_state_and_loss` checks are **not** satisfied by any manual readout; each requires explicit protocol-authorized learning/update, which remains forbidden while `train_authorized=false`. These two interpreters were on the *same* hosted machine, not a cross-host reboot; no cross-machine identity guarantee is claimed. Formal SmolLM2 machine-readable #72 adjudication and independent reviewer assessment remain future work.

Sprocket **Chat** and **Instruct** remain distinct 1GB checkpoints, tokenizer/control semantics and formal adjudications [#124](https://github.com/techrote/neurounner/pull/124) and [#122](https://github.com/techrote/neurounner/pull/122). Separate void.1 [#131](https://github.com/techrote/neurounner/pull/131) and BYTE-EOS future serialization [#94](https://github.com/techrote/neurounner/pull/94) / [#126](https://github.com/techrote/neurounner/pull/126) stay separate. The historical BYTE-EOS causal effect is **unmeasured**. No NCA/MLP comparative training or draft merge.
