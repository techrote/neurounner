# NCR-001F — exact-source same-frozen-host Sprocket NCA/MLP CPU budget prepass

**Research parent:** [#67](https://github.com/techrote/neurounner/issues/67); **conformance gate:** [#72](https://github.com/techrote/neurounner/issues/72); **matched controls:** [#73](https://github.com/techrote/neurounner/issues/73).

**Status at branch creation: EXECUTION/REVIEW PENDING.** This is a narrowly bounded no-training, synthetic forward-only telemetry extension of [draft #155](https://github.com/techrote/neurounner/pull/155), pinned to original head `2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80`. The existing #155 original frozen-model evidence, its source code and its original artifacts remain unchanged.

## Distinct source and variant control

Each matrix job loads **one** original frozen HF checkpoint. Chat is `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668` from `chat/model.safetensors`, original SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`; root Instruct uses separate `model.safetensors` SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. Each has 26 Llama blocks, hidden 1280 and post-block taps [7,16,25]. Both retain their own adapters and original model source/tokens; SmolLM2 and void.1 **are not part of this test**.

The workflow independently hashes the complete original 1,002,207,984-byte checkpoint, model card, configuration and correct variant tokenizer before offline loading. The pinned independent ordinary-role source formatter is commit `2271218d625e1462b2cf218445da975efe17fe32`. Protocol `14ce5fbaa38f8806c8c8738f19c01b91385eebaf` and source inventory `59fff1aed872b5563aa6898171983973e0cdb02c` are immutable Git blobs. All changes relative to #155 are restricted to the four new source/test/workflow/documentation files. Source readback and actual-host runs refuse changed base/source.

## Physical-host measurement contract

`scripts/ncr_001f_postlease_samehost_budget.py` reuses **the same exact #155 model and original NCA/MLP wrapper implementations**, with both wrappers sharing the original decoder-root ownership lock. It demands 658,904 NCA versus 658,432 MLP parameters (delta 472, parameter match within 5%). The reference is the **same original frozen model**, not an independently scheduled instance or a Chat/Instruct cross-comparison.

Only synthetic pinned ordinary-user text is used, sliced to token-context lengths **512 and 2048**. Verify the frozen, NCA-zero and MLP-zero outputs are exactly equal at 64 tokens, and manually perturb one untrained readout weight each (NCA 0.03, MLP 0.05), confirming nonzero finite effect. **There is no optimizer, learned adapter, external tool call or sealed held-out evaluation**.

For each context length, warm the original frozen model, active NCA and active MLP once each; then time **three full inference forwards per path**, with rotated per-repetition order frozen→NCA→MLP / NCA→MLP→frozen / MLP→frozen→NCA. All timing uses `time.perf_counter_ns()` around the same complete model forward; expensive validation and finite-logit scans are outside timed intervals. Retain every raw nanosecond sample, per-path median, NCA/MLP **total frozen-host** median ratio, exact input-token SHA, CPU model/threads, package versions, all source file hashes, run/head and PID.

A single matrix job uses one four-thread host per source variant. This eliminates separate-host confounding **within** a Chat or Instruct path comparison, but does not eliminate CPU scheduler/cache noise, thermal variation, different physical host allocations between variants, or the unequal untrained readout interventions. There are only three timing samples each; do not claim statistical significance. The report preserves whole-process peak `ru_maxrss` but **never attributes it to one adapter**, nor subtracts a frozen reference as proven isolated adapter cost.

Check the original full frozen-model tensor fingerprint unchanged, no frozen gradients, no lingering hooks, and finite untimed postflight. Source-only adversarial tests reject missing paths, wrong budgets, false source/variant identities and cross-model/cross-wrapper lease substitution. The workflow executes the source-only negatives before either full model is loaded, preserves partial failures and uploads raw original logs, JSON and environment identities with `if: always()`.

## Scientific interpretation / retention gates

A successful job would establish a **bounded observation of complete frozen-host forward wall time** under this particular four-CPU hosted environment. It cannot establish measured FLOPs, training-budget parity, full matched-host NCA-vs-MLP training comparability, tool/memory competence, or four-model #72 conformance. The existing model-free adapter-only CPU results in [#98](https://github.com/techrote/neurounner/pull/98) and [permanent original custody #198](https://github.com/techrote/neurounner/pull/198) are *different workloads* and must not be presented as directly comparable latency measurements.

The primary 2048-context masked-batch and long-conversation causal/cache scientific gates remain separate. #155's earlier original same-host 2048 forward/cache/cold reload evidence is retained, not relabelled as this new measurement. Original root-Instruct generated six-token incomplete tool-call prefixes remain negative evidence; no generated tool/memory behaviour is qualified. void.1's real EOS is 258, with historical TRAIN-002b impact **UNMEASURED**.

**Artifacts uploaded by this new workflow are transient until their entire original Actions ZIPs/logs and raw receipts are verified and permanently Git-custodied.** No green workflow alone constitutes final scientific admission. Independent scoped review, dependency order and separate #72/#73/#92 gates remain. The branch must remain draft/unmerged.

`train_authorized=false` · `full_72_conformance_pass=false` · `actual_compute_budget_qualified=false`.
