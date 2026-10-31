# NCR-001E — SmolLM2 independent full-context, HF.generate and cold-process candidate

**Parent:** [NCR-001 #67](https://github.com/techrote/neurounner/issues/67), [conformance #72](https://github.com/techrote/neurounner/issues/72). **Stacked strictly on draft [PR #89](https://github.com/techrote/neurounner/pull/89) head** `714840806e083ad9f69fb1ff6aaaf8dbb6bb7eb1`, itself on source-audit #88 → unmerged protocol #77. The new runner uses **original immutable SmolLM2 hook [PR #83](https://github.com/techrote/neurounner/pull/83) at** `ec0d4a80e15f203a06c26ee9c23858f78976370b`. It neither alters #89's original 3,335-byte run-40436942220 receipt nor re-labels source-only readback as an additional model execution.

## Evidence gap targeted

The original independent SmolLM2 [run 40436942220](https://github.com/techrote/neurounner/actions/runs/40436942220) passed source-pinned zero-readout parity through **128 tokens**, **48** genuine one-by-one cached tokens, future-suffix/request isolation, and **cold adapter re-instantiation on the same still-live frozen model in the same interpreter**. Its receipt and ZIP hashes are preserved in #89. Those measurements do **not** qualify max-context cache, actual cached-vs-uncached HF `GenerationMixin.generate()` scores on a manually nonzero cellular path, or a freshly reloaded model in another interpreter.

This separate candidate performs an actual **new original frozen-model invocation**, not a meta-review. No synthetic diagnostic prompt is a held-out evaluation example.

## Bounded fresh two-Python-process experiment

- Producer independently downloads and verifies every byte of the pinned **269,060,552-byte** SmolLM2 weight and source config/tokenizer inventory from `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`. The original hook is checked out from its exact Git commit and 30-layer/tap geometry is asserted, with protocol/preregistration source hashes.
- Producer checks **exact** zero-output logits at **512, 1024 and 2048** synthetic diagnostic token units. A manually set, **untrained** readout coefficient creates a nontrivial NCA path. For each length, compare all 16 tail-position logits from one full forward against an `N−16` KV prefill followed by **16 genuine one-token** cached forwards, preserving the original `5e-4` absolute/relative thresholds. Assert a new isolated field for every decoder call.
- Independently check 256-unit future-suffix isolation, A→B→A request reset, frozen base tensor bytes before/after and no backbone gradients. Measure full source/checkpoint/runner provenance, run/head identity, process PID, CPU, package versions, peak RSS and wall time.
- Call the original HF `generate()` machinery in **disabled, zero-readout and manually active** modes, each with cache **off/on**; record actual generated IDs, processed-score parity, and require exact frozen/zero-output equivalence. No claim about model quality follows from four diagnostic generated tokens.
- Save only the small **untrained** adapter, source metadata and a bounded logits reference in a SHA-bound handoff. **Consumer** starts a separate Python interpreter on the **same GitHub-hosted machine**, independently loads/re-hashes the entire frozen model/source, rejects changed or missing handoff bytes, reloads the adapter and compares identical reference logits. It additionally checks incorrect model-revision metadata is rejected and restores unchanged original bytes.
- Persist each phase's JSON at start, after source verification and finally, including traceback if a failure occurs. Always upload partial receipts and handoff as a 30-day Actions staging artifact. The repository must **promote raw receipt bytes and hashes** before any durable-evidence acceptance; an Actions artifact is not the archive.

**Environment:** pinned PyTorch CPU 2.14.1, Transformers 4.57.6, Hub 0.36.2; GitHub Ubuntu 24.04, four CPU threads, **55-minute** workflow watchdog. The original SmolLM2 protocol imposes a 3600-second CPU lane ceiling and 12-GiB peak RSS budget; these remain unchanged. The new context lengths are strictly pretraining cache diagnostics and do **not** alter the preregistered 256-token training sample budget or any held-out evaluation fixture.

## Qualification status

**Candidate / not accepted until the exact-head workflow has passed and original raw phase receipts have been archived and independently audited.** A red run is evidence of a failed attempt, not permission to relax the numerical tolerance, change checkpoint, extend the budget or silently skip a case.

Even a green producer/consumer result would be a **SmolLM2 source/runtime pretraining subgate only**, not a final #72 PASS. The required `base_weight_unchanged_after_step`, `nonzero_adapter_gradient_after_warmup` and supervised `finite_state_and_loss` checks remain **BLOCKED_BY_TRAINING_POLICY**. Two interpreters on one Actions host do not prove a cross-host migration or machine reboot. Other three frozen model lanes are independent; the Sprocket Chat checkpoint must never be used for Sprocket Instruct tool/memory source claims. The void.1 ByteTokenizer EOS-ID defect is separately governed by #92.

**Hard policy:** `experiments/ncr-001/protocol.json: train_authorized=false`, **0 optimizer steps**, no backward/training, no held-out model scoring, no matched control experiments, no draft merge or promotion without independent review.

**Implementation:** [runner](../scripts/ncr_001e_smol_maxctx.py), [workflow](../.github/workflows/ncr-001e-smol-maxctx.yml), [offline source guards](../tests/test_ncr_001e_smol_maxctx_source.py).
