# NCR-001E — fresh reviewer SmolLM2 cache and cold adapter prepass

**Parent:** [#72](https://github.com/techrote/neurounner/issues/72).
**Dependency:** [four-source receipt audit #88](https://github.com/techrote/neurounner/pull/88).
**Exact original model integration:** [SmolLM2 #83](https://github.com/techrote/neurounner/pull/83), reviewed from commit ec0d4a80e15f203a06c26ee9c23858f78976370b.

This is a **separate, fresh actual model invocation**, not an additional copy of the prior Actions receipt. The workflow checks out a frozen implementation source commit in a second directory and runs an independent reviewer-written Python script without changing any implementation files or model weights.

The review is **not full four-model #72 conformance**, and not a learner experiment or training admission.

## Tests

- Recompute full 269 MB SmolLM2 safetensors SHA-256 against the pinned model source at fixed revision, and require exact protocol/implementation tap identity.
- Verify zero-init logit parity for separate 16, 64 and 128 token requests, and left/right-padded two-request batches.
- Verify a five-token greedy generation sequence is identical between base and zero-output NCA.
- Inject a single *manual*, explicitly untrained 0.03 readout weight to prove the cellular path genuinely influences logits; compare a longer 48-position full forward with **incrementally KV-cached one-token** forwards at the fixed 5e-4 tolerance.
- Change the last token and establish strict future-suffix isolation (no earlier logit drift).
- Execute A→B→A requests on the same wrapper with the nonzero NCA and demand exact return to A, detecting accidental cross-request field persistence.
- Save only NCA adapter weights to a temporary checkpoint, instantiate a **new adapter on the same original frozen backbone**, reload and demand identical nonzero-modulation logits; deliberately corrupt the checkpoint's model revision and demand rejection.
- Check frozen base weights never receive gradients and that no optimizer object or update is created. Source training gate remains false.
- Preserve partial failure and host/package/RSS facts even when a test fails.

## Qualification boundaries

Cold reload is a new **adapter instance on the same still-live frozen base**, not a new operating-system process or a fresh source-model load. This qualifies a necessary subcondition, not the complete physical-source reset or cross-worker handoff surface.

No performance claim, CPU speed comparison or NCA training. No Sprocket Chat/Instruct role or tool-memory formatter checks and no void ByteTokenizer execution. All remaining model-lane #72 requirements remain independently blocked until separately reviewed.

The reviewer-owned source is [scripts/ncr_001e_smol_fresh.py](../scripts/ncr_001e_smol_fresh.py). The bounded four-thread actual-host runner is [.github/workflows/ncr-001e-smol-fresh.yml](../.github/workflows/ncr-001e-smol-fresh.yml). Evidence is staged in runtime/ncr-001/smol-independent-fresh.json and needs durable promotion after the run.
