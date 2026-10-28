# NCR-001 — unified frozen-decoder ownership composition (no training)

Parent #67; conformance #72; parameter-matched control #73. This draft composes
the four *independent* NCA source variants with the conventional gated-MLP
wrapper on the exact [PR #151](https://github.com/techrote/neurounner/pull/151)
stack. It does **not** change scientific protocol, held-out evaluation, model
checkpoint paths, serialization, tolerances or `train_authorized=false`.

## Original source custody

| Component | Original PR | Exact original source blob |
|---|---|---|
| SmolLM2 NCA | #153 (based on #83) | `722d3b52c9661be7add107ed890989de911f081c` |
| Sprocket Chat NCA | #86 | `82192acf80ba2ddfe9db9e34eaa5259c657847a8` |
| Sprocket root Instruct NCA | #84 | `e21fd8c63ca5b2de5635f8e3b517651ec36d41a6` |
| void.1 Qwen3 NCA | #85 | `3640624df6dbd994dc11b3ca5cd759bdef8757dd` |
| Four-lane gated MLP | #151 | `79cc9293d32bd97ed2441d05cd161757b14e7b5d` |

The new `scripts/ncr_001_backbone_lease.py` is one process-global weak-key
lock registry indexed by the exact **decoder root**, not separate per-wrapper
registries. It excludes simultaneous enabled/disabled NCA/MLP operation,
including two different outer LM containers sharing a decoder. NCA direct
hook-window calls now also acquire the lease. Hook installation/teardown,
three ordered taps and per-forward depth-state reset remain unchanged.

The source tests use synthetic, zero-init model layouts only, with deliberate
active-window contention, actual overlap between threads, same-decoder aliases,
independent decoders and exception cleanup. They make **no optimizer steps**,
perform no comparative learning or original 135M/500M/90M model execution and
do not consume held-out or natural-language evaluation fixtures.

## Limits and next gates

- **Not new #72 model evidence.** All existing real frozen-model/cache/cold
  receipts refer to their *old original hook source blobs*. These modified
  NCA/MLP wrappers need fresh independently run original-model requalification.
- **Not independent review.** Review integration and exact source state before
  promoting this composition to its sibling PRs.
- Ownership covers cooperative wrapper calls within one Python process;
  direct unwrapped backbone calls, external hooks installed outside the
  coordinator and interprocess access are not protected.
- The same Python decoder model object is never to be used concurrently for
  NCA and matched-MLP experiments, including `enabled=False` reference calls.
  Different checkpoint/model instances retain independent leases.
- Three protocol-required learning/update/loss checks still keep all four
  #72 lane gates **false**. Final comparative training remains prohibited.
- BYTE-EOS-001 #92 future-only serialization is unmodified and remains a
  separately reviewed prerequisite for any void.1 supervised study.

PR and workflow test result links belong in a later exact-head evidence
checkpoint; green pre-existing #151/#153 CI cannot validate this new source.
