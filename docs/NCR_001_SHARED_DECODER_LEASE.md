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

## Promoted source-only Actions evidence (28 October 2026)

The original successful [run 40673634722](https://github.com/techrote/neurounner/actions/runs/40673634722)
tested exact source `4096aa1238ba65851eae78106417131b377d8d82`,
job `122766480553`: ten synthetic cross-wrapper refusal tests, 10/10
PASS on Ubuntu 24.04 (runner `GitHub Actions 1000046432`). The original
artifact ZIP #12808873895 reports SHA-256
`6ad8c0a33e229c02019b203a9e82e743a8c720a4621f8a7ddedc396f039f82b5`.

The superseded [run 40673587828](https://github.com/techrote/neurounner/actions/runs/40673587828)
at exact source `ec9327c892e0655a8ce9464ff34a4ae22ebbbb11`
was **CANCELLED** by the later workflow edit, even though its job's ten
synthetic tests and upload steps were successful. Do not reinterpret it as
a successful completed run. Both full decoded job logs, timestamp-stripped
source-emitted JSON receipts, original workflow snapshot, runner identity,
Git blob hashes, artifact ZIP digests and cancellation status are archived
in [cross-decoder-lease evidence](../experiments/ncr-001/controls-evidence/cross-decoder-lease/).
The ZIP artifacts remain under Actions retention; their member bytes were
**not** independently extracted and compared. The complete job logs themselves
are promoted into Git, and the JSON receipts are deterministically recovered
from the original source-emitted log lines, not silently synthesized from
success claims.

The offline `scripts/ncr_001_cross_decoder_archive.py` verifies original
log/receipt Git blob identities, source SHA-256s, distinct runner/source
heads, ten distinct successful tests, exact four-lane protocol admission and
the original workflow snapshot. The companion adversarial tests refuse
altered original logs, changed source head, false success after cancellation,
fabricated runner and falsified `train_authorized` claims.
