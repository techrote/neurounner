# NCR-001E SmolLM2 shared frozen-backbone isolation — candidate

**Scope:** source-only causal/reference isolation hardening for the SmolLM2-135M Instruct NCA candidate in PR #83, parent #67 and independent gate #72. This candidate is stacked on exact #83 head `ec0d4a80e15f203a06c26ee9c23858f78976370b`. It does not apply to Sprocket Chat, root Sprocket Instruct, void.1, or the separate matched-MLP wrapper in PR #151.

## Observed original-code exposure

The original `FrozenSmolLM2DepthNCA._instrument_lock` is **per adapter**. Constructing adapters A and B over one frozen Python model object yields independent locks. Both may install pre/post/depth hooks to that same model, causing overlapping NCA residuals or a nested decoder state error. Worse, original `enabled=False` paths go directly to `self.backbone()` or `generate()` without taking any lock; a nominal frozen reference run can execute while another adapter's NCA hooks are installed. This is source-verified independently of any original-checkpoint runtime result and parallels the conventional MLP exposure separately addressed in draft PR #151.

The correction installs **module-scoped weak-keyed per-backbone nonblocking locks**, keyed on the Python model object's identity; both enabled and disabled `forward` and `generate` claim ownership. Original per-wrapper lock, decoder prehook, three ordered taps [9,19,29], field state scope and hook detachment remain unchanged. No persistent token, layer, cellular field, model weights or training data are stored in the lock registry.

## Qualification boundaries and reproduction

- `tests/test_ncr_001e_smollm2_shared_backbone.py` uses an existing **synthetic** 30-block SmolLM2-shaped decoder, two wrappers sharing the same backbone, and a second independent backbone.
- It verifies identical lock identity for shared models, rejection of enabled/disabled `forward` and `generate` overlap, rejection of same-wrapper nesting, independent model permissiveness, and complete cleanup after truncated decoder failure.
- The thread/event regression holds a real **toy hook-installed execution window**, then attempts a disabled-reference call through a second adapter: the reference must refuse rather than run under foreign hooks. The test verifies subsequent hook detachment and recovery.
- `.github/workflows/ncr-001e-smol-shared-backbone.yml` checks out **the exact candidate head**, verifies unchanged `protocol.json` blob `dddfb416b765ef6cf23e1cdd326f8834f9f1417c` and `source-pins.json` blob `59fff1aed872b5563aa6898171983973e0cdb02c`, asserts **all four identities** and `train_authorized=false`, runs only synthetic CPU tests, and uploads the original complete test log even if the check fails.
- The earlier real-host #83 receipt and subsequent #89/#150/#152 results remain **historical and unchanged**. This patch has not established real-model causal/cache conformance, separate-process cold adapter acceptance, source/weight new execution, optimizer or loss gates, or behavioral uplift.

**Open limitations:** This is a local guard for wrappers implemented by `ncr_001_smollm2.py`, **not** a common cross-module ownership registry. In particular an NCA and a separately implemented matched-MLP wrapper around one Python backbone may still overlap; do not claim that PR #151 and this patch interoperate or safely share an instance. Other three NCA modules need their own review or a separately reconciled single shared registry. Direct `backbone(...)` bypassing a wrapper also remains out of scope and must not be silently treated as protected.

Keep this draft pending independent source/runtime review. No fixture, tolerance, checkpoint, model revision, architecture, adapter geometry, trained weights or preregistration amendment; **no comparative NCA/MLP training**, no test/optimizer step here. `train_authorized=false` remains authoritative.
