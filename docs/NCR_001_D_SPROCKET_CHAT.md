# NCR-001D — Frozen Sprocket Chat NCA hook prepass

**Issue:** [#71](https://github.com/techrote/neurounner/issues/71) · **Parent:** [#67](https://github.com/techrote/neurounner/issues/67) · **Draft PR:** [#86](https://github.com/techrote/neurounner/pull/86). Branch stacks on the unmerged common NCA core #82 and provisional protocol #77.

## Exact source and experiment identity

This lane uses the **chat subfolder**, not the Sprocket **root Instruct** checkpoint, of kandivault/sprocket-500m at fixed HF revision eb157d4ddddc4619e661d4a2426bb2dd000fc668.

- Actual downloaded 1,002,207,984-byte **chat/model.safetensors** was independently hashed and matched SHA-256 **551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97**.
- Its sibling root Instruct model, **1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864**, was **not** loaded. Same Llama-26 architecture does not imply same checkpoint or baseline behaviour.
- Chat-specific config/tokenizer files were also hashed from downloaded bytes. Source attestation is recorded in the archived full receipt and its provenance.
- Model: 26 Llama decoder layers, hidden width 1,280, three NCA post-block taps [7,16,25], 501,090,560 frozen parameters and 658,904 trainable NCA parameters.
- The adapter maintains explicit per-token field state across three depth taps of **one decoder forward** and discards it when the forward completes; no module-global or cross-token cellular memory.

## Corrected hosted result

[GitHub Actions 40426220940](https://github.com/techrote/neurounner/actions/runs/40426220940) **PASS**, including seven offline hook lifecycle/source-identity tests and a real pinned frozen Chat model prepass. A prior run [40425730187](https://github.com/techrote/neurounner/actions/runs/40425730187) also passed numerically but had a misleading inherited SmolLM2/Instruct artifact filename. Its original evidence is preserved, while the correctly named later run is the **authoritative** Chat source receipt.

| Frozen model/source check | Measured |
|---|---:|
| Zero-readout original vs modified logits | Exact equality, max 0 |
| Initial padded two-request logit parity | 0 |
| Short greedy generation parity | Identical |
| Readout-gradient L1 without optimizer steps | 2,592.0615 |
| Manually nonzero but untrained NCA logit change | 0.00728989 max |
| Nonzero NCA full vs incremental KV-cache max error | **0.0000281334**, within fixed 0.0005 bound |
| Future-suffix causal invariance | 0 maximum difference |
| Peak process RSS | 3,312,512 KiB |
| Adapter optimizer steps | 0 |

Runtime: four CPU threads on Intel Xeon Platinum 8370C, torch 2.14.1+cpu, Transformers 4.57.6.

Full exact [4,003-byte raw JSON receipt](../experiments/ncr-001/sprocket-chat-preflight/receipt.json) SHA-256 **62992ee8a63c40c9cd1ad8101e90131ff957f0fe268440cd06fc253157a3a48f** and [source-run provenance](../experiments/ncr-001/sprocket-chat-preflight/provenance.json) preserve the measured numeric records, source revision, original artifact ZIP digest, runner and prior-run correction. Subsequent CI verifies committed receipt bytes against that hash.

## Qualification limits

This is a **short frozen-model integration prepass**, not a learned NCA or final independent #72 qualification. The nonzero readout was manually perturbed. No original base weights changed and there were **zero optimizer steps**.

Sprocket Chat HF source includes tokenizer metadata but a standard role-based chat template is not yet independently qualified; this prepass used ordinary/raw text tokenization. Real chat formatting, request isolation across multiple sessions, longer context/cache replay and cold checkpoint continuation must be separately proved.

Comparative training remains forbidden by experiments/ncr-001/protocol.json (train_authorized=false). There is no evidence of language quality improvement, routing speedup or a production-ready adapter. Accept/merge only in dependency order after protocol and shared-core reviews.
