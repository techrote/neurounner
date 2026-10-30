# NCR-001C — SmolLM2 frozen-depth NCA hook prepass

**Issue:** [#70](https://github.com/techrote/neurounner/issues/70).
**Dependencies:** [protocol PR #77](https://github.com/techrote/neurounner/pull/77) and [shared NCA PR #82](https://github.com/techrote/neurounner/pull/82).

## Exact design

The adapter wraps the pinned HuggingFaceTB/SmolLM2-135M-Instruct Llama causal LM at revision 12fd25f77366fa6b3b4b768ec3050bf629380bac.

SmolLM2 has 30 transformer blocks of width 576. The selected depth taps are the **outputs** of zero-based decoder blocks [9,19,29], i.e., after ten, twenty, and thirty blocks. The wrapper registers forward hooks on exactly these blocks and a begin/end pair of decoder-root hooks. The base model and tokenizer are unmodified.

Every decoder forward (prefill, a single cache decode step or an ordinary full-sequence training forward) resets a **local**, explicit field [batch, token, 4, 8, 8]. The same state evolves between the three taps, but is discarded at the end of the decoder forward. A lock prohibits simultaneous forward/generation using the same wrapper, and exception cleanup removes all hooks.

This is a **dense residual hidden-state modulation**, not a sparse skipping router. There is no cross-token cellular recurrence, hidden future information, or persistent generation-state sidecar.

All original SmolLM2 parameters, including embeddings and head, are frozen. The base is held in evaluation mode during adapter-only train/backward. The only trainable parameters are the **297,048** in the shared NCA module. The wrapper saves/loads only adapter weights with exact model ID/revision/variant/config metadata. It refuses other backbones and off-protocol taps rather than silently adapting them.

## Tests and qualification boundaries

Offline [test_ncr_001_smollm2.py](../tests/test_ncr_001_smollm2.py) uses a 30-layer toy fixture to verify:

- Zero-initialized NCA changes no logits or greedily decoded toy tokens.
- Three ordered layer taps are actually hit and cleaned up after each forward and generated token.
- No concurrent instrumentation, partial-depth execution, wrong model configuration or stale hooks can be silently accepted.
- Frozen toy base receives no gradients/optimizer updates; adapter readout trains and state_dict reload is exact.
- Base stays in eval mode while adapter is trainable.

The separate [SmolLM2 real-checkpoint preflight](../scripts/ncr_001c_smollm2_preflight.py) **downloads only this pinned 135M checkpoint**, verifies the actual safetensors bytes plus config.json, tokenizer.json and tokenizer_config.json against the durable SHA-256 inventory in [source-pins.json](../experiments/ncr-001/source-pins.json). It runs with no optimizer steps on a four-thread CPU runner. It checks:

- Frozen original logits vs zero-init NCA logits on three single requests and a padded two-request batch, at 1e-4 absolute/relative tolerance.
- Greedy generation output and per-decoder state reset under zero-init.
- Initial readout gradient under teacher-forced loss, with frozen base gradients absent, **but no optimizer update**.
- Artificially **perturbed, not trained** NCA readout: full-sequence vs incrementally cached logits (5e-4), and future-suffix invariance.

A successful prepass proves compatibility of the injection and a limited causal/cache path. It **does not** qualify full #72, which requires longer-sequence, independent cache/readback and source-bound reviewer qualification, and does **not** permit comparative training. The workflow is [ncr-001c.yml](../.github/workflows/ncr-001c.yml).

Run in the pinned CPU environment:

    python3 -m unittest discover -s tests -p "test_ncr_001_smollm2.py" -v
    python3 scripts/ncr_001c_smollm2_preflight.py

Evidence is staged at runtime/ncr-001/smollm2-preflight.json, including failed partial runs; source-bound results must be read back and promoted in Git before the Actions artifact expires.

## Hosted source-bound result — 26 October 2026

[GitHub Actions 40422167066](https://github.com/techrote/neurounner/actions/runs/40422167066) passed **all six offline toy-hook tests and the genuine frozen HF SmolLM2 integration prepass**. The actual 269,060,552-byte safetensors checkpoint SHA-256 was verified by reading downloaded bytes, not simply trusting published metadata: **5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c**. All pinned config and tokenizer hashes matched.

| Test | Measured result |
|---|---:|
| Zero-init original-vs-adapter logits, three lengths | 0 maximum absolute difference |
| Zero-init padded two-request logits | 0 |
| Greedy three-token generation | Identical outputs, three independent decoder forwards |
| Real frozen-base readout gradient L1 | 476.267 |
| Base parameters with gradients | 0 |
| Manual nontrained readout perturbation, max logit effect | 0.002409 |
| Nonzero NCA full-vs-incremental KV-cache maximum error | **5.2452 × 10⁻⁵**, under fixed 5 × 10⁻⁴ tolerance |
| Nonzero NCA future-suffix invariance | 0 maximum absolute difference |
| Real backbone optimizer steps | 0 |
| Peak process RSS | 1,166,016 KiB |

The run used four threads on Intel Xeon Platinum 8370C with torch 2.14.1+cpu and Transformers 4.57.6. **These are short-sequence source-bound conformance measurements, not throughput or model-quality results.**

The original [complete 3,891-byte JSON receipt](../experiments/ncr-001/smollm2-preflight/receipt.json) and [source-bound provenance](../experiments/ncr-001/smollm2-preflight/provenance.json) are preserved in Git. The report SHA-256, **b607dabd1a2da866e37f5c96ec5cc2db49bbd4482caa1aad405a9ee61bc14e23**, is checked in CI against the promoted bytes; the original Actions ZIP artifact #12698945189 was recorded separately.

**Verdict:** preliminary SmolLM2 hook compatibility **PASS**. Full independent #72, including longer sessions, fresh checkpoint reload and a separate reviewer, remains outstanding. No NCA training or quantitative performance comparison is authorised.

## Execution and stop conditions

This PR intentionally stacks onto shared-core PR #82 while protocol PR #77 is still draft. Do **not** merge model-specific hooks to main before the protocol and core have been independently reviewed and accepted.

Do not alter TRAIN-003/Persona/Village held-out datasets or train their existing LoRAs. No model-evaluation prompts are selected from fixed held-out outputs. The protocol training-admission flag remains false, so comparative NCA training is prohibited. If checkpoint checksum, decoder shape, causal mask or cache agreement fails, preserve evidence and block SmolLM2 without weakening tolerances to force a pass.
