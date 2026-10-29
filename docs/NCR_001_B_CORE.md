# NCR-001B — shared depth-persistent cellular adapter core

Issue [#69](https://github.com/techrote/neurounner/issues/69), parent [#67](https://github.com/techrote/neurounner/issues/67). This is **synthetic-only implementation**, stacked on the still-draft [protocol PR #77](https://github.com/techrote/neurounner/pull/77). **No HF model has been loaded, no model forward/cache conformance proved, and no comparative training is authorized.**

## Module contract

Python source: [scripts/ncr_001_core.py](../scripts/ncr_001_core.py).

    core = PersistentDepthNCA(hidden_size=576)
    state = None
    for tap_index, h in enumerate(hidden_states_after_model_specific_taps):
        h_modulated, state = core.step(h, state, tap_index=tap_index)
    # discard state after THIRD tap; never carry between generated tokens

Here each hidden tensor has shape [B,T,H]. The explicit returned DepthFieldState stores [B,T,4,8,8] and next_tap; it is **not** attached to the NCA module or global memory. This prevents accidental memory persistence through normal code paths and enforces depth-tap ordering. Callers must start a new forward with state=None. **The class cannot by itself prove the caller won't inappropriately retain or reuse the returned state**; model wrappers in #70/#71/#78/#79, and black-box #72 tests, must enforce request/cache reset and parity.

- Initial tap seeds every token's 256-scalar grid using a hidden-to-field projection and tanh.
- Later taps apply bounded new observations to the **existing** grid rather than re-creating it.
- Every cellular update uses the *same* learned zero-padded 3×3 Conv2d perception and pointwise local rule. Three iterations per tap by default; same weights across all cells/iterations/depth taps.
- Bounded gate and tanh update keep cellular values in [-1,1]; does not claim convergence for arbitrary training rates.
- Zero-initialized field-to-hidden readout gives a **numerically exact no-op** for finite float32 hidden inputs at construction. Its gradient appears on the first optimizer step; gradients in earlier NCA layers appear after the readout becomes nonzero.
- reset_each_tap=True provides a control; iterations=1 tests update-count ablation. Neither changes weights, only field evolution.
- There is no token-axis convolution, no sampled stochastic update, no hidden cross-token state, no token-time recurrence, no expert routing or conditional skip.

Field transport is pure PyTorch with no imports of model backbones, Hugging Face, llama.cpp, Cybersand or CUDA. Different backbones receive **separate instances** of the same algorithm, each with independent weights and hidden-width projections.

## Parameter and field accounting

At default 8×8×4 geometry the trainable parameter count is exactly **514×H + 984**. The count includes affine layer norm, input projection, shared local perception/update kernels and zero-initialized output projection. It does **not** scale with the number of depth taps because the weights are shared.

| Frozen backbone lane | Hidden width H | NCA trainable parameters | FP32 field state / token |
|---|---:|---:|---:|
| SmolLM2-135M-Instruct | 576 | 297,048 | 1,024 bytes |
| Sprocket-500M Chat | 1,280 | 658,904 | 1,024 bytes |
| Sprocket-500M Instruct | 1,280 | 658,904 | 1,024 bytes |
| void.1 | 640 | 329,944 | 1,024 bytes |

At batch=1, T=64, state storage alone is **64 KiB**. PyTorch activations, temporary convolutions and autograd graphs cost more; this table is not an RSS forecast for training.

Rough multiply-accumulate count per token per tap is approximately **512×H + 3×45,056** at default three cellular updates (projection/readout and 3×3/pointwise convs). This excludes normalization, nonlinearity and memory traffic, and does not predict real decoder latency. The GitHub CPU probe reports wall seconds for just the synthetic adapter, **not** a claim of end-to-end SLM throughput or efficiency.

## Test and CI gates

Run locally on CPU:

    python3 -m unittest discover -s tests -p "test_ncr_001_core.py" -v
    python3 scripts/ncr_001_core_probe.py --sequence-length 64

The eight synthetic tests cover:
- finite bounded state and single-step radius-one locality, with no toroidal wraparound;
- exact initial no-op, state change between taps, reset-vs-persistent and one-vs-three-iteration ablations;
- batch/token isolation, token-permutation equivariance, fresh first-tap state and invalid order/shape refusal;
- two optimizer steps through a frozen toy backbone: readout learns first, then upstream cellular modules receive nonzero gradients; backbone remains unchanged;
- torch state_dict architecture metadata rejection, deterministic restore and every required model width.

The dedicated [NCR-001B CPU workflow](../.github/workflows/ncr-001b.yml) installs the same CPU torch pin as Neurounner's TRAIN-003/004 recorded environment, executes tests, and saves temporary synthetic-only JSON. Source pin checks, conformance and training budget gates **remain with #68 and #72**; all four model lanes must pass their own future adapter-hook and full-vs-KV-cache tests.

## Hosted CPU verification — 26 October 2026

The [NCR-001B Actions run 40408418607](https://github.com/techrote/neurounner/actions/runs/40408418607) is **PASS**: all eight synthetic unit tests and the four-width preflight completed on a four-thread AMD EPYC 9V45 runner with PyTorch 2.14.1+cpu. No HF model files were loaded and no LLM training occurred.

| Synthetic core-only lane | NCA trainable params | Three-tap × three-step wall time (median of three) |
|---|---:|---:|
| SmolLM2 width 576 | 297,048 | 2.62 ms |
| Sprocket Chat width 1280 | 658,904 | 3.29 ms |
| Sprocket Instruct width 1280 | 658,904 | 3.29 ms |
| void.1 width 640 | 329,944 | 2.66 ms |

Measurements used [64 independent token positions](../experiments/ncr-001/core-evidence/core-smoke.json), batch size 1 and **4 CPU threads**. Exact synthetic field storage was 65,536 bytes per width. Observed process peak RSS was 243,632 KiB, but is **not per-model** and includes runtime overhead. Reported synthetic timing is **not** a valid end-to-end inference overhead ratio; no claims about speedup or model quality follow.

Raw JSON probe measurements have been [promoted into Git](../experiments/ncr-001/core-evidence/core-smoke.json), with [source-run and original artifact hashes](../experiments/ncr-001/core-evidence/provenance.json) so the Actions ZIP can expire without losing numerical history. The provenance fixes source revision and labels source-only results distinctly from future #72 model conformance.

## Next

Review synthetic CI and independently re-run the eight tests. Only after accepting #68's protocol and #69's shared module should each separate model integration (#70 SmolLM2, #71 Sprocket Chat, #78 Sprocket Instruct, #79 void.1 Qwen3) begin. No model numerical claims can be drawn from this isolated test suite.
