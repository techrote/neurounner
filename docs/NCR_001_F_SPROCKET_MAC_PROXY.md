# NCR-001F — pinned frozen Sprocket adapter MAC proxy

**Scope:** issue #73, existing draft PR #98. This is a source-only arithmetic-accounting extension. The original model source/runner, archived physical receipts, preregistered evaluation and review requirements remain unchanged. No model or tokenizer execution, optimizer, training or heldout access.

## Two distinct Sprocket checkpoints, same geometric accounting

The four-model preregistration includes three NCA post-block taps, an 8×8×4 state per token and three local update iterations at each tap. The matched conventional gated MLP uses H→256 and 256→H linear maps at each of the three taps. The NCA also uses H→256 observation and 256→H readout maps, plus a 4→16 3×3 convolution (36,864 MAC per 8×8 grid update) and 16→8 1×1 convolution (8,192 MAC per grid update). Source Git blobs for both implementations, the protocol/pins and frozen hooks are required to match immutable originals.

| Source-only proxy per token | Sprocket Chat subfolder | Sprocket root Instruct |
|---|---:|---:|
| Frozen weight file | chat/model.safetensors | model.safetensors |
| Hidden width | 1280 | 1280 |
| Gated MLP dense MAC | 1,966,080 | 1,966,080 |
| NCA dense + convolution MAC | 2,371,584 | 2,371,584 |
| Extra NCA convolution MAC | 405,504 | 405,504 |
| Ratio NCA/MLP | **1.20625** | **1.20625** |
| NCA / MLP adapter parameters | 658,904 / 658,432 | 658,904 / 658,432 |

The 472-parameter difference is 0.0716% of NCA count, within the existing 5% **parameter-only** rule. Despite that, NCA has **20.625% more counted dense/conv MACs** at H=1280 with all nine local updates. The one-iteration ablation instead has only 135,168 convolution MACs across three taps; it is not the preregistered three-step primary. Identical geometry does **not** conflate the distinct frozen Chat/Instruct weights, tokenizer behavior, causal semantics or loss.

## Not a measured compute-budget PASS

MAC proxy is **not actual FLOPs**, time, memory traffic, energy, numerical accuracy or end-to-end training cost. It excludes nonlinearities, normalization, autograd, optimizer, common frozen backbone, backend kernels and hardware efficiency. One 8×8×4 float32 field is 1,024 bytes per token before transient activations; this is not peak RSS. Actual same-host measured training/inference budgets and model-specific #72 scientific gates remain prerequisites.

The new independent standard-library auditor, tests and read-only Actions workflow refuse altered source Git blobs, changed three-step geometry, variant checkpoint substitutions and false training admission. Its only permitted classification is SOURCE_MAC_PROXY_QUANTIFIED__ACTUAL_COMPUTE_MATCH_UNMEASURED__TRAINING_BLOCKED. It does not revise the preregistered comparison suite, touch original #98 real-host artifacts or authorize comparative training.
