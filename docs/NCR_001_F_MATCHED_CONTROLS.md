# NCR-001F — matched conventional gated adapter and read-only evaluation plan

Issue [#73](https://github.com/techrote/neurounner/issues/73), parent [#67](https://github.com/techrote/neurounner/issues/67).

This branch is **not a model-training implementation**. It provides the ordinary noncellular capacity control, a deterministic four-model experiment plan and source/held-out integrity tests. It is stacked on [#82 NCA core](https://github.com/techrote/neurounner/pull/82); that branch predates the locked last revision of [#77 protocol](https://github.com/techrote/neurounner/pull/77), so this branch snapshots the current exact protocol plus the previously locked fixtures. Reconcile those snapshots in dependency order rather than treating them as independent design changes.

## Controlled comparison

[Control module](../scripts/ncr_001f_controls.py) implements a standard **LayerNorm → H×256 up projection → SiLU → learnable featurewise sigmoid gate → zero-initialized 256×H residual readout**. It is purely feed-forward and has no cellular neighbourhood, no recurrent state, no token-axis coupling and no depth-persistent field. Weights are reused at three transformer post-block taps.

Parameter arithmetic: LN scale/bias 2H, up projection 256H+256, learnable 256-feature gate, bias-free down projection 256H. The control total is **514H+512**. The NCA totals **514H+984**, exactly 472 more parameters, within the preregistered 5% margin.

| Frozen base lane | Width | NCA params | Gated control params | Relative count gap |
|---|---:|---:|---:|---:|
| SmolLM2-135M Instruct | 576 | 297,048 | 296,576 | 0.159% |
| Sprocket Chat | 1,280 | 658,904 | 658,432 | 0.072% |
| Sprocket Instruct | 1,280 | 658,904 | 658,432 | 0.072% |
| void.1 Qwen3 | 640 | 329,944 | 329,472 | 0.143% |

**Parameter matching is not compute matching.** The NCA repeats three cellular iterations across three taps and performs additional convolutions. Quality differences must be interpreted alongside measured RSS, FLOPs and prompt/decode throughput. Do not claim free speedups.

The zero-initialized output projection makes both adapters initially exact no-ops. Readout gradients occur first; upstream gate/core gradients begin after readout weights move. Synthetic toy tests demonstrate this without training any actual language model.

## Data leakage and schedule guarantees

[Read-only experiment planner](../scripts/ncr_001f_plan.py) verifies the exact 768 TRAIN-003 train rows and 128 held-out rows, actual disjoint row IDs and semantic families, fixed manifest SHA-256 and all five sealed evaluation fixture hashes. Held-out records are never sampled for model updates.

For each of three preregistered seeds, it constructs a 192-example source-row schedule. Each control lane within one model must use identical example order, optimizer updates, original tokenizer and source checkpoint. The plan records model revision/variant, post-block taps, token budget, trainable parameter count and seed-specific schedule SHA. void.1 counts tokenizer bytes/units distinctly from BPE.

The plan is deliberately **BLOCKED_FOR_TRAINING** and includes explicit false flags for training authorization and independent four-model conformance. It performs no model downloads, optimizer steps, weight modifications, held-out model generation or learned checkpoint production.

## Executable source gates

The [bounded four-thread CPU workflow](../.github/workflows/ncr-001f.yml) runs synthetic parameter, no-op, frozen toy-gradient, locality-independent-control and negative source/leakage tests.

    python3 -m unittest discover -s tests -p "test_ncr_001f_controls.py" -v
    python3 scripts/ncr_001f_plan.py --out runtime/ncr-001/matched-control-plan.json

The tests reject a tampered fixture, unauthorized training flag, overlapping held-out ID/family, wrong hidden shape, wrong architecture metadata and unregistered parameter-count mismatch.

**Still pending:** connecting the control to real HF transformer hooks with identical tap semantics; source/role tokenizer qualification for Sprocket Chat/Instruct and void.1; independent #72 longer/cold-reload/concurrent-request tests; equal compute budgets and real language-quality assessment. No NCA vs MLP learning result can be claimed yet.

## Next

Reconcile protocol #77 and core #82, independently qualify each model’s #72 source/cache/formatter behaviour, and add model-specific MLP hook integration. Only thereafter may a separately reviewed finite optimizer run in #74/#75/#80/#81 be admitted. Negative or inconclusive outcomes must be preserved.
