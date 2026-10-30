# NCR-001F — fail-closed preregistration under optimized Python

Parent [NCR-001 #67](https://github.com/techrote/neurounner/issues/67), matched-control issue [#73](https://github.com/techrote/neurounner/issues/73).

## Defect and scope

At baseline Neurounner `main` `12f7bf0475aefb61227f676ea377134734fa5072`, `scripts/ncr_001f_plan.py::build_plan` used eight Python `assert` statements (covering nine locked invariants) to validate preregistered NCA geometry, tap/step counts, optimizer type and hyperparameters, and matched MLP bottleneck. Running `python3 -O` strips these statements. Thus an otherwise internally consistent but altered protocol could bypass these specific source-only gates, while the planner could still produce a `PASS_COMPARISON_PLAN_PRECHECK__BLOCKED_FOR_TRAINING` receipt. This does not demonstrate an actual training or physical-model bypass.

This change replaces exactly those nine assertions with **explicit fail-closed RuntimeError checks**, including strict Python type matching (so `True` cannot masquerade as integer 1). It does not loosen source/fixture hashes, change the immutable protocol, mutate any held-out content, load model weights, run a model or execute an optimizer.

## Executable negative sentinels

The `test_locked_pretraining_gates_fail_closed_under_optimization` test creates **ten temporary protocol mutations** and rejects each through the actual `build_plan` path. The same entire control suite runs in CI under ordinary Python and under **`python3 -O`**. This directly verifies the important invariant: **optimized execution must not erase preregistration guards**.

Mutations cover wrong field scalar count, tap count, cellular steps, optimizer kind, learning rate, betas, weight decay, gradient clipping (including a boolean type-confusion sentinel), and MLP bottleneck. Each mutation uses a temporary copy; frozen `protocol.json`, source-pins, fixtures and datasets remain untouched.

The preexisting source-only plan recreation/hash check remains in the same workflow. The accepted frozen protocol must still reproduce the original 7,204-byte report SHA-256 `f906206cfde550625512efe4c5af2f2add1a281a17c9cd3fff16991d95f17ed0`, because the output schema/content is unchanged. The new CI execution must verify this rather than assuming it.

## Scientific boundary

This is **source-level integrity repair** only. It does not establish actual frozen-host cache/causal conformance for any model, matched physical backward/optimizer cost, tokenizer-correct held-out evaluation, trained NCA/MLP quality or routing speedup. Four pinned lanes remain independent. Keep `train_authorized=false`, `full_72_conformance_pass=false`, `actual_compute_budget_qualified=false`; the PR is draft/unmerged, and a same-account COMMENTED review is never a distinct-person scientific approval. Preserve the original pre-fix source as the parent Git commit.
