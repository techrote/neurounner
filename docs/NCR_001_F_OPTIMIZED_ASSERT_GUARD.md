# NCR-001F — fail-closed preregistration under optimized Python

Parent [NCR-001 #67](https://github.com/techrote/neurounner/issues/67), matched-control issue [#73](https://github.com/techrote/neurounner/issues/73).

## Defect and scope

At baseline Neurounner `main` `12f7bf0475aefb61227f676ea377134734fa5072`, `scripts/ncr_001f_plan.py::build_plan` used eight Python `assert` statements (covering nine locked invariants) to validate preregistered NCA geometry, tap/step counts, optimizer type and hyperparameters, and matched MLP bottleneck. Running `python3 -O` strips these statements. Thus an otherwise internally consistent but altered protocol could bypass these specific source-only gates, while the planner could still produce a `PASS_COMPARISON_PLAN_PRECHECK__BLOCKED_FOR_TRAINING` receipt. This does not demonstrate an actual training or physical-model bypass.

This change replaces those eight assertions (covering nine predicates) with **explicit fail-closed RuntimeError checks**, including strict Python type matching (so `True` cannot masquerade as integer 1). It does not loosen source/fixture hashes, change the immutable protocol, mutate any held-out content, load model weights, run a model or execute an optimizer.

## Executable negative sentinels

The `test_locked_pretraining_gates_fail_closed_under_optimization` test creates **ten temporary protocol mutations** and rejects each through the actual `build_plan` path. The same entire control suite runs in CI under ordinary Python and under **`python3 -O`**. This directly verifies the important invariant: **optimized execution must not erase preregistration guards**.

Mutations cover wrong field scalar count, tap count, cellular steps, optimizer kind, learning rate, betas, weight decay, gradient clipping (including a boolean type-confusion sentinel), and MLP bottleneck. Each mutation uses a temporary copy; frozen `protocol.json`, source-pins, fixtures and datasets remain untouched.

The preexisting source-only plan recreation/hash check remains in the same workflow. The accepted frozen protocol must still reproduce the original 7,204-byte report SHA-256 `f906206cfde550625512efe4c5af2f2add1a281a17c9cd3fff16991d95f17ed0`, because the output schema/content is unchanged. The new CI execution must verify this rather than assuming it.

## Scientific boundary

This is **source-level integrity repair** only. It does not establish actual frozen-host cache/causal conformance for any model, matched physical backward/optimizer cost, tokenizer-correct held-out evaluation, trained NCA/MLP quality or routing speedup. Four pinned lanes remain independent. Keep `train_authorized=false`, `full_72_conformance_pass=false`, `actual_compute_budget_qualified=false`; the PR is draft/unmerged, and a same-account COMMENTED review is never a distinct-person scientific approval. Preserve the original pre-fix source as the parent Git commit.

## Production follow-up — exact seed identity and primary-step type

The prior guard protected NCA geometry and optimizer hyperparameters but left a
distinct fail-open case: `len(seed_list) == 3` accepted substitutions,
duplicates and reordering of the three preregistered random seeds. The planner
would still report `PASS_COMPARISON_PLAN_PRECHECK__BLOCKED_FOR_TRAINING` while
writing different source-row schedule identities. Numeric equality also accepted
`primary_steps_per_seed=192.0` despite the preregistered integer count.

The read-only source planner now requires the original ordered integer seed list
`[20261026,20261027,20261028]` and exactly 192 **integer** primary steps,
using unconditional, type-strict `RuntimeError` checks. Ten additional
mutated temporary-protocol sentinels reject changed/duplicate/reordered/float/
string/bool seeds and wrong/float/string/bool step counts under normal and
`python3 -O` execution.

This is a separate, narrowly scoped source-integrity correction to the same
draft PR #223. The unchanged original protocol and model/source/fixture pins
must still produce the original 7,204-byte plan SHA-256
`f906206cfde550625512efe4c5af2f2add1a281a17c9cd3fff16991d95f17ed0`
in exact-head CI. No model or optimizer is run and no training or admission gate
is elevated. Original G0 SHA/CI/review remain historical evidence.

## Production follow-up — frozen model source identity and sequence-unit gates

A four-key model-dictionary presence check was not enough: `build_plan`
previously copied each model's mutable `repo_id`, revision, Chat/root variant,
weight path, depth taps, width and per-model token-unit budget into a superficially
successful **source-only** plan without comparing them to an independent frozen
identity contract. A mutated yet self-consistent temporary protocol could
therefore change its planned model population while still keeping training blocked.
This is an **integrity/preflight bypass**, not evidence of actual model execution.

The planner now requires each of the four separately pinned lanes to match its
original NCR-001A source: exact model repo/revision, distinct Sprocket
Chat `chat/model.safetensors` versus root Instruct `model.safetensors`,
void.1 Qwen3/ByteTokenizer variant, layer count/hidden width, post-block depth
taps and preregistered within-model sequence budgets (256 subword units, or
1024 byte-tokenizer units for void.1). All checks use unconditional,
type-strict `RuntimeError` refusal, including float/bool fields equal under
Python's ordinary numeric equality. The independent constants are based on
the existing checked-in `protocol.json` and `source-pins.json` but are **not**
read from the mutable protocol argument under test.

One new source-only test supplies **40 adversarial temporary-protocol
mutations** spanning all four models, nine fields per model and four equality/
type-confusion sentinels. Both normal and `python3 -O` CI must reject all
cases while reproducing the exact original 7,204-byte source-plan SHA-256
`f906206cfde550625512efe4c5af2f2add1a281a17c9cd3fff16991d95f17ed0`.
No frozen source/weight/fixture/protocol bytes, original ZIP/log custody or
scientific admission gates are modified; this remains draft-only source work.
