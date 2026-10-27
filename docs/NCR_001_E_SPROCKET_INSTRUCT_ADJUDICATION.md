# NCR-001E — Sprocket Instruct #72 lane adjudication

**Status:** `BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS`

This is a deterministic, source-only adjudication of the exact accumulated
Sprocket **root Instruct** conformance evidence. It does not execute a model,
does not inspect sealed held-outs, does not authorize training, and does not
merge any draft dependency.

Reviewed stacked heads:

- PR #102: `3592841802d85c442ca2521479b5d8795bee7233`
- PR #113: `7dec2f91b6542854762542807978e5bddead4f64`

The authoritative machine-readable report is
[`experiments/ncr-001/adjudication/sprocket-instruct-pretrain.json`](../experiments/ncr-001/adjudication/sprocket-instruct-pretrain.json).
It is regenerated from durable original evidence by
[`scripts/ncr_001e_sprocket_instruct_adjudication.py`](../scripts/ncr_001e_sprocket_instruct_adjudication.py).

## What is now adjudicated PASS at pre-training source/runtime level

| #72 required check | Adjudication | Core evidence |
|---|---|---|
| Same frozen base logits | **PASS** | exact frozen-vs-zero at multiple lengths, padding modes and generation paths |
| Causal suffix invariance | **PASS** | future-suffix drift and A→B→A request leakage exactly zero |
| Full/prefix/incremental cache | **PASS** | 48/256/2048-tail, all-position structural, padded-gap and real HF `generate()` paths below the fixed 5e-4 gate |
| Per-token isolation | **PASS_SOURCE_RUNTIME** | ordered 3-tap traces, all-position cached source test, masked-control sentinels |
| Per-request reset | **PASS** | independent requests, batches, fresh interpreters and generation calls reset exactly |
| Cold reload/state_dict | **PASS_PRETRAIN** | same-process fresh adapter construction plus separate-Python-process adapter/frozen-base replay |

The adjudicator binds all of these to the distinct root Instruct
`model.safetensors` SHA-256
`1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`,
revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`, original NCA hook
`b95c010f9b2ba3cbd9f81317981316bb9c8acd19`, taps `[7,16,25]`, and
the separate Chat checkpoint SHA
`551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`.

### Original actual-model evidence consumed by the adjudicator

Only original actual-model runs count as runtime evidence; archive-only CI is
not relabelled as another model invocation:

- `40469703065` — exact Chat/Instruct source roles, 16/64/128 no-op,
  left/right padding, 48-token cache, causal/request reset and same-process
  cold adapter reconstruction.
- `40515083104` — independent producer/consumer Python interpreters,
  240-prefill + 16 incremental units and exact adapter/reference replay.
- `40544384293` — distinct Chat/Instruct 512/1024/2048 source-context
  tests and control-ID transport.
- `40567832272` — source-authorized SFT host/tool/memory wire boundaries.
- `40573188117` — direct publisher renderer parity plus all-position source
  diagnostic and preserved negative generations.
- `40590487839` — 133/31-unit masked left/right batch isolation and true
  cached continuation.
- `40598034739` — right-padding KV gap and actual HF `GenerationMixin`
  cache-off/cache-on conformance.

The original receipts, runner identities, source hashes, numerical tolerances,
negative failures and archive limitations remain in the stacked
`experiments/ncr-001/evidence-review/` and
`datasets/ncr-001-sprocket-wire-diagnostic/` records.

## Why the lane is still BLOCKED

Issue #72 and `experiments/ncr-001/protocol.json` require **all** required
checks to pass. Three required checks cannot honestly be marked PASS while the
same protocol says `train_authorized=false`:

1. **`base_weight_unchanged_after_step` — BLOCKED_BY_TRAINING_POLICY**  
   No optimizer update is authorized, so there is no legitimate “after step”
   evidence.
2. **`nonzero_adapter_gradient_after_warmup` — BLOCKED_BY_TRAINING_POLICY**  
   The current qualification work deliberately performs inference-only
   manually activated readout tests, not a training warmup/backward pass.
3. **`finite_state_and_loss` — PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY**  
   Finite NCA state/logits are repeatedly demonstrated, but supervised
   training loss is not executed.

Because `any_failure_blocks_model_qualification=true`, the machine-readable
`per_model_gate` remains **false**. The adjudicator explicitly refuses to
set `sprocket_instruct_conformance_pass=true` or authorize training.

## Important scope limits

Structural source serialization is now well constrained, but the original
bounded generation diagnostic did **not** demonstrate reliable model-generated
tool calls or memory actions. Those negative observations are retained and are
not converted into a capability PASS.

The publisher's public SFT repository remains useful source authority for
role/mask/wire semantics but is **not** a cryptographic attestation that the
exact public Git commit built the released HF checkpoint.

No matched NCA-vs-MLP training, sealed held-out evaluation, optimizer update,
new trained checkpoint or merge is part of this adjudication.

## Decision

**Sprocket Instruct source/runtime pre-training conformance is independently
well supported, but final #72 per-model admission remains BLOCKED by the three
protocol-required learning/update/loss checks.**

Do not resolve the blocker by weakening the checklist or treating existing
green CI as an optimizer/gradient result. A future explicit owner decision to
authorize the bounded conformance learning step would be a separate change to
the frozen protocol and must preserve all current evidence and negative
history.
