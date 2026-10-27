# NCR-001E — Sprocket Chat #72 lane adjudication

**Status:** `BLOCKED_REQUIRED_LEARNING_CHECKS__SOURCE_RUNTIME_PRETRAIN_PASS`

This is a deterministic, source-only adjudication of the accumulated
**Sprocket Chat** conformance evidence. It does not execute a model, access
sealed held-outs, run gradients/optimizers, authorize training, or merge any
draft parent.

Reviewed exact heads:

- source/runtime reviewer PR #113:
  `7dec2f91b6542854762542807978e5bddead4f64`
- Chat generation reviewer PR #123:
  `57bf036f1a8728b35545473b475e3306eee440d7`

The authoritative machine-readable result is
[`experiments/ncr-001/adjudication/sprocket-chat-pretrain.json`](../experiments/ncr-001/adjudication/sprocket-chat-pretrain.json),
reproducibly derived by
[`scripts/ncr_001e_sprocket_chat_adjudication.py`](../scripts/ncr_001e_sprocket_chat_adjudication.py).

## Exact source identity

The Chat lane is explicitly distinct from root Instruct:

- repository/revision:
  `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`
- Chat weight:
  `chat/model.safetensors`, 1,002,207,984 bytes,
  SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`
- root Instruct sibling:
  `model.safetensors`,
  SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`
- original Chat NCA hook:
  `83a56835a7476f17a17980d28dd1a7c14bb6f2f6`
- depth taps: `[7,16,25]`
- ordinary source formatter Git blob:
  `39cf015735e74a1d1e08682b623404bc1e600b9b`

The Chat lane uses the publisher-documented **ordinary user/assistant framing**.
It explicitly does **not** inherit Instruct tool-call/result or memory semantics.
The publisher's training-source descriptions indicate those tool/memory
examples were excluded from Chat fine-tuning. Dedicated control-token
recognition therefore is not evidence of Chat tool/memory competence.

## Required #72 source/runtime checks now independently supported

| Required check | Adjudication | Core source evidence |
|---|---|---|
| `same_base_logits` | **PASS** | frozen-vs-zero exact under roles/padding and actual HF generation |
| `causal_suffix_invariance` | **PASS** | strict future suffix influence zero |
| `full_prefix_vs_incremental_cache` | **PASS** | 48, 256, 2048-tail, right-gap, actual HF generate cache on/off |
| `per_token_isolation` | **PASS_SOURCE_RUNTIME** | true one-token cached calls, ordered 3-tap traces, masked-padding sentinels |
| `per_request_reset` | **PASS** | A→B→A, independent process replay, batch and generation reset exact |
| `cold_reload_state_dict` | **PASS_PRETRAIN** | new adapter instance plus separate-Python-process replay exact |

### Original actual-model evidence counted

Only original physical model executions are listed as runtime evidence:

1. **40469703065** — pinned source ordinary roles, distinct full Chat/Instruct
   checkpoints, 16/64/128 frozen-vs-zero exactness, left/right padded batches,
   48-token incremental cache, suffix/request reset, same-process cold adapter.
2. **40515083104** — separate Python producer/consumer interpreters for the
   exact Chat checkpoint; 240-prefill + 16 true incremental units;
   max cache error **3.814697265625e-05**, request/suffix/cold replay drift zero.
3. **40544384293** — 512/1024/2048 exact pinned Chat context tests;
   2048-tail cache max **2.193450927734375e-05**; raw dedicated control-ID
   transport max **2.5033950805664062e-05**. This remains transport only,
   not tool semantics.
4. **40603226200** — new ordinary-role right-padded KV-gap and real
   `GenerationMixin.generate()` cache-on/cache-off review:
   right-gap max **2.8133392333984375e-05**,
   generation max **2.002716064453125e-05**,
   masked control-ID padding exact, deliberately invalid unmasked-padding
   sentinel detected, frozen tensors unchanged.

Archive-only workflows are never counted as additional model executions.

## Why final Sprocket Chat #72 is still BLOCKED

The frozen protocol requires every item in
`conformance.required_checks` and specifies
`any_failure_blocks_model_qualification=true`.

Exactly three required items cannot be honestly marked PASS while
`train_authorized=false`:

1. **`base_weight_unchanged_after_step` —
   `BLOCKED_BY_TRAINING_POLICY`**  
   There is no authorized optimizer step, so “after step” evidence does not
   exist.
2. **`nonzero_adapter_gradient_after_warmup` —
   `BLOCKED_BY_TRAINING_POLICY`**  
   Existing nonzero effects are deliberately manual, untrained readout
   interventions; no backward/training warmup is authorized.
3. **`finite_state_and_loss` —
   `PARTIAL_BLOCKED_LOSS_BY_TRAINING_POLICY`**  
   Finite state/logit behavior is repeatedly checked, including generation.
   A supervised training loss has intentionally not been executed.

The machine-readable `per_model_gate` therefore remains **false**.

## Scope that must not be inflated

This adjudication does **not** establish:

- Chat tool/memory competence or Instruct tool/memory grammar;
- any optimizer, gradient warmup or trained adapter;
- sealed held-out behavior;
- NCA-versus-matched-MLP comparative performance;
- publisher Git source as cryptographic training-build attestation;
- four-model final #72 admission;
- permission to merge any draft dependency.

The original failed Chat reviewer source-only run **40603167288** is preserved:
it failed before any model execution because a test miscounted its own
synthetic vector lengths. It was corrected without changing runtime behavior
or numerical gates.

## Decision

**Sprocket Chat source/runtime pre-training conformance is independently
well-supported. Final per-model #72 admission remains BLOCKED by exactly the
three protocol-required learning/update/loss checks above.**

Do not resolve this by weakening #72 or interpreting existing green inference
CI as gradient/optimizer evidence. Any future bounded learning qualification
requires an explicit, separate owner decision that changes the admission state
while preserving the current immutable evidence and negative history.
