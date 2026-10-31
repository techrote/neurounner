# NCR-001E void.1 post-lease NCA-only diagnostic learning gate

**Status: DRAFT — source and physical execution candidate.** This is a
narrow missing-check follow-up to the [formal post-lease adjudication #171](https://github.com/techrote/neurounner/pull/171),
not approval to train or merge. Parent [#67](https://github.com/techrote/neurounner/issues/67);
conformance [#72](https://github.com/techrote/neurounner/issues/72),
controls [#73](https://github.com/techrote/neurounner/issues/73),
void integration [#79](https://github.com/techrote/neurounner/issues/79),
BYTE-EOS [#92](https://github.com/techrote/neurounner/issues/92).

## Evidence behind the candidate

- Source/runtime NCA+MLP forward, synthetic corrected causal-EOS and separate
  Python replay: original [40747035876](https://github.com/techrote/neurounner/actions/runs/40747035876),
  archived losslessly under #168. No optimizer/backward.
- Source/runtime NCA+MLP 512/1024/2048 genuine incremental cache, strict
  suffix and real generation cache on/off: original
  [40755894558](https://github.com/techrote/neurounner/actions/runs/40755894558),
  archived losslessly under #170. No optimizer/backward; 12/12 four-step
  generations emitted IDs [101,110,32,109], **no true EOS 258**.
- The [#171 machine-readable decision](https://github.com/techrote/neurounner/blob/review/ncr001e-void1-postlease-adjudication-20261028/experiments/ncr-001/adjudication/void1-postlease-pretrain.json)
  specifically marks **three required #72 checks blocked**:
  frozen source unchanged *after a diagnostic step*, post-readout-warmup
  NCA gradient, finite supervised adapter loss *after warmup*.

## New, strictly bounded execution

One fixed **synthetic and public** ByteTokenizer completion
(`Q: π?\nA: ` → `🔧 café 日本語`) under the original correct
future-only EOS-ID 258 serializer at source commit
`9fbdee880697903aaf5b91cc29442c9e53479e50`.
The prompt is masked; the completion bytes plus the *integer EOS ID*
are supervised after Qwen3's native causal shift. No TRAIN-002b,
TRAIN-003 or NCA heldout/sample corpus is loaded.

The **same original frozen Qwen3 model** must match exact SHA-256
`ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`;
the actual executed ByteTokenizer Python source must hash to
`279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`.
Keep the four-model preregistered protocol and pins Git blobs exactly
`14ce5fbaa38f8806c8c8738f19c01b91385eebaf` and
`59fff1aed872b5563aa6898171983973e0cdb02c`.

The original post-lease **NCA alone** (329,944 trainable parameters)
receives precisely one in-memory SGD diagnostic update, learning rate
0.005, using one synthetic case. Initial zero-readout logits must equal
the original disabled frozen backbone exactly; the initial scalar
native causal-EOS loss and readout gradient must be finite/nonzero while
upstream gradients remain zero. Following the one readout update, a
second **backward-only** probe must yield finite/nonzero gradients for
readout, observe, perception, local rule and input normalization,
and three ordered, distinct, finite 8×8×4 cellular depth states.
The model's complete named frozen-tensor fingerprint and disabled
logits must remain bit-exact before and after, without backbone gradients
or leaked decoder hooks.

A previously constructed MLP wrapper may be present in the immutable
source loader, but **no MLP optimizer, gradient, loss or comparison run
is performed**. No saved trained adapter, ongoing optimizer schedule,
checkpoint, evaluation or probe of competence is created. Any negative
outcome, including failed source validation, NaN, missing gradient or
backbone contamination, is preserved as an original-host receipt,
raw log and Actions artifact. No numeric tolerance or dataset is changed.

Original-host workflow: `.github/workflows/ncr-001-void-nca-learning-gate.yml`.
Source-only negative tests: `tests/test_ncr_001_void_nca_learning_gate.py`.
Runtime receipt: `runtime/ncr001-void-nca-learning/receipt.json`.
The exact host/runner, SHA-256 source inventory, actual HF files,
EOS supervision, before/after gradients and frozen fingerprint are
retained. The workflow installs pinned `torch==2.14.1+cpu` and
`transformers==4.57.6`, uses an offline model invocation after
source-prefetch, and preserves failures. These artifacts **must be
independently read back and archived byte-for-byte into Git** before
any positive physical result may be used as qualified evidence.

## Scientific/admission boundaries

A one-step diagnostic is a **#72 conformance test** to establish
gradient routing and frozen-weight invariance, not NCR comparative
NCA-vs-MLP training. It is explicitly not a positive quality result
or permission to initiate the preregistered multiseed studies.
`train_authorized=false` must remain unchanged.
Even if physical CI passes, the output status remains
`PASS_VOID1_ONE_STEP_NCA_DIAGNOSTIC__72_SCIENTIFIC_ADMISSION_BLOCKED`:
final #72 admission, independent evidence/source review and all
dependency PR reconciliations remain separate, and no locked heldouts
are opened or modified. Other three model lanes are untouched.
