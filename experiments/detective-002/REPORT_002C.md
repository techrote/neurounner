# DETECTIVE-002C — LFM2.5-350M Transformers-v5 compatibility result

Authoritative run: [Actions 40609927745](https://github.com/techrote/neurounner/actions/runs/40609927745), exact pre-evidence source `053ddd90dcae646a647806fc65ca851ae8d85232`.

**Outcome: `tokenizer_incompatible`.**  
**Source integrity: PASS. Wheel integrity: PASS. Independent readback: PASS.**  
**Model weight downloaded: NO. Model loaded: NO. Dialogue generated: NO.**

This is a focused compatibility result, not a model-quality result.

## Historical control

DETECTIVE-002B used the same exact frozen LFM2.5-350M source under Transformers 4.57.6.

- exact Stage-A model source/weight identity verified;
- tokenizer construction failed:
  `ValueError: Tokenizer class TokenizersBackend does not exist or is not currently imported.`
- no dialogue was generated.

That result remains unchanged.

## Frozen v5 requalification

DETECTIVE-002C changed only the preregistered tokenizer/runtime stack to **Transformers 5.20.1**.

Before installation, CI downloaded the 5.20.1 binary wheel and verified SHA-256:

`5f51ce444193e5d0a751e3fa489ddbc2224ff12f6675a5cfefd4044b09757c19`

Resolved runtime packages recorded by the experiment:

| Package | Version |
|---|---|
| Python | 3.12.15 |
| Torch | 2.14.1+cpu |
| Transformers | **5.20.1** |
| huggingface_hub | 2.3.0 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |

## Tokenizer-only result

All frozen small-source identities passed before tokenizer construction:

- config SHA-256 `720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8`;
- tokenizer config SHA-256 `3701f370c70034d06e28947ff51c1063983393319f429e3ce58f06a45fae67cc`;
- chat template SHA-256 `ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6`.

Unlike Transformers 4.57.6, **Transformers 5.20.1 successfully instantiated the frozen tokenizer as `TokenizersBackend`**, with the frozen declaration `backend: tokenizers`.

The next preregistered step—rendering the exact merged DETECTIVE-002B B01–B04 messages—failed with:

- exception type: `AttributeError`;
- exception message: empty;
- rendered prompt count: 0.

Therefore the tokenizer stage is classified `tokenizer_incompatible`.

The experiment deliberately does **not** infer a more specific cause from an empty exception message. In particular, it does not claim that the checkpoint, the standalone Jinja template, the v5 tokenizer backend, or a specific template variable is individually defective without a separately instrumented study.

## Tokenizer-before-weight gate worked

The most important custody result is negative but clean:

- `weight_downloaded: false`;
- no `weight_stage`;
- no model stage;
- no model load;
- no dialogue output.

The exact 708,984,464-byte model weight was never downloaded. This proves the requalification stopped at the boundary that was preregistered before observing the v5 result.

## Comparison with DETECTIVE-002B

| Runtime | Frozen tokenizer class | Result |
|---|---|---|
| Transformers 4.57.6 | `TokenizersBackend` | class unavailable during tokenizer construction |
| Transformers 5.20.1 | `TokenizersBackend` | class constructs; matched chat-message rendering raises `AttributeError` |

So v5 resolves the **specific class-availability failure**, but the exact checkpoint/template still does **not** qualify for the common DETECTIVE chat runtime.

This is progress in diagnosis, not a runtime PASS.

## Decision

**DETECTIVE-002C: tokenizer_incompatible.**

- LFM2.5-350M remains excluded from the common Transformers detective-body runtime.
- The DETECTIVE-002B `runtime_fail` remains historical evidence.
- The LFM `lfm1.0` production/game-distribution gate remains unchanged.
- No body is selected for training.
- No hidden DETECTIVE-001R1 mapping was accessed.
- No population expansion is admitted.

## Next diagnostic, if pursued

Do not change 002C post hoc.

A separate compatibility issue may instrument **chat-template rendering only**, using the same frozen small files and no model weights, to capture a full traceback and isolate whether the empty `AttributeError` comes from:

1. a Jinja/template assumption;
2. the `TokenizersBackend` implementation;
3. a missing template context field;
4. an incompatibility between this frozen LFM2.5 tokenizer/template and Transformers 5.20.1.

Only after that source-level diagnosis should another model-load attempt be considered.

Durable evidence: [datasets/detective-002c](../../datasets/detective-002c/).
