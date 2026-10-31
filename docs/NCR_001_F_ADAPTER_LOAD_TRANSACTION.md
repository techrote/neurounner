# NCR-001F — transactional matched-MLP adapter import

Research [#67](https://github.com/techrote/neurounner/issues/67); conformance [#72](https://github.com/techrote/neurounner/issues/72); matched controls [#73](https://github.com/techrote/neurounner/issues/73).

**TOY SOURCE-ONLY / NO ORIGINAL MODEL / NO TRAINING / NO ADMISSION.**

## Reproduced integrity defect

The existing FrozenDepthGatedMLP.load_adapter verified separate source metadata, then invoked PyTorch Module.load_state_dict(strict=True) directly on untrusted adapter tensor data. PyTorch's strict mode verifies key/shape consistency but does not make a load atomic: a valid earlier parameter can be copied before a later malformed shape raises. A standalone local CPU toy reproduced exactly that scenario: a first valid Linear weight changed despite a size-mismatched later Linear weight causing RuntimeError. This observation is **not** evidence that a physical NCR frozen model experienced corruption.

The existing adapter loader also bypassed the shared decoder-root lease. A second wrapper could therefore import adapter state while an active instrumented forward or disabled reference baseline was executing on the same frozen decoder. These defects affect source-integrity and reproducibility of matched-control comparisons, not the preregistered scientific protocol.

## Correction

The import now takes the existing per-wrapper lock **and** shared frozen-backbone lease across all metadata/file checks, validation, commit and rollback. It requires exact source/variant metadata, full expected adapter state keys, unchanged architecture extra state, dense tensors with the correct shape/dtype, and finite floating values **before changing any live parameter**. If a late load operation nevertheless raises, detached copies of every live parameter and buffer are restored before releasing the lease. The saved adapter file format and model identity contracts are unchanged.

## Acceptance matrix

- Valid adapter-only roundtrip at all four separate toy decoder geometries: SmolLM2 576, Sprocket Chat 1280, distinct root Instruct 1280, void.1 640.
- Refuse malformed late shapes even when an earlier incoming tensor is valid and altered; refuse missing/extra keys, wrong dtype, NaN/Inf, wrong extra state, malformed container and mismatched Chat/root metadata.
- Verify all rejected imports leave **every existing adapter parameter and reference forward output unchanged**, the original toy backbone unmodified and both locks released.
- Reject cross-wrapper import while another instance holds the same decoder-root lease; reject nested load under the same wrapper; afterward a valid import can succeed.
- Force a real mid-commit exception via test injection after modifying multiple parameters and prove complete rollback before release.
- Run tests under normal and optimized Python, plus all existing frozen-hook and shared-backbone ownership regressions.

The [read-only workflow](../.github/workflows/ncr-001f-adapter-load-transaction.yml) pins the original protocol Git blob 14ce5fbaa38f8806c8c8738f19c01b91385eebaf and source-pins Git blob 59fff1aed872b5563aa6898171983973e0cdb02c, exact checkout SHA, and CPU PyTorch 2.14.1. Its generated report is **source-only transient CI**, not an original physical-model experiment or permanent archival replacement.

## Scientific boundary and follow-up

No LM checkpoint loaded, model run, optimizer, held-out evaluation, new trained checkpoint, protocol/source-pins/fixture change, old raw ZIP/log rewrite, or main merge. The four frozen model lanes, taps, parameter matching and tokenizer distinctions are unchanged. Existing original physical receipts remain historical and must not be retrospectively relabelled as re-qualified on this new source.

The safety gate is insufficient by itself for actual frozen-model load qualification: an independently approved real-host checkpoint re-import and full #72/#73 review would remain future work. Values remain **train_authorized=false**, **full_72_conformance_pass=false**, and **actual_compute_budget_qualified=false**. Same-account COMMENTED review is not GitHub APPROVED or independent scientific acceptance.
