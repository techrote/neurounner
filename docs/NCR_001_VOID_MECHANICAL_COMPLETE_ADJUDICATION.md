# NCR-001E — void.1 locked-#72 mechanical coverage reconciliation

Source-only adjudication stacked on permanent masked-KV custody #221.

The exact evidence graph now contains bounded physical observations for all nine locked #72 check names **and** the expanded long-context/cache/batch/mask/generation mechanics:
- frozen/no-op identity;
- post-step frozen base, finite loss and post-warmup gradients;
- 2048 B=1 all-position incremental KV;
- strict 2048 future-suffix invariance;
- native 2044+4 GenerationMixin cache parity;
- B=2 left/right masked full-forward isolation;
- B=2 left/right masked **incremental KV** with 64 real cached continuation units per row, B=1 cached references and row-order replay;
- request reset and separate-interpreter cold reload.

New original masked-cache run 41136709967: left/right B=2 full-vs-cached max abs **1.6927719116210938e-05**; cached-vs-B1 cached <= **9.5367431640625e-06**; row-swap cached max abs 0.

Accordingly, this source review records **no currently known missing mechanical evidence category inside the locked #72/expanded cache-mask acceptance surface**.

That is not the same as granting `void1_conformance_pass=true`. Explicit final scientific per-model admission remains separate. Fresh-host replication, multi-sample learned effectiveness, matched NCA/MLP quality/compute and EOS stopping remain useful downstream scientific work, but are **not retroactive additions to the locked mechanical #72 checks**.

Historical #171 statuses remain archival facts and are not rewritten. No model/optimizer/heldout executes in this adjudication.

Locked: `train_authorized=false`; `void1_conformance_pass=false`; `full_72_conformance_pass=false`; `actual_compute_budget_qualified=false`.
