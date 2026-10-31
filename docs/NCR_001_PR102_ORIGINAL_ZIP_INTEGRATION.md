# Source-matched PR #102 whole ZIP/TAR custody — main composition

31 October 2026. This is a **read-only historical custody integration**, not a new model experiment.

Parent [PR #197](https://github.com/techrote/neurounner/pull/197) originally
descended from historical [PR #102](https://github.com/techrote/neurounner/pull/102).
Its exact historical head `30995e9d7a1bd0ee1612e99ef414e6c12db4f246` is preserved in the second-parent Git ancestry
of the current-main composition, whose prior base was `e248d1231ee43035a0df8c73e11dd1b7c071de55`.

The original complete GitHub Actions #40573188117 artifact **ZIP** (79,486
bytes, SHA-256 `7e38a9c8438701c460597fc87016f3a26ece2d12b97482cfbcb44b0e6974a2ab`),
embedded 68,602-byte source snapshot TAR, and original decoded job log are
preserved as exact original binary/text Git blobs, rather than reconstructed
or just hash-referenced. The original manifest and fail-closed offline auditor
are kept byte-for-byte unchanged, including 7 source/negative tests.

The integration changes **only** the new read-only Actions trigger/branch
guard to match `main`, retains the original custody code, replaces obsolete
historical base comparison with the exact current-main SHA, and asserts
ancestry and unchanged RAG. It does not allow Actions write permissions, has
no push/commit stage, and does not execute a frozen model. `RAG.md` itself
is left unchanged because an unrelated PR RAG change would activate
root/Chat physical model workflows. Its historical two-line pointer remains
in immutable original ancestry; the current custody links are here.

Original source/custody failure [#40584453555](https://github.com/techrote/neurounner/actions/runs/40584453555)
and corrected [#40584618404](https://github.com/techrote/neurounner/actions/runs/40584618404)
remain historical. Original 21 generated negative observations (no valid
generated tool/memory actions, five of seven budget exhaustions), all
four model identities, and every original source/model byte are unchanged.

`train_authorized=false`, `full_72_conformance_pass=false`, and
`actual_compute_budget_qualified=false` remain binding. No heldouts,
comparative model training, optimizer, learned competence claim, or final
scientific admission.
