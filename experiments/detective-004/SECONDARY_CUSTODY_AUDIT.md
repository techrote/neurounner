# DETECTIVE-004A — secondary Stage-A custody check (author-side)

This check supplements [Stage A](STAGE_A.md) and [draft PR #149](https://github.com/techrote/neurounner/pull/149). **It is not independent review and does not grant Stage B, learner-final materialization, production admission or cast expansion.**

## Specific gap addressed

The original producer validator validates the separate `row["view"]` structure and the rendered turn sequence, but does not itself prove that the view used for permission/argument checks exactly matches the JSON embedded in the **actual model-visible user message**. A modified row can therefore contain a broader audit-side view than the model was shown. The committed generator still deterministically produces matched rows; this is an adversarial verifier gap, not evidence that the original 36-row archive was corrupted.

The additional `scripts/detective_004a_custody_audit.py` mechanically cross-checks each of the 36 committed source-bound training rows against the exact user-message projection, the pinned B3 tool manifest, family-bound case IDs, assistant content + closing end-token **character** supervision spans, visible assets/credentials, and host-only receipt separation. It rejects out-of-band view changes, case-family aliasing and forged results.

Run with:

```sh
python3 scripts/detective_004a_custody_audit.py --archive datasets/detective-004a
python3 -m unittest tests.test_detective_004a_custody -v
```

The additional verifier only consumes frozen Stage-A data. It does **not** change the corpus, `training.jsonl` SHA-256 `a8ea42c3cd2344fcc285e35c98dfcf2745282fa5b7738e02722c820c0983e513`, the manifest, generator/config Git blobs, or the sealed 12-task posthash final recipe. No final is generated, and no tokenizer/weights are loaded.

## Unresolved admission gates

A distinct independent reviewer must still qualify the source-native role/assistant loss design, case-family separations, source identities, final custody and the full protocol before Stage A merges. The frozen tokenizer's real token-label arrays and 512-token limit remain **unverified** in Stage A and are mandatory prior to any Stage B optimizer step. Archive-side character spans are not token-level labels.

The current `adapter_dir_sha256` helper binds file-path/size/hash identity; merely naming two files does not prove a genuine trained adapter or correct base ancestry. Any future Stage-B final custodian must independently authenticate the actual adapter, optimizer receipt, root-weight identity and the posthash final before allowing the evaluator to be interpreted as evidence. A naked `admission_gate` result on arbitrary task objects is not qualification.

DETECTIVE-001R1 #121 remains separately reviewer-blocked: no opaque relevance ratings, no population expansion.


## Future Stage-B scoring-layout guard (no final opened)

A distinct pre-admission failure mode was identified in the **frozen** Stage-A
`protocol.admission_gate` helper: `len(final_tasks) == 12` together with
matching output-key *sets* does **not** imply 12 unique task IDs. A repeated
successful tool task can be counted twice while its omitted counterpart is not
detected. The existing helper is retained unchanged because its Git blob is
recorded in the durable training manifest; modifying it would invalidate the
source-bound 36-row archive.

The secondary `scripts/detective_004a_custody_audit.py` now exposes
`audit_future_gate_layout(tasks, b3)` and
`guarded_admission_gate(tasks, adapted_outputs, base_outputs, b3)`.
The guard checks exactly 12 distinct IDs, the 9/1/1/1 task-kind counts, the
three frozen final families, the per-family operation/extra-task slots, the
same 10-hex-character posthash case prefix across families, the model-visible
user projection, pinned system manifest, source-native character spans and
the host-owned success/denial receipts. Only then does it invoke the original
strict parser and score gate.

Four regression tests exercise the guard on **synthetic unsealed unit-test
objects only**: complete layout, duplicated passing task IDs, category/user
projection substitution, forged receipt and mixed posthash seed prefixes.
These tests do not use a trained adapter, materialize the actual posthash
12-task suite, read C1/C2/C3 finals or touch opaque relevance mappings.

**Important:** Structural validation is **not** an adapter/source attestation,
nor cryptographic proof that the supplied task list was independently
materialized from the preregistered recipe. Future Stage B must independently
reconcile the source/weight/tokenizer pins, optimizer and adapter ancestry
receipts, exact final recipe/training manifest/adapter digests and raw outputs,
and must use the guarded entry point before considering any evaluator result.
A standalone score from the unguarded helper is insufficient. Independent
Stage-A approval remains missing; no training or population expansion is
authorized by this addition.
