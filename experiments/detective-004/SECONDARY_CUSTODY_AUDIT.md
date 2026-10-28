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

## Source-native model-only prompt projection (answer contamination guard)

The Stage-A row intentionally retains the gold final assistant content, the full
`rendered` supervised training wire, and `target` labels. **None of those
artifacts is a valid inference prompt for a future held-out evaluation.**
Passing an entire final-task object or its `rendered` field directly to the
model would expose the correct answer before scoring.

The new read-only `model_input_prefix(row, b3)` derives the **model-only prompt**
from `turns[:-1]` using the exact pinned DETECTIVE-001B3
`native.render_text` implementation. It insists that the prompt ends with
the publisher's dangling `<|assistant|>` cue; equals the full audited wire
only up to the **last assistant character span start**; embeds the exact
game-visible user projection and pinned system manifest; and contains
**none of the target assistant completion**. Result-continuation tasks
deliberately include the earlier synthetic assistant tool call followed by
the host-owned tool result, but never the subsequent expected assistant reply.

Archive readback applies this check to all 36 immutable training rows. Additional
unit tests check all 36 training projections, 12 **unsealed synthetic test
objects** (not a real adapter-derived final), and adversarial earlier-user
gold-answer/view substitution. These checks cover character-level prompt
separation only; they do not establish the real frozen tokenizer's prompt IDs,
supervision labels, 512-token maximum or a valid Stage-B evaluation.

**Future Stage-B rule, if separately authorized:** the model runner must
consume only this projected prompt, record its exact raw UTF-8 bytes/hash
and tokenizer IDs, and must never consume the final record's `target`,
`rendered`, `loss_char_spans` or last assistant turn as input. An independent
custodian must validate the actual executed inputs and adapter provenance;
this author-side addition supplies no training or final-admission authority.

## Full frozen-recipe task equality (synthetic structural check)

The previous future-final layout guard established 12 unique IDs, expected
categories/families, exact user projection, source-native character spans and
certain host receipts. It did **not** establish that the supplied task records
were exactly those the frozen Stage-A recipe would generate. A substituted
visible asset and matching assistant/tool target, or a rewritten no-tool
reference answer, could pass those internal-consistency checks and change the
effective evaluation without changing the recorded final slots.

The author-side guard now reconstructs **only an expected in-memory mirror
of supplied records** using the pinned Stage-A \`family_data\`,
\`make_example\` and \`render_with_mask\` source. It derives the common
10-character hexadecimal case suffix already validated across the supplied
rows, then requires full object equality for each expected record, including
the exact user request, case evidence, credential, tool argument, host receipt,
reference completion, target labels, source wire and character spans. Two
unsealed synthetic adversarial tests show that internally consistent swapped
assets and substituted restraint labels now refuse. The test also retains its
complete 12-row synthetic positive control.

This closes a **structural recipe-substitution** gap only. A caller could
fabricate a coherent 12-row suite under another ten-character suffix; that
suffix is not proof of the actual adapter-derived posthash seed. **Stage B
must independently authenticate the full 64-hex posthash seed against the
frozen recipe, committed training manifest, actual signed-off adapter directory
and optimizer receipts, and must attest the executed raw model-only inputs.**
No actual posthash final is generated, opened, persisted or scored in Stage A.
The checks do not touch weights, model runtimes, tokenizer labels, hidden
relevance maps or sealed witness finals. They grant no Stage-B optimizer,
production or population authority.
