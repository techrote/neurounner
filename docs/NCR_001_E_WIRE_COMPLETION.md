# PR #102: bounded source-wire completion and diagnostic contract

This completes the **implementation scope** of the existing PR, not NCR-001's
four-model experiment. The draft remains unmerged, with `train_authorized=false`.
The previous exact original JSONs and source snapshots are historical evidence
and must not be regenerated, relabelled or edited.

## Review findings and repairs

The original serializer at `3583905ce2c91a813643e4a4e836eaa659cbde34`
(Git blob `0abf73a8877cad61392517baa1a8487276ba5507`) omitted dedicated
end-of-text ID 0 and padding ID 1 from its injection checks. It also accepted
JSON `1e309`, which becomes positive infinity despite the `parse_constant`
check. The new regression invokes that exact old source, records its acceptance
of these invalid cases, and requires the new source to reject all four witnesses.
All twelve dedicated IDs are checked; invalid Unicode, unexpected tokenizer
control IDs, duplicate JSON keys, nonfinite numbers and unknown envelopes fail
closed. This remains a deliberately restricted diagnostic subset of publisher
syntax, not a claim to implement every reasoning/action combination.

Source matching now executes the **hash-exact publisher ChatTemplate renderer**
against the original HF tokenizer and compares both token IDs and assistant
label masks. Only the reviewed class and its three declarative constants are
loaded from the pinned source AST; training/packing code is not executed.
The source's original role mapping and asymmetric host/model ownership are
unchanged. The publisher source revision remains descriptive provenance, not
an attestation of the historical training build; this caveat must not be
invented as a new gate for unrelated model lanes.

## Fixed diagnostic before first runtime

`scripts/ncr_001e_sprocket_wire_diagnostic.py` fixes seven NEW synthetic cases:
a tool request, two counterfactual tool results (42/17), two counterfactual
host memories (amber-17/violet-28), a memory-write request, and a memory-read
request. These do not replace or open any preregistered held-out fixture.

Each case executes disabled, zero-readout and manually activated **untrained**
NCA modes: 21 greedy outputs total, maximum 64 new units each, stopping on
source IDs 0 or 5. All original prompts, encoded IDs, masks, raw generations,
stop reasons, timing, source/library/runner hashes and negative observations are
saved before assertions. No tool is executed. No response is repaired. Literal
presence is explicitly **not semantic correctness**. There is no quality pass
threshold and no tuning after seeing the outputs.

The rich multi-turn source example also undergoes full-forward versus genuine
one-token-at-a-time cache comparison at **every** input position, including
both host and assistant memory-read occurrences. Direct backbone versus zero
NCA logits must be exact; generation/request isolation, hook cleanup and
before/after frozen tensor hashes must hold. Fixed 5e-4 absolute/relative cache
gates are unchanged. No optimizer is instantiated or stepped.

The workflow skips a new model invocation for archive-only updates, while
still rechecking guards and archived evidence. A green archive check is not a
fresh model run. First model receipt and any partial failure output must be
promoted to `datasets/ncr-001-sprocket-wire-diagnostic/` under the filing policy.

## Completion boundary

Passing this finite package finishes the repaired serializer, executed-source
parity and bounded observed-output diagnostic. It does **not** establish general
tool/memory competence, final independent #72 acceptance, optimizer-step
qualification, held-out comparative quality or authorization to merge/train.
Chat keeps its own frozen checkpoint and inherits no Instruct action result.
SmolLM2, void.1/Byte-EOS and matched-MLP controls retain their existing gates.
A reviewer must assess the final exact source/evidence commit separately;
implementation-session checks are not a substitute for that independent review.
