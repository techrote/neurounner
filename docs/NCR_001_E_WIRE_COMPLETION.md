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
fresh model run. The first original diagnostic's seven-case plan, complete 21 generated outputs,
original raw prompts and masks, source-prefetch metadata, package inventory,
historical negative reproduction and test log have now been preserved byte-for-byte
under [the durable diagnostic archive](../datasets/ncr-001-sprocket-wire-diagnostic/README.md).
The original full artifact ZIP and binary source tar remain in Actions retention
only; their exact SHA-256/size and the immutable Git source commit are recorded.
Never describe those unarchived binaries as preserved in Git.

## Completion boundary

Passing this finite package finishes the repaired serializer, executed-source
parity and bounded observed-output diagnostic. It does **not** establish general
tool/memory competence, final independent #72 acceptance, optimizer-step
qualification, held-out comparative quality or authorization to merge/train.
Chat keeps its own frozen checkpoint and inherits no Instruct action result.
SmolLM2, void.1/Byte-EOS and matched-MLP controls retain their existing gates.
A reviewer must assess the final exact source/evidence commit separately;
implementation-session checks are not a substitute for that independent review.

## Original diagnostic readback and independent audit — 27 October 2026

Source execution [Actions 40573188117](https://github.com/techrote/neurounner/actions/runs/40573188117)
ran on full pinned root Sprocket Instruct (original source commit
`905e237136e4ea75208a8ed92392489a9f62849d`). Original
`receipt.json` is 75,464 unchanged bytes, SHA-256
`94b1ff4f3ee4cb19d3e1e8cd57f8e9e7eee9534503b87a86d1aa6c5951de3b45`.
Actions ZIP SHA-256 `7e38a9c8438701c460597fc87016f3a26ece2d12b97482cfbcb44b0e6974a2ab`.
Text original bytes are compressed in a deterministic USTAR/xz envelope with
SHA-256 `982dfaee356bbcdce705194413a71ace1ce7787c4421ea017dd8b95e8aec2921`.
The exact 68,602-byte original source tar is not committed (SHA independently
recorded). Original source files remain in immutable Git history.

- Full 108-position genuine one-token cache maximum absolute logit deviation:
  `3.590583801269531e-4`, below the unchanged 5e-4 gate.
- Seven synthetic cases × three modes: **21 raw originals**; frozen/zero generation
  identical, no strict tool envelope or memory action, five cases exhausted the
  64-unit budget. Counterfactual host tool result `42` was absent from output,
  `17` was present, and both synthetic memory labels appeared. These are
  *descriptive observations*, not behavioural quality PASS criteria.
- [First archival readback CI 40584453555](https://github.com/techrote/neurounner/actions/runs/40584453555)
  **FAILED** because three altered *manifest* claims (training, tool semantics,
  final model acceptance) were not rejected; the original receipt itself remained
  valid. [Commit `88cf5df`](https://github.com/techrote/neurounner/commit/88cf5df383556ec28246cbb93755014fb91bde95)
  fixed the direct semantic verifier while retaining the failure evidence.
- [Follow-up Actions 40584618404](https://github.com/techrote/neurounner/actions/runs/40584618404)
  **PASS**, all nine original-byte/false-promotion test groups. It correctly
  reported `FRESH_MODEL_DIAGNOSTIC_REQUIRED=False`: this is an archive verifier,
  not another actual frozen-host execution.

The durable [dataset manifest](../datasets/ncr-001-sprocket-wire-diagnostic/manifest.json),
[original JSON archive](../datasets/ncr-001-sprocket-wire-diagnostic/original-text-members.tar.xz.b64),
[original artifact/runner provenance](../datasets/ncr-001-sprocket-wire-diagnostic/provenance.json)
and [standard-library audit](../scripts/ncr_001e_sprocket_wire_archive.py) are
available for independent reviewer custody. None authorizes comparative training,
opens a sealed fixture, qualifies general model-generated tool/memory behavior, or
changes `train_authorized=false`. PR #102 and every upstream dependency remain
**draft, unmerged, pending independent final #72 review**.
