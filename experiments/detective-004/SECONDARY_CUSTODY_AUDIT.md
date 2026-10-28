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

## Claimed full-seed binding without adapter attestation

The full Stage-A posthash derivation is SHA-256 over canonical JSON of the
frozen recipe digest, the verified training manifest digest, a supplied
adapter-directory SHA-256 and the frozen salt. The existing exact-record
structural guard checks an internally consistent **10-hex** case suffix, but
**cannot establish that the suffix came from this posthash derivation**.

The separate offline `audit_untrusted_posthash_claim(tasks, claim, b3)`
now checks a supplied claim with **exactly** these three 64-hex fields:
`training_manifest_sha256`, `adapter_dir_sha256`,
`post_hash_seed_sha256`. It independently reloads the exact frozen
source/config/B3 authority, re-verifies the *committed* training archive
byte-for-byte, recomputes the full seed from the pinned recipe, compares the
claimed 64-hex seed and all 12 case suffixes, and invokes the existing
full-record frozen-recipe equality guard. It never calls the scorer,
loads weights, inspects a real adapter or opens/mints a real posthash final.

The function returns **`claim_consistent_only`**, with explicit false flags
for adapter-source/optimizer receipt verification, independent final custody,
model input/token-ID verification, independent Stage-A approval, training,
production and population admission. This is an intentionally *incomplete*
future-custody primitive. An attacker can supply a mutually consistent fake
adapter hash and matching synthetic seed; this function detects contradictions
*between claims*, not dishonest or unverified adapter ancestry.

New offline tests use **made-up adapter digests**, recomputed dummy full seeds
and 12 **unsealed synthetic unit-test rows**, never the future post-trained
learner-final. They reject altered training manifests, mismatched adapter-hash
claims, incorrect/short full seeds and fully coherent final task suites made
under another prefix. A self-consistent fake claim is deliberately retained
as a positive demonstration of the strict non-admission result.

### Independent custodian's remaining obligations

Before any future final can count as research evidence, a distinct reviewer
must verify the *actual frozen model weights* and tokenizer source/checksums,
confirm the real tokenizer input IDs, source-native masked token labels and
512-token cap, inspect the real adapter files/PEFT configuration and its
original frozen-root ancestry plus immutable optimizer receipts, and derive
the posthash seed using the **measured** adapter directory hash. The reviewer
must preserve exact raw model-visible prompt bytes, prompt token IDs, generated
IDs/text, base and adapted outputs, and the 12-task posthash evidence identity.
Never use a gold training `rendered` string or `target` record as inference
input. The complete source-native training masks, role delimiters, end token
and no-host-target rule need independent inspection against the pinned
publisher SFT implementation before authorizing Stage B.

No independent approval has been obtained, and this author-side hardening
must not be represented as training or learner-final authorization. Draft
PR #149 remains review-gated; separate blind relevance issue #121 still blocks
population expansion. All Stage-A generator/config/source Git blobs, committed
36-row corpus, manifest, archive hashes and historical model failures remain
unchanged.

## Native control-token completion boundary (future raw scoring)

A further pre-admission evaluator weakness was found in the frozen Stage-A
`protocol.score_output`: non-tool restraint and result-continuation checks
reject a hand-written subset of role/action markers but **do not reject**
the publisher's native `<|pad|>` or `<|endoftext|>` control tokens.
Consequently, a raw `<|pad|>` response can count as **1/1 restraint**
despite containing no meaningful answer; a valid-looking result continuation
can append `<|endoftext|>` without invalidating its score. The frozen evaluator
is left byte-identical because its Git blob is bound to the committed Stage-A
archive, and its historical raw/scoring record must not be rewritten.

The **separate** read-only `audit_raw_native_controls` now checks raw
completions in **both adapted and exact root-control cohorts** using the
pinned B3 `native.CONTROL_RX`, not a duplicate informal token list.
For tool-selection outputs it permits one leading `<|tool_call|>`; for
all output kinds it permits one terminal `<|end|>`, only in its final
position. Any unexpected `<|pad|>`, `<|endoftext|>`, duplicated
yield, hidden role marker or misplaced native control refuses the entire
`guarded_admission_gate` call **before scoring**. The existing strict
B3 parser and unchanged source-bound research thresholds remain authoritative
for actual action semantics.

Three synthetic adversarial test methods demonstrate the historical
restraint/result false-positive behavior, rejection by the new guard for
both adapted and root baselines, and acceptance of legitimate trailing
yield tokens. These cases use only the explicitly **unsealed synthetic**
test fixture; they never load an adapter, construct a real posthash learner
final or score actual model generations.

The guarded scorer's return now explicitly marks
`raw_output_execution_verified=false`,
`independent_final_custody_verified=false` and
`stage_b_evaluation_admitted=false`, including when synthetic outputs
produce `status=research_candidate`. Neither these booleans nor syntactic
raw control checks authenticate the executed prompts/token IDs or the
actual frozen-root / trained-adapter lineage. Independent Stage-A review
and subsequent Stage-B token/runtime/optimizer/final custody gates remain
mandatory. No optimizer, inference, hidden relevance access or population
expansion is authorized here.

## Fail-closed unverified score envelope and raw-output claim ledger

The earlier `guarded_admission_gate` used to propagate the frozen scorer's
`status="research_candidate"` as its own top-level status, even when its
inputs were **invented synthetic gold completions** with no model execution
or independently authenticated final. The separate custody auditor now
**never returns a top-level research-candidate status**. Its top-level
`status` is always `unverified_score_only`; the original frozen
scorer's exact decision and numerical results remain accessible as
`mechanical_gate_status`, `mechanical_gate_passed` and the original
aggregate counts. The source-bound evaluator, thresholds, generator,
manifest and archived corpus remain untouched. Downstream consumers must
not treat a mechanical pass as genuine research/model admission.

For each of the 12 **supplied** task records, the audit also returns a
stable, task-ID-sorted `claimed_raw_output_ledger` with the task kind,
separate SHA-256 digests of the exact **UTF-8 strings supplied** as adapted
and root-control responses, and the original strict scorer's two booleans.
This exposes task-level changes and failure counts without rewriting raw
texts. It does **not** authenticate the source of those strings; hashing
fabricated outputs proves only their identity, not that any checkpoint
generated them. A baseline's ordinary no-tool prose can legitimately
pass the restraint criterion even if its tool-selection score is zero.
All returned custody and admission flags remain false.

Two additional regression methods use the existing 12 **unsealed synthetic
test objects** to verify exact digest/score reconciliation, deterministic
ordering, and that changing one fabricated response changes its digest and
its scored outcome without promoting the result to an admitted candidate.
They do **not** generate or open a real posthash final, load models,
consume hidden relevance ratings or train a new adapter.

### Independent reviewer handoff — acceptance questions, not sign-off

An independent reviewer of [draft PR #149](https://github.com/techrote/neurounner/pull/149)
should inspect and record distinct review evidence addressing:

1. Whether the root Instruct weight/tokenizer hashes and publisher source
   blobs in `specialization.json` match the frozen B3 authority and
   publisher-native SFT control semantics, without substituting Chat weights.
2. Whether all 36 committed rows are regenerated byte-for-byte, the six
   training families are disjoint from the three recipe-only final families,
   and actual model-visible user/host-only tool messages never expose
   hidden case authority or gold assistant targets.
3. Whether assistant content **and** closing `<|end|>` are the *only*
   intended trained regions. Stage A has **character spans only**, not
   frozen-tokenizer label arrays; real 512-token checks remain **unverified**
   and are a pre-optimizer Stage-B prerequisite.
4. Whether the future final would be materialized only after authentic
   adapter/source/optimizer custody, from the frozen recipe and measured
   hash, with actual executed model-only prompt bytes/token IDs and
   all malformed/failing raw outputs preserved.
5. Whether the supplemental layout/full-seed-claim/native-control guards
   and this **unverified-only score envelope** are treated as author-side
   synthetic checks, not as substitute independent approval or proof
   of a trained model's research qualification.

No reviewer approval is represented by this checklist. Source-matched
offline CI, archive verification and author-side adversarial tests do
not satisfy the distinct-review requirement. Until that review is submitted
and Stage A merges, **Stage B is unauthorized**. Even after Stage A
approval, actual tokenizer/mask/length and adapter/final custody must
separately pass before optimization or final interpretation. Independent
DETECTIVE-001R1 opaque relevance ratings are still required for any
population expansion.


## Publisher-native token supervision preflight (prepared, not executed with weights)

A source-level readback of the exact publisher SFT implementation,
[kandivault-source/Sprocket-500M \`src/train/sft_data.py\` at commit
\`a003767e70d766bc14b40c89792543ee8641c8bc\`](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/src/train/sft_data.py),
confirmed Git blob \`90bddf8976b9d6410fe7c5c1f084708e1cebab73\`.
Its \`ChatTemplate.render\` inserts each role header and each closing
\`<|end|>\` as pinned **individual special-token IDs**, while tokenizing
*each turn's content separately*. Only assistant content and assistant yield
\`<|end|>\` are supervised. The publisher's shifted causal target is the token
**after** the input position, with host/header positions ignored. A byte-perfect
character-span audit does **not** prove that this mask survives actual
tokenization or that a concatenated native inference prompt yields the same
token IDs as separate-turn publisher rendering.

New separate \`scripts/detective_004a_token_preflight.py\` implements an
explicit, **read-only** check for this future pre-optimizer requirement:

- Source-bound regeneration and custody audit of all **36 committed** training
  rows, without opening or generating any posthash final.
- Pinned 12 special token IDs; source-native **per-turn** content encoding,
  assistant-header/host masking, trained assistant content and closing yield;
  native tool-call ID 6 trained and host tool-result ID 7 masked.
- Explicit shifted next-token labels; exact **512-token** limit per row, with
  refusal rather than silent truncation.
- Bit-exact equality between the future B3 **whole-wire** inference-token IDs
  and the publisher's turnwise prompt-token IDs, including the final dangling
  assistant cue; mismatches fail closed rather than being called model behavior.
- Deterministic per-row hashes of token IDs, token mask, causal labels,
  exact model-only prompt bytes and prompt IDs. The future explicit local
  invocation also preserves the **complete raw arrays** (input IDs, train mask,
  causally shifted inputs/labels and prompt IDs/text) in its receipt; hashes
  alone cannot replace raw token-ID receipt custody.

The Stage-A CI exercises these functions **only with a deliberately artificial
toy tokenizer**. The toy is NOT Sprocket, is NOT a verified model input, and
cannot establish the 512-token bound for the actual checkpoint. It tests the
preflight machinery, including deliberate wrong special IDs, incompatible
whole-wire encoding, oversize input and wrong on-disk tokenizer hashes. No
tokenizer is downloaded or model/optimizer invoked in this Stage-A work.

**Future explicitly authorized local-only check, not an instruction to start
training:**

\`\`\`sh
python3 scripts/detective_004a_token_preflight.py \
  --tokenizer-dir /path/to/already-local/root-instruct-tokenizer \
  --output /new/custody/location/frozen-token-preflight.json
\`\`\`

The optional command refuses unless an already-local \`tokenizer.json\`,
\`tokenizer_config.json\` and \`special_tokens_map.json\` have the exact three
B3-pinned SHA-256 values, and the native \`tokenizers\` library is installed.
It never downloads weights or tokenizers, imports a model, or starts an
optimizer; an existing receipt cannot be overwritten. Actual results from
this command **have not been run or published here**. A later independent
custodian must preserve exact local source identities, raw token IDs,
loss-label arrays, runtime/library version and output receipts before an
optimizer may be authorized.

**This is preparatory implementation, not Stage-A approval or Stage-B
qualification.** PR #149 still needs an independent reviewer before merging.
Actual frozen-tokenizer loss labels and 512-token measurement remain
unverified until a genuine pinned-tokenizer run; source/adapter ancestry,
optimizer receipts, raw inference and posthash final custody are separate
gates. Witness relevance review #121 still requires independent external
scores. No training, model inference, sealed learner-final materialization,
hidden relevance-map access, production admission or population expansion.

## Future 36-step single-example input custody (source-only preparation)

The frozen Stage-B budget specifies **one pass over exactly 36 examples,
one example/optimizer step, 36 steps total, at most 512 tokens/example**.
This is **not** interchangeable with the publisher's `pack()` training helper,
which concatenates conversations with `<|endoftext|>` separators and cuts
fixed `ctx+1` blocks (possibly placing multiple conversations in a block).
Silently switching to packed batches could repeat, omit or mix admitted
examples while still reporting 36 optimizer steps.

The auxiliary `planned_single_example_steps(tok)` now derives 36 ordered
one-example input claims from the byte-verified committed Stage-A archive and
the exact selected tokenizer's token/mask/causal-label receipt. Its order must
match the committed manifest's 36 IDs; every item retains source-native
untruncated token length, trained label count and SHA-256 of full IDs,
assistant-only mask, shifted labels and **model-only** prompt IDs.
`audit_claimed_training_steps(tok, claimed_steps)` fails closed on omitted,
duplicated or reordered IDs, length drift, different token/mask/label/prompt
hashes, and any mismatch from the single-row input plan.

The Stage-A regression tests use **only the artificial toy tokenizer**, so
the positive 36-step plan is a mechanical fixture, not real Sprocket input
evidence. The helper returns `optimizer_execution_verified=false`,
`training_admitted=false`; the claimed-steps audit returns
`actual_optimizer_batches_independently_attested=false`,
`adapter_ancestry_verified=false` and `stage_b_training_admitted=false`.
Even a byte-perfect claim can be fabricated. A future independent custodian
must reconcile these *expected* inputs against separate, immutable **actual
optimizer step/batch execution receipts**, the verified frozen tokenizer and
actual adapter/root ancestry before any training or final qualification
could be considered. This author-side hardening neither invokes an optimizer
nor authorizes one.

The original 36-row JSONL, manifest, frozen generator/config, exact source
pins, posthash final recipe, prior raw model evidence and C1/C2/C3 sealed
witness finals were not modified. No real tokenizer, model, inference,
training, relevance mappings or population mutation was accessed.

## Frozen optimizer preregistration gap — explicit blocker, no guessed defaults

A further source-level review of the frozen Stage-B resource budget found that
it is **not yet a complete reproducible optimizer execution contract**. The
current `future_training` object fixes the root weight, LoRA rank/alpha/dropout,
`q_proj`/`v_proj`, one epoch / 36 optimizer steps, learning rate `2e-4`,
`AdamW`, gradient clip `1.0`, four CPU threads and the resource envelope. It
does **not** freeze several semantics that can materially change a short LoRA
run:

- training RNG seed;
- batch size;
- gradient-accumulation steps;
- AdamW beta1 / beta2;
- AdamW epsilon;
- weight decay;
- learning-rate scheduler;
- warmup steps.

These omissions must **not** be silently filled from whichever PyTorch,
Transformers, PEFT or training helper defaults happen to be installed later.
The Stage-A config, generator and durable training manifest are intentionally
left byte-identical here; choosing values is a protocol decision for the
required independent review / subsequent explicitly frozen execution contract,
not an author-side cleanup.

`scripts/detective_004a_token_preflight.py` now exposes
`audit_optimizer_preregistration()`. It mechanically reports the missing
fields and keeps all execution, adapter, production and population authority
false. `planned_single_example_steps()` and
`audit_claimed_training_steps()` propagate the same blocker, so a perfect
36-row token/mask/input claim cannot be mistaken for optimizer readiness.

Three synthetic/offline regression methods verify:

1. the current frozen config is reported incomplete with the exact missing
   field list;
2. the 36-step input plan and a structurally exact claimed-step ledger retain
   the optimizer-preregistration blocker; and
3. even a **synthetic test-only** object containing all field names receives
   no independent-review, optimizer-execution, adapter or training authority.

This finding does **not** retroactively alter the 36-row corpus, manifest,
source hashes, historical model evidence, sealed witness finals or posthash
final recipe. It does not select optimizer defaults and does not start Stage B.
Before any optimizer invocation, an independently reviewed execution contract
must resolve these semantics, the real frozen tokenizer/mask/512-token check
must pass, and actual root/adapter/optimizer custody must be established.

## Source-export LoRA projection mapping and untrusted optimizer-claim hardening

An exact-read of the independently pinned publisher source commit
[`a003767e70d766bc14b40c89792543ee8641c8bc`](https://github.com/kandivault-source/Sprocket-500M/tree/a003767e70d766bc14b40c89792543ee8641c8bc)
resolved a material naming ambiguity in the frozen LoRA budget:

- Native [`src/model/model.py`](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/src/model/model.py)
  (Git blob `0e3aa3af55d75ec0c02c8abd9e6e5628df659101`) names its attention
  linear projections `attn.wq` and `attn.wv`, **not** `q_proj`/`v_proj`.
- The same commit's [`scripts/export_hf.py`](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/scripts/export_hf.py)
  (Git blob `2f4f3c712b7e12ddab97e4755d92252f634c31c6`) explicitly maps
  `blocks.{i}.attn.wq.weight` to
  `model.layers.{i}.self_attn.q_proj.weight` and
  `blocks.{i}.attn.wv.weight` to
  `model.layers.{i}.self_attn.v_proj.weight`, then exports
  `LlamaForCausalLM` for Transformers. Thus the Stage-A
  `target_modules=[q_proj,v_proj]` are **source-compatible with the HF export**
  used by the B3 `AutoModelForCausalLM` load path, rather than with the
  native unexported GPT class.

**Limit:** neither public publisher file proves which exact exporter built the
pinned root `model.safetensors`, which module names actually occur in that
source-frozen loaded checkpoint, or whether a specific PEFT version attaches
rank-8 adapters only to the intended Q/V projections. The publisher already
labels its code as not a checkpoint build attestation. Those checks require
a later exact-weight, read-only runtime module-name/config inventory and
independent review before an optimizer step. No weight or tokenizer was
downloaded for this source audit.

The previous auxiliary optimizer-gap reporter also accepted a caller-mutated,
syntactically filled `future_training` dictionary and returned
`optimizer_execution_contract_complete=true` despite the immutable on-disk
Stage-A config remaining incomplete. That positive field could mislead later
automated consumers even though all admission flags were false. It now
**compares the entire candidate config to the verified frozen contract**;
only that exact immutable source can possibly be marked contract-complete.
It additionally reports invalid numeric/batch/scheduler parameter claims
separately from missing field names, requiring single-example batch size
and accumulation of one under the 36-example/36-step constraint.
Malformed/noncanonical supplied dictionaries fail closed to an untrusted
override report, rather than becoming evidence that the source was amended.

Two additional purely synthetic regression tests verify a plausible but
source-unbound filled configuration still returns incomplete and a list of
invalid values (including nonfinite beta, zero epsilon, negative weight
decay, bad batch/accumulation and out-of-range warmup) cannot pass domain
checks. The previously existing synthetic test is updated to refuse
`optimizer_execution_contract_complete=true`. No optimizer defaults were
chosen; independently reviewed Stage-A approval and a separately frozen
Stage-B execution contract are still mandatory. Corpus and source pins,
raw historical model evidence, final recipe and sealed witness-finals
remain untouched.

## Native GPT versus exported HF Llama causal-loss custody

**Additional concrete incompatibility discovered in the frozen source.**
The publisher's native SFT example arrays and the export-target Hugging Face
Llama loss API do **not** accept interchangeable target tensors, even if
publisher role masks and token IDs are otherwise correct:

- Pinned publisher [`src/train/sft_data.py`](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/src/train/sft_data.py)
  (Git blob `90bddf8976b9d6410fe7c5c1f084708e1cebab73`) packs/externally
  shifts labels: `X=ids[:-1]` and `Y[i]=ids[i+1]` for supervised next
  tokens, otherwise **`-1`**. Pinned native
  [`GPT.forward`](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/src/model/model.py)
  (Git blob `0e3aa3af55d75ec0c02c8abd9e6e5628df659101`) consumes these
  externally shifted labels with `cross_entropy(ignore_index=-1)`.
- The same publisher's pinned
  [`scripts/export_hf.py`](https://github.com/kandivault-source/Sprocket-500M/blob/a003767e70d766bc14b40c89792543ee8641c8bc/scripts/export_hf.py)
  (Git blob `2f4f3c712b7e12ddab97e4755d92252f634c31c6`) exports
  **`LlamaForCausalLM`**, and frozen B3 uses
  **`AutoModelForCausalLM`**. In upstream Transformers
  [v4.57.6 `ForCausalLMLoss`](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py)
  (Git blob `5db1c3f3e9fde23567c0e67c9d9fcb9bd5b9d1b2`), the default
  ignore index is **`-100`**, and the loss function performs the
  next-token shift **internally**; the tagged
  [Llama forward](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py)
  (Git blob `f43a5fc9b5237b852cbc0db5bc333711797eaa97`) dispatches its
  supplied labels to that loss function.
- Therefore feeding the publisher's native `Y` vector into HF Llama
  `forward(input_ids=..., labels=...)` could **double shift** supervision
  and incorrectly use **`-1` instead of `-100`** on masked targets.
  The HF contract needs **full, unshifted input IDs** and an equal-length
  aligned label vector: token ID at supervised positions, otherwise `-100`.
  Its internal one-position shift must then equal the rebased native
  next-token targets, with a final ignored position.

A new **offline, read-only** `hf_forward_supervision()` in
`scripts/detective_004a_token_preflight.py` constructs the HF-form label
array from the same source-native role mask and mechanically checks that its
internal effective shift agrees with the publisher's causal targets. A caller
can supply claimed HF labels, which fail closed on wrong masks, native `-1`,
already shifted vectors and host-role training. `inspect_row` preserves
**both raw label representations** and their separate canonical SHA-256
receipts. Future 36-step input claims also bind the HF forward-label and
effective-shift hashes; a single `shifted_labels_sha256` is insufficient to
qualify the HF optimizer interface.

**Scope/authority:** Three additional regression methods use only the
artificial tokenizer and do not execute either GPT or Llama. Upstream tagged
source is a reference, **not a receipt of the actual loaded Transformers
version or loss path**. The exact Stage-B optimizer interface/backend,
real tokenizer, 512-token bounds and executed tensor/loss receipts must
still be independently approved/attested. This work does not choose the
backend or settle the previously recorded incomplete optimizer parameters.
It leaves the frozen 36-example archive, manifest/config/generator Git blobs,
posthash-final recipe, raw historical evidence and sealed learner-finals
unchanged. No training, inference, adapter, final, hidden relevance map or
population access was performed.

