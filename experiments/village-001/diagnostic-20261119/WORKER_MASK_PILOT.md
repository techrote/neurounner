# VILLAGE-001F: matched worker-mask pilot

## State and purpose

This is a prepared, bounded follow-on to Stage E. No pilot model inference,
provider judging, or optimization was performed while preparing this plan.
Original C/D evidence and all frozen E sources remain unchanged. Execution is a
separate, explicitly selected workflow after Stage E reaches a terminal result.

The pilot asks whether preserving real chat-turn boundary tokens during
preference training improves task quality. It also measures where quality changes
across BASE, C SFT, original D, reproduced D, and corrected D. The masking defect
is confirmed from source; its contribution to weak behavior is not yet measured.

## Confirmed source defect

The exact pinned model's [primary tokenizer configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer_config.json)
defines both its padding token and end-of-turn token as `<|im_end|>`, ID 2.
The complete retrieved bytes and URL/date/hash receipt are retained in
`worker-source-tokenizer-config.json` and `worker-tokenizer-source-receipt.json`.
The byte SHA-256 is
`4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df`.

The historical `scripts/village_001d_preference_train.py:pair_batch` creates its
attention mask by comparing every token ID to the padding ID. This zeros actual
system, user, and assistant turn boundaries as well as appended batch padding.
Stage C instead uses all-ones attention for each unpadded sequence. D training and
D's reported before/after preference log-probabilities both use the historical
mask. The pilot's corrected arm derives attention from each sequence's true
length, preserving all real tokens and zeroing only appended padding.

The tokenizer audit will retain every native token ID, label, rendered training
sequence, both attention masks, and the exact positions incorrectly masked by
the historical rule. It covers all 720 chosen/rejected sequences in the six
members' 360 pairs. Each chat template contributes at least three real boundary
tokens; actual positions and counts will be measured, not substituted with this
template-derived expectation.

`worker-source-audit.json` records the separate source checks. The inspected
assistant target shift is consistent; C/D reject overlength training examples
instead of silently truncating them. No wrong-template finding is established.
The source also contains a latent reporting issue on partial D resumes:
`preference_before` is then evaluated at the resumed checkpoint. None of the 36
archived original D results was resumed, so this does not explain that population.
New pilot receipts explicitly disclose this condition if a pilot resumes.

## Predeclared members and sole training intervention

The panel is fixed independently of E scores:

| Member | Lineage | Sibling |
|---|---|---|
| V-G01-A | G01 | A |
| V-G03-B | G03 | B |
| V-G05-C | G05 | C |
| V-G07-A | G07 | A |
| V-G09-B | G09 | B |
| V-G11-C | G11 | C |

There are two members of each sibling type and six distinct lineages. Findings
apply directly to this panel; it is not a population-wide qualification.

Each member starts twice from the same exact archived C adapter and trains on
the same original 60 preference pairs for the same 120 optimizer steps. Parent
bytes, pair bytes, member seed, order, token IDs, labels, optimizer, learning
rate, objective, gradient clipping, and runtime are fixed. The only optimization
variable is the attention mask:

| Arm | Attention rule |
|---|---|
| historical_token_id_mask | Existing D token-ID comparison |
| length_derived_mask | One for every real sequence position; zero for appended padding |

The implementation invokes the unchanged D engine through an F wrapper. It
replaces only `pair_batch` in the corrected arm. Both arms omit the old engine's
post-optimization T-response generation. This omission occurs after all
optimizer steps and does not change either optimization trajectory. The two arms
use the same separate Q inference program instead. New terminal results identify
themselves as `VILLAGE-001F`; the original D source and original D result hashes
remain explicit references.

Runtime is pinned to torch 2.14.1+cpu, transformers 4.57.6, and peft 0.21.2. The
base is SmolLM2-135M-Instruct at revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`.

## Evaluation reserve and workload

`worker-eval.json` contains twelve newly authored Q prompts, two from each of the
six campaign families. They were frozen before any pilot optimization and before
using E outcomes to choose a panel or prompts. Their SHA-256 is
`126ee2a159db08d4cdf6452d272af9f921a22f28f6298bd8f8c59b165d53ee9b`.
Exact normalized matching found zero collisions against 3,407 distinct training,
T/P, and existing E-F prompts. E prompts were read solely for this exclusion;
no E outputs or scores are training targets or prompt-selection inputs.

All Q prompts remain evaluation-only. A shared BASE lane produces twelve
responses. C, original D, reproduced D, and corrected D each produce twelve
responses for every panel member: 24 adapter lanes and **300 responses total**.
Every lane uses the same public system instruction, greedy decoding, 160-token
limit, all-ones attention on the unpadded native prompt, and no member/persona
instruction. Native input/output token IDs, rendered prompts, timing, and full
checkpoint chains are retained.

The full bounded worker pilot is:

- 12 training lanes × 120 steps = 1,440 completed optimizer steps.
- 25 inference lanes × 12 prompts = 300 responses, at most 48,000 generated tokens.
- One tokenizer-only audit of 720 existing chosen/rejected sequences.
- 18 fixed-weight scoring lanes: BASE, C, and original D for each member. Eight
  existing pairs per member, under both masks, give 288 forward batches and zero
  optimizer steps or generated responses.

The eight scoring pairs cover one T case from each family plus one P case from
each sibling source, selected by a fixed hash without response/score selection.
These inputs participated in D training. Their likelihoods are explicitly
training diagnostics and are not generalization scores.

## Blinding, baseline reproduction, and interpretation

The primary comparison is corrected D versus reproduced D on 72 matched Q
member/case pairs. Presentation order uses opaque identifiers; judges do not see
the mask arm, model stage, member ID, optimizer metrics, or expected answer.
Contextual BASE/C/original-D scores locate earlier quality loss. The planned
judge workload is 18 requests containing four primary pairs each and 26 requests
containing six contextual responses each: 44 baseline requests and a hard limit
of 88 logical requests including missing-only repairs. Final judging code and
receipts enforce their own source binding and transport budgets.

Baseline reproduction compares original and reproduced D adapter tensor
fingerprints, per-tensor identities, current-runtime Q responses, and diagnostic
pair scores. Directory/archive hash differences alone cannot establish weight
differences because README or configuration bytes can differ. Both inference and
training receipts retain the safetensors-file hash and a canonical hash over
sorted tensor names, shapes, dtypes, and raw tensor-byte hashes. A baseline
mismatch must be reported and limits attribution to the original D campaign.

Task relevance is the primary quality metric. Coherence, groundedness,
genericness, paired preference, and member-level regressions provide context.
All 72 differences are retained; six member means are the replication units.
The preregistered working effect threshold is a mean relevance gain of 0.5/4,
positive gains in at least four members, and no member mean decline exceeding
0.5/4. This is a decision aid, not a new definition of village qualification.
There is no automatic 36-member rollout.

Historical T scores already show low relevance at C for this panel (about
0.92–1.00/4). Therefore the masking defect cannot be assumed to explain all poor
behavior. A weak or negative masking result is useful evidence; it directs
attention toward earlier training data/objective effects without replacing
failed members or rewriting their history.

## Commands and artifacts

The driver is `scripts/village_001f_worker_diagnostic.py`; every command accepts
`--config experiments/village-001/diagnostic-20261119/worker-config.json`.

| Command | Purpose | Main output |
|---|---|---|
| plan | Verify all frozen sources, panel, pair/parent bindings, and prompt exclusions | worker-plan.json |
| self-test | Offline contract, synthetic mask boundary, and mocked actual train-wrapper checks | stdout |
| token-audit | Tokenizer-only native mask and label audit | token-mask-audit.json |
| train | One explicitly selected member and mask arm | adapter/, result.json, training.json, training-identity.json |
| infer | One member/stage or the shared BASE | result.json, generation-identity.json, rows/ |
| score | Both attention masks at fixed BASE/C/original-D weights | score.json |

Runtime outputs must be outside repository source/evidence directories. Training
and inference support `--resume` into a new empty output directory. Full output
trees, including identity and checkpoint files, must remain together. Adapters
belong in separate release assets, not ordinary Git history. Dataset/config/
report custody is separate from model-weight storage.

New adapter inference requires `--training-evidence` so the correct member,
arm, config, completed step count, result hash, and adapter bytes are verified.
Original C/D adapters are checked against their frozen archived results. The
wrapper never writes the old D manifest or original C/D evidence.
