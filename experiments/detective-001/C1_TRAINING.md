# DETECTIVE-001C1 — first detective-specific witness-policy continuation

Parent issue: [#111](https://github.com/techrote/neurounner/issues/111). This is the first authorized training after the DETECTIVE-001A/B source and prompt-control studies.

## Scope

Continue exactly two immutable `village-001-sft-v1` SmolLM2-135M LoRA cartridges:

- V-G02-A → clock-damaged/protective witness policy;
- V-G09-C → transactional/social-scavenger witness policy.

Both models see the same 48 detective inputs; only the learned targets differ. Sixteen byte-exact parent-corpus anchors/member are added to resist gross loss of the original temperament. The model-facing inputs contain natural-language admitted memories and player questions only — no member IDs, relation labels, canonical truth ledger or evaluation fields.

## Relation semantics

`truthful_known_fact`, `corrupted_belief`, `intentional_lie`, `withheld_fact`, `contradiction_repair`, and `uninformed_speculation` are **orchestrator-only annotations**.

- A lie binds a true fact known to the witness and a different target claim.
- A corrupted belief places the false belief into admitted memory; the model is not instructed to lie.
- A withheld fact remains in admitted knowledge but the target omits it.
- Speculation targets explicitly refuse to elevate unsupported rumor into evidence.
- Contradiction repair supplies a physical record and expects the witness to correct its memory.

These labels make model failure types measurable rather than calling every wrong statement a character feature.

## Training

Per member: existing LoRA continuation, frozen SmolLM2 base, 64 examples, one epoch/64 optimizer steps, LR 1.5e-4, assistant-only loss, max length 512, four CPU threads, <=30-minute hosted lane, no GPU, no paid API, no sweep.

The original parent adapters are never overwritten. Final continuations receive new directory/archive hashes and a neutral `detective-001c1-v1` release even if evaluation is weak.

## Learner-final custody

The repository contains only the generation recipe before training. The exact 12 final prompts do not exist until both trained adapter directory hashes are known. Their seed binds the recipe, training-dataset manifest, both immutable parent hashes and both trained hashes.

Three final case families are disjoint from the four training families. The exact generated prompts are collision-checked against the new detective training inputs and both original parent Village corpora. Parents and trained continuations answer the identical materialized final suite.

## Mechanical scoring and limits

Scoring measures required proposition markers, protected/forbidden marker leakage, false-vs-true target selection for deliberate lies, correction after physical contradiction, expected withholding/uncertainty phrases, truncation, repeated sentences and accidental evidence-ID leakage.

This scorer does **not** prove human-perceived characterization quality. The issue's 0–4 reference task-relevance gate remains explicitly `UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE` in this zero-paid-API experiment. Consequently a larger-cast expansion is never automatically admitted by this run, even if both relation-policy scores improve.

Four old Persona/Village T prompts are retained as raw parent-vs-trained reference responses for later independent relevance review; they are not part of the new learner-final policy score.

## Evidence

The workflow must persist:

- deterministic `datasets/detective-001c/` training archive + hashes;
- both 64-step result/loss/throughput records;
- exact post-training final suite and its seed/source bindings;
- all parent/trained final responses and four reference responses per lane;
- deterministic comparison summary;
- neutral trained adapter release manifest/assets;
- source Actions run and branch head.

Weak results are evidence. Do not regenerate a character until it passes. Expansion to the rest of the village requires a separate measured decision.