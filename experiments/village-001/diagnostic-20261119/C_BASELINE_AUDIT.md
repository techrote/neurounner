# Original C baseline: read-only audit

This audit examines original VILLAGE-001C training, its source corpus and its archived validation responses. It does not analyze E, F or Q outputs, change a protocol, or establish a causal explanation for low task relevance. No model, provider or optimizer work was performed.

## Source identity and implementation

The [C provenance receipt](../sft-evidence/provenance.json) identifies source commit `e1a45d282e3a9d34c7ed2eb4a27e7f73ff35e73e` and workflow run `40227472424`. The current C trainer, campaign configuration and PERSONA evaluation configuration match that commit. All 36 current corpus-file hashes match their original C result receipts.

No objective, masking, truncation or template implementation defect was confirmed. In [the C trainer](../../../scripts/village_001c_train_member.py), `encode_example()` constructs native system/user/assistant messages, masks the common prompt prefix with `-100`, supplies all-ones attention for an unpadded example, and rejects lengths above 512 instead of truncating. Training and evaluation use the same neutral system prompt. Independent review of the pinned [tokenizer](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/12fd25f77366fa6b3b4b768ec3050bf629380bac/tokenizer_config.json), Transformers 4.57.6 [causal loss](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py) and [generation attention](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py) confirmed the assistant header, internal causal-label shift and unpadded generation-attention behavior. C did not archive native token IDs, so this is not a per-example token audit.

C trains only the positive `target`. The `generic_bad_target` and `why_bad` fields describe curation choices; they are not negative-training terms. The recipe is 160 examples, two epochs, 320 batch-one optimizer steps, constant learning rate `0.0008`, and LoRA rank 8 across seven projection types. It contains no separate task-quality or reference-retention term. See [campaign configuration](../config.json) and the trainer's `encode_example()` and optimization loop.

## Confirmed curriculum and acceptance tension

The [B compiler](../../../scripts/village_001b_generate_member.py) explicitly requests useful, awkward and suboptimal behavior, including failure-prone tendencies. Its main critic gates behavior fidelity, genericness and style dependence, without a separate task-relevance score. All **2,160** archived expansion reviews pass at round 0; **1,779** have scores **4/0/0**. There are **zero adjudicator records**, so the sparse adjudicator's explicit task-relevance condition was not exercised. These counts come from `datasets/village-001/quality/members/*/{critic,adjudicator}.json`.

| Positive training example | Concrete evidence |
|---|---|
| `V-G11-A-E120` | The character stretches for a box, declines to fetch a stool, catches a watch chain and leaves a corner stuck. Its critic passes it at 4/0/0, citing the refusal and incomplete task. [Corpus](../../../datasets/village-001/member-corpora/V-G11-A.jsonl); [critic](../../../datasets/village-001/quality/members/V-G11-A/critic.json). |
| `V-G01-A-E160` | After sneezing from a flour tin, the character reopens it and creates another cloud. Its critic passes it at 4/0/0, citing repeated exposure and mistaken suspicion. [Corpus](../../../datasets/village-001/member-corpora/V-G01-A.jsonl); [critic](../../../datasets/village-001/quality/members/V-G01-A/critic.json). |
| `V-G01-A-E050` | Rather than locating a washer beneath one of three saucers, the character keeps all three. The contrast explanation rejects the ordinary alternative for resolving the problem neatly. [Corpus](../../../datasets/village-001/member-corpora/V-G01-A.jsonl). |

This establishes tension with practical task completion, not that every positive target is unsuitable. For example, `V-G02-A-E101` gives a concrete recovery sequence after a coupling failure: stop, isolate, inspect and test. [Source corpus](../../../datasets/village-001/member-corpora/V-G02-A.jsonl).

## What original C validation establishes

All 36 original results contain 320 finite loss values and report no checkpoint resumes. Each corpus has 100 ancestral copies and 60 expansion examples; the 100 ancestral input/target pairs are identical across each lineage's three siblings. Across the population this is 5,760 rows, including 3,600 ancestral copies and 2,160 expansion rows.

The median of the 36 member-level mean online losses is **3.141 in epoch 1** and **2.423 in epoch 2**; every member's epoch mean declines. This shows optimization on the supplied targets, not held-out task quality. The report's first and last individual losses concern different shuffled examples and should not be interpreted as a paired validation measure.

Original C outputs already show repetition and weak grounding. `V-G05-B/T01` introduces and repeatedly positions a tin cup while answering a desk-fan problem. `V-G09-C/T12` invokes a stripped gear and scraping while responding to somebody moving an object. `V-G01-A/T01` repeatedly calls the fan or its noise a trap. [G05-B result](../sft-evidence/results/V-G05-B.json); [G09-C result](../sft-evidence/results/V-G09-C.json); [G01-A result](../sft-evidence/results/V-G01-A.json).

Of 864 common-suite responses, **118** reach the 160-token limit; of 288 member probes, **20** reach it. Limit saturation alone does not establish repetition. The [C assembler](../../../scripts/village_001c_assemble.py) verifies execution, identities, counts and custody but has no held-out task-quality threshold or contemporaneous BASE-generation stage. The [C report](../sft-evidence/REPORT.md) therefore establishes completed training, not usable task behavior.

## Hypotheses and interpretation limits

The narrow positive-only curriculum could encourage motif substitution or weak transfer. Optimization-induced loss of general capabilities, limited capacity and generation dynamics could also contribute. These are hypotheses: the archived loss traces and examples do not distinguish them causally, and no learning-rate, data or model-size ablation was performed here.

A null result in the separate F mask comparison would remain compatible with weaknesses inherited from C. It would not by itself identify which of these explanations is responsible.

## Proposed next test only

**This proposal is outside F's frozen workload.** No execution request or configuration has been added for it; its value will be assessed after the matched pilot.

Freeze members `V-G01-A` through `V-G12-A` and original training IDs `E001` and `E101` for each: 24 seen situations, covering one ancestral and one expansion example per lineage. Compare the existing BASE and C artifacts with the original neutral template and greedy 160-token settings: **48 generations, zero optimizer steps, and no F/Q cases**. Separately assess the authored target and generated responses for task completion, grounding and repetition.

- Faithful imitation of unsuitable authored targets would implicate the curriculum directly.
- Repetitive or unrelated C answers to suitable, already-seen targets would motivate a fitting or generation-dynamics investigation.
- Successful imitation on seen inputs alongside weak archived held-out behavior would support narrow transfer as the next hypothesis to test.

This first check would not distinguish model capacity from learning-rate or training-duration effects.
