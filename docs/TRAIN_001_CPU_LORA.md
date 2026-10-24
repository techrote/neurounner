# TRAIN-001 — SmolLM2-135M CPU LoRA checkpoint/resume proof

Date: 24 October 2026  
Repository: `techrote/neurounner`  
Actions run: `40110636171`  
Source head: `4a9d183fb5aa85df94e90a77393641edb29083f1`

## Mission

Test whether a genuinely useful sub-1B specialization workflow can run entirely on standard GitHub-hosted CPU runners and continue across separate jobs.

The task was deliberately narrow:

```text
messy behaviour idea
    ->
SmolLM2-135M-Instruct + LoRA
    ->
canonical JSON KernelSpec
```

The required output keys were exactly:

- `trigger`
- `state`
- `update`
- `action`
- `failure`

Training loss was masked to the assistant/output portion. The dataset contains 24 synthetic training examples and 6 held-out examples.

## Model and software

Base model: `HuggingFaceTB/SmolLM2-135M-Instruct`  
Resolved Hugging Face revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`

Observed package stack:

- PyTorch: `2.14.1+cpu`
- Transformers: `4.57.6`
- PEFT: `0.21.2`

LoRA:

- rank: 8
- alpha: 16
- dropout: 0.05
- trainable parameters: **2,442,240**
- trainable fraction: **1.783%** of the model

## Stage 1

Fresh GitHub-hosted `ubuntu-24.04` runner.

CPU: **AMD EPYC 7763 64-Core Processor**  
Logical CPUs exposed: 4.

24 optimizer steps:

- training tokens: **3,683**
- training wall time: **31.43 s**
- throughput: **117.18 training tokens/s**
- first loss: **4.7135**
- last loss: **3.5182**
- mean loss: **3.6504**
- peak RSS: **1,748,660 KiB (~1.67 GiB)**
- complete resumable checkpoint: **29,721,169 bytes (~28.3 MiB)**

Held-out behaviour before stage 1:

- valid JSON: **3/6**
- exact five-key schema: **0/6**
- mean expected keys present: **0.83 / 5**

After 24 optimizer steps:

- valid JSON: **5/6**
- exact five-key schema: **5/6**
- mean expected keys present: **4.17 / 5**

The untrained/instruct baseline often emitted explanations, fenced JSON, malformed nested schemas or even Python code. After 24 steps it overwhelmingly emitted compact one-line JSON with the requested key set.

## Cross-run checkpoint

Stage 1 saved:

- PEFT adapter weights;
- optimizer state;
- Python RNG state;
- PyTorch RNG state;
- global optimizer step;
- dataset cursor;
- exact base-model revision;
- package and source provenance.

The checkpoint artifact was uploaded by stage 1, downloaded by a separate stage-2 GitHub Actions job, inspected, and loaded successfully.

This proves that the training process can be split across independent hosted-runner jobs rather than relying on one long-lived machine.

## Stage 2 — fresh-runner resume

Separate GitHub-hosted job, again exposing:

CPU: **AMD EPYC 7763 64-Core Processor**  
Logical CPUs: 4.

It resumed from global step 24 and trained for another 16 optimizer steps.

Stage-2 measurements:

- global step after resume: **40**
- training tokens this stage: **2,463**
- training wall time: **21.11 s**
- throughput: **116.66 training tokens/s**
- first stage-2 loss: **1.8453**
- last stage-2 loss: **2.3604**
- mean stage-2 loss: **2.2758**
- peak RSS: **1,765,168 KiB (~1.68 GiB)**
- checkpoint size: **29,721,079 bytes**

Held-out exact-schema rate at resume: **5/6**  
After another 16 steps: **6/6**

The near-identical 117.18 vs 116.66 tok/s across the two jobs is a useful sanity check that the resumed training workload behaved consistently.

## What the model actually learned

The behavioural change is obvious, but mostly at the level of **representation and vocabulary**.

Before fine-tuning it frequently answered with conversational explanations, code fences, nested schemas and unrelated Python.

After 24–40 steps it reliably produced compact five-field JSON containing concepts such as:

- bounded scalar state;
- unit-vector directional memory;
- decay;
- local triggers;
- failure modes.

That is substantial adaptation from only a few thousand training tokens.

However, semantic quality is **not yet good enough for an authoritative compiler**. Examples after 40 steps still include errors such as:

- converting “refuse another jump while shy” into an action like “steer away from shyness”;
- inventing awkward variables such as `slip_side_side`;
- using an update/action that merely says to store state instead of implementing the requested behaviour;
- producing failure modes that restate the trigger rather than predict an emergent failure;
- occasional contradictions between trigger, state and action.

TRAIN-001 therefore proves fast specialization and resumability, **not useful kernel quality yet**.

## Five-hour scaling implication

Ignoring evaluation/setup/checkpoint overhead, the observed ~117 training tokens/s corresponds to roughly **2.1 million sequence tokens in five hours**.

Observed optimizer-step cadence was about 0.76 steps/s with these short examples, or roughly **13,500–13,800 optimizer steps in five hours** at the same sequence distribution.

Those are capacity estimates, not a recommendation to train that long on the current 24-example dataset. The tiny dataset would overfit long before that point.

## Conclusion

**PASS — strongly.**

For a 135M model, GitHub-hosted CPU LoRA training is not merely plausible; the actual model-update portion is fast.

The limiting problems are now:

1. dataset quality and coverage;
2. semantic evaluation rather than JSON-format compliance;
3. avoiding overfitting;
4. deciding whether LoRA is sufficient or whether full-parameter tuning is worth testing;
5. persistent checkpoint storage for runs longer than artifact retention.

## Recommended follow-up

### TRAIN-002A — better 135M compiler dataset

Expand from 24 examples to several hundred or a few thousand generated/reviewed pairs with explicit negative cases:

- preserve a strange mechanism;
- repair only invalid mechanics;
- distinguish state from action;
- retain local-only constraints;
- produce genuinely causal failure modes;
- tag `INTERESTING_BROKEN` rather than sanitizing novelty.

Add semantic validators and judge held-out outputs on mechanical correctness, not just schema.

### TRAIN-002B — void.1 comparison

Run the same dataset and budget on `dotlabs/void.1` (~90M) to test whether its symbolic pretraining produces better mechanics per CPU second despite starting as a base model.

### TRAIN-002C — 135M full-parameter feasibility

The LoRA run peaked below 1.7 GiB RSS, leaving substantial room inside the standard ~16 GiB runner. A short full-parameter AdamW spike is therefore worth testing separately.

That would directly answer how much throughput is lost when updating all 135M parameters rather than only the 2.44M LoRA parameters.
