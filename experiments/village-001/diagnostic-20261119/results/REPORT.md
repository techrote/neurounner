# VILLAGE-001F — matched diagnostics

## Worker attention-mask pilot

Primary corrected-minus-reproduced task relevance: **+0.0000/4**.
Prespecified working-effect criteria: **NOT MET**; no automatic rollout.
Positive member means: 3/6. Worst member relevance change: -0.2500/4.

Paired blind scoring of corrected and reproduced responses; six prespecified member means are replication units, not 72 independent models.

| Member | Reproduced relevance | Corrected relevance | Change | Original/reproduced exact text |
|---|---:|---:|---:|---:|
| V-G01-A | 1.417 | 1.750 | +0.333 | 11/12 |
| V-G03-B | 1.750 | 1.583 | -0.167 | 12/12 |
| V-G05-C | 1.417 | 1.500 | +0.083 | 12/12 |
| V-G07-A | 1.417 | 1.500 | +0.083 | 12/12 |
| V-G09-B | 2.167 | 1.917 | -0.250 | 12/12 |
| V-G11-C | 2.333 | 2.250 | -0.083 | 12/12 |

The attention mask is the only planned algorithmic intervention. C parent weights, pair bytes, original member seed, optimizer, objective, 120-step budget and required library versions are fixed. Each member's two fresh processes are assigned to one shared runner job, with three members per predeclared arm order. Actual optimizer provenance and reproduction differences are reported below.

## Context and reproducibility

BASE/C/original D were scored individually in different requests; contrasts with paired-arm ratings are contextual, not the primary causal estimate.

| Stage | Responses | Relevance | Coherence | Genericness | Groundedness |
|---|---:|---:|---:|---:|---:|
| BASE | 12 | 1.750 | 2.083 | 2.917 | 1.917 |
| C | 72 | 1.639 | 1.667 | 2.153 | 1.458 |
| D_ORIGINAL | 72 | 2.181 | 2.083 | 1.875 | 2.028 |

Original D versus reproduced D has exact output-text agreement on 71/72 Q cases. This is measured rather than assumed.
Canonical named-tensor fingerprints agree for 0/6 members; safetensors file hashes agree for 0/6. Directory hashes include auxiliary files and are reported separately.

### Paired optimizer execution

Immutable start, durable-checkpoint and finish receipts identify each invocation and the optimizer steps it retained. Shared-job and shared-runner checks use those positive-step segments, including prior invocations after a resume. A completed copy adds no optimizer host. Fresh uninterrupted pairs must use distinct processes in the predeclared sequential order; recovered pairs remain in the analysis with their limitations disclosed.

| Member | Execution classification | Predeclared first arm | Shared optimizer job | Shared optimizer runner | Retained steps | Copy invocations | Unfinished invocations |
|---|---|---|---|---|---:|---:|---:|
| V-G01-A | fresh_same_job_same_runner | historical_token_id_mask | match | match | 120 / 120 | 0 | 0 |
| V-G03-B | fresh_same_job_same_runner | length_derived_mask | match | match | 120 / 120 | 0 | 0 |
| V-G05-C | fresh_same_job_same_runner | historical_token_id_mask | match | match | 120 / 120 | 0 | 0 |
| V-G07-A | fresh_same_job_same_runner | length_derived_mask | match | match | 120 / 120 | 0 | 0 |
| V-G09-B | fresh_same_job_same_runner | historical_token_id_mask | match | match | 120 / 120 | 0 | 0 |
| V-G11-C | fresh_same_job_same_runner | length_derived_mask | match | match | 120 / 120 | 0 | 0 |

| Member / arm | Invocation evidence | Retained steps | Run / attempt / job | Paired-job nonce | Runner / boot ID / hostname | Process | Status |
|---|---|---:|---|---|---|---:|---|
| V-G01-A / D_REPRODUCED | [start 1](sources/training/V-G01-A/historical_token_id_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 3f8b087a0d1a47358beff639637d43e7 | GitHub Actions 1000132513 / 216f5bfe-ab25-4000-86f8-d27d7b2c44c2 / ip-10-0-128-105 | 2369 | completed |
| V-G01-A / D_CORRECTED | [start 1](sources/training/V-G01-A/length_derived_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 3f8b087a0d1a47358beff639637d43e7 | GitHub Actions 1000132513 / 216f5bfe-ab25-4000-86f8-d27d7b2c44c2 / ip-10-0-128-105 | 2446 | completed |
| V-G03-B / D_REPRODUCED | [start 1](sources/training/V-G03-B/historical_token_id_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | f55f0dad1d75453a88ffbd1f176c6f08 | GitHub Actions 1000132512 / d2b31e58-a428-463d-a731-094bc85357fd / ip-10-0-135-97 | 2442 | completed |
| V-G03-B / D_CORRECTED | [start 1](sources/training/V-G03-B/length_derived_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | f55f0dad1d75453a88ffbd1f176c6f08 | GitHub Actions 1000132512 / d2b31e58-a428-463d-a731-094bc85357fd / ip-10-0-135-97 | 2388 | completed |
| V-G05-C / D_REPRODUCED | [start 1](sources/training/V-G05-C/historical_token_id_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | affd6c6bc7f7429d8f8f5ae33be82f06 | GitHub Actions 1000132507 / 22e2c9c1-1cd2-4b77-82fe-6eff8da7c38c / ip-10-0-129-166 | 2382 | completed |
| V-G05-C / D_CORRECTED | [start 1](sources/training/V-G05-C/length_derived_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | affd6c6bc7f7429d8f8f5ae33be82f06 | GitHub Actions 1000132507 / 22e2c9c1-1cd2-4b77-82fe-6eff8da7c38c / ip-10-0-129-166 | 2444 | completed |
| V-G07-A / D_REPRODUCED | [start 1](sources/training/V-G07-A/historical_token_id_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 2a23413c018b46468d9a9cb6144b4100 | GitHub Actions 1000132511 / b5aa6240-fe97-436b-ab2c-f47ede3d898c / ip-10-0-138-80 | 2428 | completed |
| V-G07-A / D_CORRECTED | [start 1](sources/training/V-G07-A/length_derived_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 2a23413c018b46468d9a9cb6144b4100 | GitHub Actions 1000132511 / b5aa6240-fe97-436b-ab2c-f47ede3d898c / ip-10-0-138-80 | 2365 | completed |
| V-G09-B / D_REPRODUCED | [start 1](sources/training/V-G09-B/historical_token_id_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 1db59049e45946d38e5fc0513279e656 | GitHub Actions 1000132510 / 7d05822b-39a8-4ec1-b993-1f489e1b8f71 / ip-10-0-138-166 | 2345 | completed |
| V-G09-B / D_CORRECTED | [start 1](sources/training/V-G09-B/length_derived_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 1db59049e45946d38e5fc0513279e656 | GitHub Actions 1000132510 / 7d05822b-39a8-4ec1-b993-1f489e1b8f71 / ip-10-0-138-166 | 2418 | completed |
| V-G11-C / D_REPRODUCED | [start 1](sources/training/V-G11-C/historical_token_id_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 208c53977d5e4f1da066c03cd9d43daa | GitHub Actions 1000132509 / c8e26ba5-a9b2-4c7f-b723-53204ff459de / ip-10-0-142-35 | 2444 | completed |
| V-G11-C / D_CORRECTED | [start 1](sources/training/V-G11-C/length_derived_mask/training-invocations/invocation-0001-start.json) | 0–120 | 44467827716 / 1 / train | 208c53977d5e4f1da066c03cd9d43daa | GitHub Actions 1000132509 / c8e26ba5-a9b2-4c7f-b723-53204ff459de / ip-10-0-142-35 | 2390 | completed |

Full native provenance, exact source hashes and any unattributed interrupted segments are retained in summary.json and the linked source files. Same-runner execution does not guarantee numerical determinism. Original D has no equivalent invocation history, so historical comparisons remain more limited.

### Recorded training execution

Each stage links to its complete hash-bound training result. Exact result SHA-256s, full package metadata and field-level differences are also retained in summary.json. The recorded result host does not attest every earlier host of a resumed lane. Matching recorded metadata does not guarantee numerical equivalence.

| Member | Stage / result | Recorded CPU model | Logical CPUs | Platform | Packages | Resumed / step |
|---|---|---|---:|---|---|---|
| V-G01-A | [D_ORIGINAL](sources/training-references/original-D/V-G01-A.json) | AMD EPYC 7763 64-Core Processor | 4 | Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G01-A | [D_REPRODUCED](sources/training/V-G01-A/historical_token_id_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G01-A | [D_CORRECTED](sources/training/V-G01-A/length_derived_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G03-B | [D_ORIGINAL](sources/training-references/original-D/V-G03-B.json) | AMD EPYC 7763 64-Core Processor | 4 | Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G03-B | [D_REPRODUCED](sources/training/V-G03-B/historical_token_id_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G03-B | [D_CORRECTED](sources/training/V-G03-B/length_derived_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G05-C | [D_ORIGINAL](sources/training-references/original-D/V-G05-C.json) | AMD EPYC 7763 64-Core Processor | 4 | Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G05-C | [D_REPRODUCED](sources/training/V-G05-C/historical_token_id_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G05-C | [D_CORRECTED](sources/training/V-G05-C/length_derived_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G07-A | [D_ORIGINAL](sources/training-references/original-D/V-G07-A.json) | AMD EPYC 7763 64-Core Processor | 4 | Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G07-A | [D_REPRODUCED](sources/training/V-G07-A/historical_token_id_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G07-A | [D_CORRECTED](sources/training/V-G07-A/length_derived_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G09-B | [D_ORIGINAL](sources/training-references/original-D/V-G09-B.json) | AMD EPYC 7763 64-Core Processor | 4 | Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G09-B | [D_REPRODUCED](sources/training/V-G09-B/historical_token_id_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G09-B | [D_CORRECTED](sources/training/V-G09-B/length_derived_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G11-C | [D_ORIGINAL](sources/training-references/original-D/V-G11-C.json) | AMD EPYC 9V45 96-Core Processor | 4 | Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G11-C | [D_REPRODUCED](sources/training/V-G11-C/historical_token_id_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |
| V-G11-C | [D_CORRECTED](sources/training/V-G11-C/length_derived_mask/result.json) | Intel(R) Xeon(R) 6975P-C | 4 | Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 | peft=0.21.2; python=3.12.3; torch=2.14.1+cpu; transformers=4.57.6 | False / 0 |

| Member | Compared stages | CPU model | CPU count | Platform | Full package metadata | Recorded differences |
|---|---|---|---|---|---|---|
| V-G01-A | D_REPRODUCED / D_CORRECTED | match | match | match | match | none |
| V-G01-A | D_ORIGINAL / D_REPRODUCED | different | match | different | match | runner.cpu_model: AMD EPYC 7763 64-Core Processor -> Intel(R) Xeon(R) 6975P-C; runner.platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 -> Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |
| V-G03-B | D_REPRODUCED / D_CORRECTED | match | match | match | match | none |
| V-G03-B | D_ORIGINAL / D_REPRODUCED | different | match | different | match | runner.cpu_model: AMD EPYC 7763 64-Core Processor -> Intel(R) Xeon(R) 6975P-C; runner.platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 -> Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |
| V-G05-C | D_REPRODUCED / D_CORRECTED | match | match | match | match | none |
| V-G05-C | D_ORIGINAL / D_REPRODUCED | different | match | different | match | runner.cpu_model: AMD EPYC 7763 64-Core Processor -> Intel(R) Xeon(R) 6975P-C; runner.platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 -> Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |
| V-G07-A | D_REPRODUCED / D_CORRECTED | match | match | match | match | none |
| V-G07-A | D_ORIGINAL / D_REPRODUCED | different | match | different | match | runner.cpu_model: AMD EPYC 7763 64-Core Processor -> Intel(R) Xeon(R) 6975P-C; runner.platform: Linux-6.17.0-1022-azure-x86_64-with-glibc2.39 -> Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |
| V-G09-B | D_REPRODUCED / D_CORRECTED | match | match | match | match | none |
| V-G09-B | D_ORIGINAL / D_REPRODUCED | different | match | different | match | runner.cpu_model: AMD EPYC 7763 64-Core Processor -> Intel(R) Xeon(R) 6975P-C; runner.platform: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 -> Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |
| V-G11-C | D_REPRODUCED / D_CORRECTED | match | match | match | match | none |
| V-G11-C | D_ORIGINAL / D_REPRODUCED | different | match | different | match | runner.cpu_model: AMD EPYC 9V45 96-Core Processor -> Intel(R) Xeon(R) 6975P-C; runner.platform: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39 -> Linux-7.0.0-1014-aws-x86_64-with-glibc2.39 |

### Attribution qualifications

- **V-G01-A:** Original and reproduced D training receipts differ in runner.cpu_model, runner.platform; this metadata difference alone does not establish a weight or behavior effect or explain a reproduction failure.
- **V-G01-A:** The original D tensor fingerprint was not reproduced. The paired effect applies to these new executions and does not by itself isolate a repair of the historical D run.
- **V-G01-A:** Original and reproduced D agree on 11/12 exact Q responses; historical behavior was not fully reproduced.
- **V-G03-B:** Original and reproduced D training receipts differ in runner.cpu_model, runner.platform; this metadata difference alone does not establish a weight or behavior effect or explain a reproduction failure.
- **V-G03-B:** The original D tensor fingerprint was not reproduced. The paired effect applies to these new executions and does not by itself isolate a repair of the historical D run.
- **V-G05-C:** Original and reproduced D training receipts differ in runner.cpu_model, runner.platform; this metadata difference alone does not establish a weight or behavior effect or explain a reproduction failure.
- **V-G05-C:** The original D tensor fingerprint was not reproduced. The paired effect applies to these new executions and does not by itself isolate a repair of the historical D run.
- **V-G07-A:** Original and reproduced D training receipts differ in runner.cpu_model, runner.platform; this metadata difference alone does not establish a weight or behavior effect or explain a reproduction failure.
- **V-G07-A:** The original D tensor fingerprint was not reproduced. The paired effect applies to these new executions and does not by itself isolate a repair of the historical D run.
- **V-G09-B:** Original and reproduced D training receipts differ in runner.cpu_model, runner.platform; this metadata difference alone does not establish a weight or behavior effect or explain a reproduction failure.
- **V-G09-B:** The original D tensor fingerprint was not reproduced. The paired effect applies to these new executions and does not by itself isolate a repair of the historical D run.
- **V-G11-C:** Original and reproduced D training receipts differ in runner.cpu_model, runner.platform; this metadata difference alone does not establish a weight or behavior effect or explain a reproduction failure.
- **V-G11-C:** The original D tensor fingerprint was not reproduced. The paired effect applies to these new executions and does not by itself isolate a repair of the historical D run.

These disclosures do not exclude members, change the prespecified working-effect criteria or authorize additional training.

## Sprocket feedback pilot

| Arm | Strict JSON/schema | Grounded | Strict, grounded and actionable | Response grounding | Scenario grounding | Usefulness |
|---|---:|---:|---:|---:|---:|---:|
| original | 0/18 | 9/18 | 0/18 | 1.222/2 | 1.722/2 | 0.833/2 |
| compact_schema_first | 0/18 | 8/18 | 0/18 | 1.000/2 | 1.500/2 | 0.667/2 |

Syntax, grounding and usefulness are separate measurements. This small matched prompt pilot does not establish a causal benefit to worker learning.

## Limits and custody

- Six prespecified members and twelve new prompts do not establish full-population generalization.
- Each member's two fresh training processes are assigned to one shared runner job in predeclared counterbalanced order. Native invocation receipts establish whether that pairing survived execution or recovery; a shared runner does not guarantee numerical determinism.
- Original D lacks the new per-invocation execution provenance. Historical CPU, platform and package comparisons cannot establish that its full optimizer trajectory shared the new execution environment.
- Execution and reproduction differences qualify attribution without changing the prespecified panel, quality criteria or workload.
- Remote critic backend is not version-pinned; scores are evidence from this archived request cohort.
- Repeated prompts and Sprocket seeds are within-case observations, not independent replications.
- No Stage E outcome selected the panel, Q prompts, arms or thresholds.
- Original C/D evidence and final E qualification are preserved; this diagnostic makes no automatic rollout decision.

All 300 worker responses, 36 Sprocket outputs, matched training receipts, blind inputs, private mapping, raw judge requests/responses/repairs, per-case judgments and hashes accompany this report. No weights are copied into this evidence tree.
