# SCHEDUSWARM.md — Neurounner v2 dispatcher policy

```yaml
scheduswarm_version: 2
status: active
hub: https://github.com/techrote/scheduswarm
queue_branch: scheduswarm/queue
coordination_ledger: https://github.com/techrote/neurounner/issues/213
max_active_lanes: 3
project_concurrency_budget: 2
successors_per_lane: 1
integration_writes_parallel: false
automatic_dispatch_allowed: false
```

## Authority

Scheduswarm coordinates execution only. It does not supersede `RAG.md`, experiment protocols, source pins, evidence-retention rules, scientific gates, issue authority, or explicit owner decisions.

For NCR-001, every dispatched worker must freshly inspect `RAG.md`, issues #67–#81/#92 as applicable, `experiments/ncr-001/protocol.json`, `source-pins.json`, current NCR PRs/reviews/exact-head Actions/original evidence, and the owner decision on issue #72.

The four frozen lanes remain distinct: SmolLM2-135M Instruct, Sprocket-500M Chat, Sprocket-500M root Instruct, and dotlabs/void.1. Never transfer behavior, thresholds, evidence, weights or admission across them.

## Queue and completion

The dispatcher watches `scheduswarm/queue:.scheduswarm/tasks/*.md`.

Workers do **not** create scheduled-task successors. At completion they:
1. publish durable project evidence/handoff;
2. write `.scheduswarm/completions/<task_id>.json` on the queue branch;
3. write zero or one same-lane successor task envelope under `.scheduswarm/tasks/`;
4. create extra parallel tasks only if lane admission and capacity permit;
5. otherwise record an explicit stop reason.

Task prompts are checkpoints; every run starts with a fresh live audit.

## Limits and lane classes

Maximum active lanes: **3**. Project concurrency budget: **2**. Same-lane successor maximum: **1**.

| Lane | Purpose |
|---|---|
| scientific | model-specific qualification/adjudication |
| evidence | custody/replay/provenance/adversarial audit |
| accounting | #73 source/physical compute qualification |
| integration | serialized protocol/status reconciliation |

## Serialized surfaces

Parallel workers may not concurrently change `protocol.json`, `source-pins.json`, sealed/global fixture definitions/hashes, project-wide training/admission registries, or shared conformance state promoted across model lanes.

## NCR gates

- `train_authorized=false` and `full_72_conformance_pass=false` remain closed until explicit protocol scientific admission.
- Never start comparative NCA/MLP training, open sealed held-outs, alter preregistered suites, relax causal/cache/byte/fixture gates, or promote a lane from source-only/admin evidence.
- Receipt/archive CI is not a fresh physical-host run; physical inference is not automatically learned-quality or semantic competence.
- Preserve the issue #72 same-account COMMENTED exception exactly; do not call it GitHub APPROVED or a second human identity.
- Preserve negative results and complete raw evidence required by the experiment.

## Dispatch commissioning

`automatic_dispatch_allowed: false`. Use confirmation mode until the local Chrome adapter reliably verifies new-chat state, requested model/effort and exact prompt insertion. Automatic mode requires a future explicit policy change.

## Live ownership

Issue #213, branches/PRs and immutable queue records are the live ownership surfaces. Do not introduce a shared mutable current-state file.
