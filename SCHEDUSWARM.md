# SCHEDUSWARM.md — Neurounner task-lane policy

```yaml
scheduswarm_version: 1
status: active
hub: https://github.com/techrote/scheduswarm
coordination_ledger: https://github.com/techrote/neurounner/issues/213
max_active_lanes: 3
project_concurrency_budget: 2
successors_per_lane: 1
integration_writes_parallel: false
```

## Authority

Scheduswarm coordinates execution only. It does not supersede `RAG.md`, experiment-specific protocols, source pins, evidence-retention rules, scientific gates, issue authority, or explicit owner decisions.

For NCR-001, every lane must freshly inspect at minimum:

- `RAG.md`;
- issues #67–#81 and BYTE-EOS-001 #92 as applicable;
- `experiments/ncr-001/protocol.json` and `source-pins.json`;
- current NCR PRs/branches, reviews, exact-head Actions and retained original evidence;
- the owner review decision recorded on issue #72.

Four frozen model lanes remain scientifically distinct: SmolLM2-135M Instruct, Sprocket-500M Chat, Sprocket-500M root Instruct, and dotlabs/void.1. Do not transfer model behavior, thresholds, evidence or admission between them.

## Limits

**Maximum active lanes = 3.** Up to three logical successor chains may exist, including scheduled/waiting chains.

**Project concurrency budget = 2.** At most two Neurounner Scheduswarm lanes should intentionally execute or be launched into the same coordination window. A third active lane is staggered/waiting.

The active-lane limit controls coordination complexity. The concurrency budget controls simultaneous branch/CI/review race pressure.

## Lane classes

| Lane class | Purpose | Typical ownership |
|---|---|---|
| scientific | model-specific scientific qualification/adjudication | one model/evidence cohort and its bounded PR/issues |
| evidence | original evidence custody, replay, provenance, adversarial audit | distinct archive/auditor write set |
| accounting | #73 source/physical compute accounting and preflight | accounting scripts/tests/docs only unless separately authorized |
| integration | serialized cross-lane protocol/status reconciliation | shared NCR authority surfaces |

A normal lane may not create more than one successor for itself. A new parallel lane requires the Scheduswarm lane-admission test and free capacity.

## Serialized surfaces

While multiple NCR lanes are active, only the integration owner may intentionally change shared authoritative surfaces such as:

- `experiments/ncr-001/protocol.json`;
- `experiments/ncr-001/source-pins.json`;
- global/sealed fixture definitions or hashes;
- any project-wide training authorization/admission registry;
- shared conformance status promoted across model lanes.

Parallel lanes may propose changes in isolated branches but must not race integration.

## NCR safety gates

- `train_authorized=false` and `full_72_conformance_pass=false` remain closed until the repository's explicit scientific protocol admits otherwise.
- Never start comparative NCA/MLP training, open sealed held-outs, alter preregistered evaluation suites, relax causal/cache/byte/fixture gates, or promote a model lane from source-only/admin evidence.
- Green receipt/audit CI is not a fresh physical-host model run.
- Physical-host execution is not automatically learned-quality or semantic competence evidence.
- Same-account COMMENTED review is handled exactly as the current owner decision specifies; do not call it GitHub APPROVED or a second human identity.
- Preserve negative results and complete raw evidence required by the relevant experiment.

## Live ownership and handoff

Use issue #213 comments plus concrete branches/PRs as the append-only live lane registry. Each claim should state:

- lane ID/class;
- target issue/acceptance criterion;
- branch/PR and intended write set;
- dependencies/non-goals;
- successor task ID when created;
- stop/throttle reason when applicable.

Avoid a shared mutable “current lane state” file.

Every one-shot run must end with exact SHA/PR/run evidence, what physically executed versus what was source-only, retained failures, remaining gates, next highest-value task, and either one successor task ID or an explicit stop.
