# PERSONA-001 — four 135M personality cartridges and blind temperament tournament

Raw blind identification accuracy: **46.9%** (45/96).
Style-neutral behavioural identification accuracy: **41.7%** (40/96).
Raw-minus-behaviour accuracy gap: **5.2 percentage points**.
Mean pairwise lexical divergence across cartridge responses: **81.4%**.

## Training

| Persona | CPU | steps | seq tok/s | loss | peak RSS MiB | adapter MiB |
|---|---|---:|---:|---:|---:|---:|
| G02 | AMD EPYC 7763 64-Core Processor | 200 | 114.2 | 3.970→1.959 | 1671 | 9.37 |
| G03 | AMD EPYC 9V74 80-Core Processor | 200 | 130.0 | 3.251→1.687 | 1628 | 9.37 |
| G05 | AMD EPYC 7763 64-Core Processor | 200 | 106.0 | 4.412→1.976 | 1636 | 9.37 |
| G12 | AMD EPYC 9V45 96-Core Processor | 200 | 179.4 | 3.107→1.764 | 1702 | 9.37 |

## Blind identification by family

| Family | Raw | Behaviour-only |
|---|---:|---:|
| abstract_unfamiliar | 0.0% | 25.0% |
| conflicting_motives | 31.2% | 56.2% |
| embodied_decision | 56.2% | 37.5% |
| failure_recovery | 75.0% | 31.2% |
| mundane | 87.5% | 50.0% |
| social | 31.2% | 50.0% |

## Genericness / persona persistence

| Persona | Raw persona fit /4 | Raw genericness /4 | Behaviour persona fit /4 | Behaviour genericness /4 |
|---|---:|---:|---:|---:|
| G02 | 3.38 | 0.83 | 3.25 | 1.46 |
| G03 | 3.62 | 0.88 | 3.58 | 1.25 |
| G05 | 3.67 | 0.46 | 3.58 | 1.29 |
| G12 | 3.38 | 0.88 | 3.71 | 1.25 |
| BASE control | — | 3.83 | — | 3.71 |

## Interpretation

- Raw matching uses the original generated prose.
- Behaviour-only matching first strips wording/cadence/metaphor and keeps choices, priorities, hesitation, persistence, risk appetite, territoriality, ritual and recovery behavior.
- A large raw/behaviour accuracy gap indicates lexical/style leakage; strong behaviour-only accuracy indicates learned temperament beyond voice.
- The base model is never one of the four assignment classes; it is scored separately as a genericness/task-relevance control.
- All 24 prompts are held out from BABBLE-002 and no persona instruction is supplied at inference time.

Machine-readable evidence: experiments/persona-001/evidence/summary.json.
