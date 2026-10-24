# PERSONA-002 — anti-sanitization preference continuation

PERSONA-002 continues the four PERSONA-001 SmolLM2-135M LoRA cartridges using BABBLE-002's paired preferred target vs generic/sanitized rejected target.
The exact PERSONA-001 24-scenario blind tournament is reused unchanged.

## Headline comparison

| Metric | PERSONA-001 | PERSONA-002 | Delta |
|---|---:|---:|---:|
| Raw blind identification | 46.9% | 71.9% | +25.0 pp |
| Behavior-only identification | 41.7% | 51.0% | +9.4 pp |
| Raw→behavior style gap | 5.2% | 20.8% | +15.6 pp |

## Preference training

| Persona | CPU | steps | pair margin before | pair margin after | preference acc before | after | seq tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| G02 | AMD EPYC 9V45 96-Core Processor | 200 | 1.610 | 4.945 | 100.0% | 100.0% | 168.7 |
| G03 | AMD EPYC 9V45 96-Core Processor | 200 | 1.428 | 4.493 | 100.0% | 100.0% | 162.0 |
| G05 | AMD EPYC 7763 64-Core Processor | 200 | 1.030 | 4.073 | 98.0% | 100.0% | 100.8 |
| G12 | INTEL(R) XEON(R) PLATINUM 8573C | 200 | 1.608 | 4.745 | 100.0% | 100.0% | 173.0 |

## Confusion / transfer

G03↔G12 raw cross-confusions: **10 → 7**.
G03↔G12 behavior-only cross-confusions: **18 → 5**.
Abstract-unfamiliar raw accuracy: **0.0% → 62.5%**.
Abstract-unfamiliar behavior-only accuracy: **25.0% → 81.2%**.

## Per-persona change

| Persona | Raw fit Δ | Raw genericness Δ | Raw relevance Δ | Behavior fit Δ | Behavior genericness Δ | Behavior relevance Δ |
|---|---:|---:|---:|---:|---:|---:|
| G02 | +0.42 | -0.25 | +0.50 | +0.38 | -0.33 | -0.08 |
| G03 | +0.25 | -0.25 | +0.38 | +0.00 | -0.21 | +0.00 |
| G05 | +0.17 | -0.04 | +0.54 | +0.08 | -0.25 | +0.00 |
| G12 | +0.46 | -0.38 | +0.58 | +0.04 | -0.25 | +0.08 |

## Verdict

**PASS.**

The primary success criterion is improvement in style-neutral behavioral identification without a major task-relevance regression. Negative results are retained as evidence rather than optimized away.

Machine-readable evidence: experiments/persona-002/evidence/summary.json.
