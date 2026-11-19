# VILLAGE-001E — final village qualification

The final preference-matured cartridges were evaluated on a fresh F01-F24 suite generated only after preference training, without prompt-time member/persona descriptions. Tournament providers received opaque response IDs; ecology models/judges received opaque participant handles.

## Population gate

- usable village: **FAIL**
- behavior-only lineage accuracy: 0.257 (gate 0.167)
- behavior-only sibling accuracy with correct lineage supplied: 0.359 (gate 0.417)
- exact behavior-only 36-member accuracy: 0.100
- mean final raw-tournament genericness: 1.752 (same F-suite untuned BASE 2.125)
- mean style-neutral genericness: 1.814 (same F-suite untuned BASE 1.917)
- mean final raw task relevance: 3.192/4 (BASE 3.083)
- mean final style-neutral task relevance: 2.834/4 (BASE 3.250)
- individual quality checks: 14/36 (gate 30)

## Raw versus behavior-only

- raw lineage accuracy: 0.302
- behavior-only lineage accuracy: 0.257
- lineage style gap: +0.045
- raw sibling accuracy given correct lineage: 0.361
- behavior-only sibling accuracy given correct lineage: 0.359

## Ecology

- sessions: 24 (12 same-lineage pairs, 12 cross-lineage small groups)
- differentiation: mean 1.333/4
- generic_convergence: mean 0.833/4
- information_sharing: mean 0.875/4
- information_hoarding: mean 0.583/4
- conflict_persistence: mean 0.708/4
- conflict_repair: mean 0.833/4
- cross_agent_contamination: mean 1.000/4
- outcome_progress: mean 0.667/4
- cross-lineage contamination: mean 0.083/4
- cross-lineage expected-lineage fit: mean 1.194/4
- cross-lineage distinction survival: mean 0.972/4
- participant dominance: mean 1.217/4
- participant deference: mean 0.983/4
- Super convergence concerns: 10/24 sessions

Ecology is diagnostic evidence, not a retroactively invented hard gate. The population gate is exactly the predeclared lineage/sibling/genericness/individual-quality gate above.

## Transfer families

- abstract_unfamiliar behavior lineage accuracy: 0.181
- abstract_unfamiliar behavior sibling accuracy: 0.347
- failure_recovery behavior lineage accuracy: 0.292
- failure_recovery behavior sibling accuracy: 0.368

## Member summary

| Member | Behavior lineage | Task relevance | Genericness | Maturation | Individual gate |
|---|---:|---:|---:|---|---|
| V-G01-A | 4/24 | 3.333 | 1.750 | PASS | FAIL |
| V-G01-B | 10/24 | 3.375 | 1.625 | PASS | FAIL |
| V-G01-C | 8/24 | 3.125 | 1.750 | PASS | PASS |
| V-G02-A | 3/24 | 3.458 | 1.625 | PASS | PASS |
| V-G02-B | 4/24 | 3.458 | 1.500 | PASS | FAIL |
| V-G02-C | 0/24 | 3.333 | 1.833 | PASS | FAIL |
| V-G03-A | 10/24 | 3.292 | 1.708 | PASS | PASS |
| V-G03-B | 16/24 | 2.833 | 1.500 | PASS | PASS |
| V-G03-C | 4/24 | 3.292 | 1.917 | PASS | FAIL |
| V-G04-A | 8/24 | 3.167 | 1.792 | PASS | PASS |
| V-G04-B | 4/24 | 3.083 | 1.708 | PASS | PASS |
| V-G04-C | 5/24 | 3.250 | 1.667 | PASS | FAIL |
| V-G05-A | 16/24 | 3.083 | 1.667 | PASS | FAIL |
| V-G05-B | 1/24 | 3.167 | 1.667 | PASS | FAIL |
| V-G05-C | 6/24 | 3.083 | 1.792 | PASS | PASS |
| V-G06-A | 5/24 | 3.333 | 1.875 | PASS | PASS |
| V-G06-B | 1/24 | 3.375 | 1.875 | PASS | FAIL |
| V-G06-C | 4/24 | 3.208 | 1.708 | PASS | FAIL |
| V-G07-A | 15/24 | 3.000 | 1.792 | PASS | PASS |
| V-G07-B | 10/24 | 2.958 | 1.917 | PASS | FAIL |
| V-G07-C | 15/24 | 3.167 | 1.833 | PASS | FAIL |
| V-G08-A | 2/24 | 3.250 | 1.750 | PASS | FAIL |
| V-G08-B | 7/24 | 3.375 | 1.833 | PASS | PASS |
| V-G08-C | 3/24 | 3.208 | 1.875 | PASS | FAIL |
| V-G09-A | 12/24 | 3.167 | 1.542 | PASS | PASS |
| V-G09-B | 2/24 | 3.208 | 1.833 | PASS | FAIL |
| V-G09-C | 1/24 | 3.417 | 1.792 | PASS | FAIL |
| V-G10-A | 0/24 | 3.292 | 1.625 | PASS | FAIL |
| V-G10-B | 2/24 | 2.583 | 1.958 | PASS | FAIL |
| V-G10-C | 0/24 | 2.917 | 1.958 | PASS | FAIL |
| V-G11-A | 4/24 | 3.375 | 1.667 | PASS | PASS |
| V-G11-B | 21/24 | 3.292 | 1.583 | PASS | PASS |
| V-G11-C | 10/24 | 3.042 | 1.958 | PASS | FAIL |
| V-G12-A | 4/24 | 2.792 | 1.500 | PASS | PASS |
| V-G12-B | 1/24 | 3.167 | 1.792 | PASS | FAIL |
| V-G12-C | 4/24 | 3.458 | 1.917 | PASS | FAIL |

Full confusion matrices, per-scenario raw provider evidence, ecology transcripts/judgments and hashes are retained beside this report.

