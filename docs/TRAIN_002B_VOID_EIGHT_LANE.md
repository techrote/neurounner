# TRAIN-002b — eight-lane byte-aware void.1 evaluation

Issue: #1

Eight independent GitHub-hosted runners trained void.1 in parallel. Two controls differed only by seed; six lanes changed one logical factor relative to control A.

The evaluation budget was increased from TRAIN-002's 120 byte-oriented tokenizer units to 768, and throughput is reported in UTF-8 bytes/s and characters/s as well as tokenizer units/s.

| Lane | Factor | CPU | train B/s | loss | exact schema | budget exhausted | deterministic proxy | cloud mechanical /4 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ctrl-a | control_seed | AMD EPYC 7763 64-Core Processor | 162.8 | 0.986→0.267 | 100% | 100% | 0.342 | 1.83 |
| ctrl-b | control_seed | AMD EPYC 9V45 96-Core Processor | 254.2 | 1.130→0.273 | 100% | 100% | 0.351 | 2.00 |
| lr-low | learning_rate | AMD EPYC 7763 64-Core Processor | 161.5 | 0.986→0.344 | 83% | 100% | 0.257 | 1.67 |
| lr-high | learning_rate | AMD EPYC 9V45 96-Core Processor | 253.3 | 0.986→0.257 | 83% | 100% | 0.301 | 1.50 |
| rank4 | lora_capacity | AMD EPYC 9V74 80-Core Processor | 199.7 | 0.986→0.324 | 100% | 100% | 0.363 | 1.17 |
| rank16 | lora_capacity | AMD EPYC 9V74 80-Core Processor | 203.5 | 0.986→0.308 | 100% | 100% | 0.362 | 1.50 |
| dropout0 | lora_dropout | AMD EPYC 9V74 80-Core Processor | 247.8 | 0.986→0.272 | 100% | 100% | 0.334 | 1.33 |
| failure-weighted | training_material | AMD EPYC 7763 64-Core Processor | 163.4 | 1.011→0.359 | 100% | 100% | 0.367 | 2.00 |

## Ranking

Cloud mechanical-fidelity ranking: failure-weighted > ctrl-b > ctrl-a > lr-low > rank16 > lr-high > dropout0 > rank4.

Control mean exact-schema rate: 100%.
Control mean cloud mechanical score: 1.92/4.

## Interpretation rules

- Treat control A/B spread as the seed/noise floor before attributing small differences to a perturbation.
- Byte-tokenizer units are implementation-specific; bytes/s and characters/s are the cross-run throughput measures for void.1.
- Exact JSON schema is necessary for compiler use but does not imply mechanical correctness; deterministic similarity and an independent MiniMax mechanical judge are recorded separately.
- All held-out generations, including malformed/truncated outputs, are preserved under experiments/train-002b/evidence/lanes/.
- NVIDIA judge request/response history is preserved under experiments/train-002b/evidence/semantic-judge/.

Full machine-readable comparison: experiments/train-002b/evidence/summary.json.
