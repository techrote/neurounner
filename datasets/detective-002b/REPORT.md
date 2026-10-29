# DETECTIVE-002B — seven-body no-training runtime/dialogue smoke

This report is a runtime/behavior characterization, not an NPC ranking or training admission. DETECTIVE-001R1 #121 remains independently reviewer-blocked.

| Body | Runtime | Params | Load s | RSS GiB | Peak GiB | Mean completion tok/s | Trunc. | Repeats | Memory uncertainty | Corrected record | Rumor qualifier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| smollm2-135m-instruct | runtime_pass | 134515008 | 0.29 | 0.95 | 1.20 | 17.78 | 3 | 0 | no | no | yes |
| smollm2-360m-instruct | runtime_pass | 361821120 | 0.33 | 1.72 | 2.39 | 9.48 | 0 | 0 | no | yes | no |
| sprocket-500m-chat | runtime_pass | 501090560 | 0.38 | 2.31 | 3.24 | 6.80 | 4 | 1 | no | yes | yes |
| sprocket-500m-instruct | runtime_pass | 501090560 | 0.33 | 2.31 | 3.25 | 7.55 | 3 | 0 | no | yes | yes |
| qwen25-0.5b-instruct | runtime_pass | 494032768 | 0.48 | 2.35 | 3.27 | 7.66 | 0 | 0 | no | yes | no |
| qwen3-0.6b | runtime_pass | 751632384 | 0.51 | 2.77 | 4.74 | 6.57 | 0 | 0 | yes | no | no |
| lfm25-350m | runtime_fail | 354483968 | — | — | 0.53 | — | 0 | 0 | no | no | no |

Mechanical markers above are not human quality judgments. A response can mention the corrected timestamp while still being incoherent; raw outputs are the authority.

## Boundaries

- optimizer steps: **0**;
- paid API calls: **0**;
- relevance mapping accessed: **false**;
- population expansion admitted: **false**;
- body training selection made: **false**.

LFM2.5-350M remains research-runtime eligible but production/game distribution is still license-gated under lfm1.0. Qwen3-0.6B remains an explicit near-band exception at 751,632,384 parameters. Sprocket Chat and Instruct remain distinct checkpoint hashes.

Any later body-selection decision must separately reconcile runtime evidence with DETECTIVE-001R1 relevance and C1/C2/C3 policy evidence.
