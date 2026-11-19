# Stage F worker inputs: independent numerical findings

These findings describe the completed, published inputs before blinded judging. They establish input integrity, observed mask effects, execution pairing and numerical reproduction. They do not establish an improvement in task quality.

Input publication: [`783a73ca7c8e97c3e03ef4e56fb5a5b125d58785`](https://github.com/techrote/neurounner/tree/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119), tag `village-001f-pilot-inputs-20261119`; generation run [`44467827716`](https://github.com/techrote/neurounner/actions/runs/44467827716), source `eaeddb22142339c0cd838e588bc09daaa980f652`. Scientific contracts remain bound to `78a72238a8403c4cddb9e8651242a89e17ee21be`.

## Completeness and execution control

Independent reconstruction passed for all 12 training lanes, 1,440 retained optimizer steps, 25 response lanes, 300 Q responses, 720 mask-audit sequences and 288 fixed-forward batches. Every lane has one fresh, uninterrupted 0→120 optimizer segment. There are no resumed invocations, completed-copy invocations or unattributed retained steps. All six member pairs have matching job nonce, boot ID, hostname and full recorded environment; separate process IDs and nonoverlapping timestamps confirm the predeclared historical-first/corrected-first order (three members each). Recorded native training-invocation durations range from 133.56 to 175.04 seconds.

All twelve new training arms recorded Intel Xeon 6975P-C CPUs, four available CPUs and the same Linux platform and package versions. Historical D recorded AMD EPYC 7763 for five selected members and AMD EPYC 9V45 for G11-C, with different Linux platform strings. Recorded Python, torch, transformers and peft versions match. The original-versus-reproduced parameter differences cannot be attributed to a particular runtime cause from these receipts alone.

## Exact reproduction and numerical parameter distance

Every adapter has 420 F32 tensors containing 2,442,240 scalar parameters. All 420 tensor hashes differ in both comparisons for each member. Exact weight reproduction therefore remains **false** throughout. Numerical distances are descriptive and introduce no replacement threshold. The full report preserves each archive SHA-256, safetensors SHA-256 and canonical fingerprint, and all 5,040 per-tensor comparison records.

F32 values were converted exactly to float64 before subtraction. Overall L2 is the square root of the sum of squared scalar differences across all tensors. Relative L2 divides by the left/reference adapter’s full L2 norm. These are distances between LoRA factor parameters, not a metric of full-model behavior or task quality.

### Original D → historical-mask reproduction

| Member | Maximum absolute difference | Overall L2 | Relative L2 | Exact Q text/token matches |
|---|---:|---:|---:|---:|
| V-G01-A | 0.00032047741 | 0.00088926276 | 2.5025236e-05 | 11/12 |
| V-G03-B | 0.00028658392 | 0.00076777404 | 2.1161916e-05 | 12/12 |
| V-G05-C | 0.00031709066 | 0.00087257906 | 2.4010676e-05 | 12/12 |
| V-G07-A | 0.00020658597 | 0.00073917917 | 2.0577491e-05 | 12/12 |
| V-G09-B | 0.00018412992 | 0.00060174348 | 1.6624689e-05 | 12/12 |
| V-G11-C | 5.5905432e-05 | 0.00011136369 | 3.1137395e-06 | 12/12 |

Original D and reproduction match exactly on 71 of 72 Q texts and generated-token arrays, despite the tensor differences. G01-A supplies the single nonidentical response.

### Historical-mask reproduction → corrected mask

| Member | Maximum absolute difference | Overall L2 | Relative L2 | Exact Q text/token matches |
|---|---:|---:|---:|---:|
| V-G01-A | 0.013210555 | 1.240635 | 0.034913393 | 0/12 |
| V-G03-B | 0.015039701 | 1.0200159 | 0.028114379 | 2/12 |
| V-G05-C | 0.017543473 | 1.0165503 | 0.027972314 | 2/12 |
| V-G07-A | 0.016316726 | 1.5224925 | 0.042383594 | 0/12 |
| V-G09-B | 0.025316679 | 1.6504297 | 0.045597297 | 0/12 |
| V-G11-C | 0.013480594 | 1.423315 | 0.039796026 | 1/12 |

The paired mask intervention changes 67 of 72 Q texts/token arrays. Its parameter distances are larger than the original-versus-reproduced differences. Neither observation establishes that the changed outputs are better.

## Native mask and fixed-weight evidence

The published token audit is byte-identical to the preliminary artifact: SHA-256 `3566f8a5279ba5acf367f019fbdb23c32cbad611a5942e8b159d655f1d8c82f5`. All 720 sequences have two masked prompt boundaries and one masked assistant boundary: 2,160 input positions total. Native sequence lengths are 73–230, with zero prompt-prefix/label mismatches.

The assistant boundary is penultimate in every sequence, followed by supervised newline token 198. Shifted loss predicts the boundary target from the preceding position and predicts the newline at the boundary position. Its attention can therefore affect the final newline prediction; the two prompt boundaries can affect the assistant context more broadly. No target label is deleted, and the 2,160 positions are not equally influential loss contributions. This derivation does not isolate each boundary’s causal effect.

All 144 fixed-weight pair rows change both chosen and rejected mean log probabilities under the corrected attention mask. All 144 preference margins change numerically; none changes sign. The 18 lanes comprise six members × BASE/C/original-D, with eight fixed T/P pairs and two masks per lane (288 forward batches). These are training diagnostics, not held-out evaluations.

| Fixed weights | Pair rows | Mean chosen log-probability change | Mean rejected change | Mean margin change | Margin sign flips |
|---|---:|---:|---:|---:|---:|
| BASE | 48 | -0.02382784 | -0.00857520 | -0.01525264 | 0 |
| C | 48 | +0.02514697 | +0.02926732 | -0.00412035 | 0 |
| D_ORIGINAL | 48 | +0.00684608 | +0.02638309 | -0.01953701 | 0 |

Changes are length-derived minus historical-mask mean log probabilities (natural-log units). The comparisons hold weights fixed and change the complete attention mask; they do not identify a separate contribution for each boundary.

## Response completeness and interpretation limits

All 300 Q responses are present and nonempty. The native Q prompt text, rendered wire, input IDs and all-ones attention mask match across all 25 cohorts for each case. A 160-token limit without terminal EOS occurs in 11/12 BASE responses and 16/72 C responses; it occurs in none of the 216 original-D/reproduced-D/corrected-D responses. These truncation counts limit interpretation of contextual comparisons.

The six members remain the replication units. No confidence intervals, significance claims, new quality gates or rollout decision are added. Blinded judgments will determine the frozen relevance criteria. Original D stays a reference; paired reproduced-versus-corrected D is the primary comparison. All raw evidence remains unchanged.

## Reproduction and custody

- [worker-input-analysis.json](worker-input-analysis.json): detailed independent native reconstruction, SHA-256 `d61a6349996288aef85875c6420385851db72f994b30fe3219873e1e2f6863a2`.
- [worker-input-findings.json](worker-input-findings.json): compact execution/source and numerical-forward findings, SHA-256 `cc954cb4bc658be5602b96930e86d4b099f7db3984137cdfce700ff73c1eb3f0`.
- [adapter-distances.json](adapter-distances.json): exact verified archive identities and complete per-tensor distances, SHA-256 `79821f755d4f55f686ba81a3384b0723ce9522f3b1f13efa1da0ec1fd72b2cc2`.
- Separate full input acceptance: [input-acceptance.json](../input-acceptance.json), SHA-256 `c6f882d2755f293cfd236452619ff039f161843e0b99aabdf44ab5958fbeefc3`; it independently verifies all 24 published/historical adapters and release/Git custody.

The independent reconstruction is reproduced with `rederive_worker_evidence.py --phase inputs`. Parameter distances are reproduced with `compare_adapter_distances.py` using the accepted input report and the original input/reference TARs. Neither command loads a model or calls a tokenizer/provider. Temporary signed download URLs are excluded from portable review receipts.

Native evidence links: [training invocation histories](https://github.com/techrote/neurounner/tree/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/inputs/training), [all worker responses and fixed forward scores](https://github.com/techrote/neurounner/tree/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/inputs/worker), [token audit](https://github.com/techrote/neurounner/blob/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/inputs/token-audit/token-mask-audit.json), [input custody](https://github.com/techrote/neurounner/blob/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/input-custody.json), and [input release manifest](https://github.com/techrote/neurounner/blob/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/input-release-assets.json). The [shifted-loss interpretation](shifted-loss-interpretation.json) binds the original source/native reasoning to the published audit.

This portable document derives from the preserved scratch report SHA-256 `cf37377a51fa018342b1315ace28c72057c73d87d746691491d2421d8f26902d`. Its numeric input reports are copied byte for byte. The parameter-distance implementation received an independent review of archive/fingerprint binding, F32 promotion, aggregate L2 and the relative-norm denominator; no threshold or exact-reproduction flag changed.
