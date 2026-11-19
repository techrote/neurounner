# Independent worker evidence review

This supplementary analysis describes the actual published Stage F inputs. Read [INPUT_FINDINGS.md](INPUT_FINDINGS.md) for the scientific findings and limitations. The primary blinded relevance result remains pending; no figure has been rendered from partial or invented results.

## Evidence and provenance

- [Complete numeric input reconstruction](worker-input-analysis.json): 12 fresh training histories, 1,440 steps, 300 responses, 720 mask sequences and 288 fixed forward batches.
- [Compact execution and forward findings](worker-input-findings.json): actual paired runner/order proof, historical runtime differences and stage-level numerical changes.
- [Numerical parameter distances](adapter-distances.json): 18 verified adapter archive identities and all 5,040 per-tensor comparison rows. Exact non-reproduction flags remain unchanged.
- [Native shifted-loss interpretation](shifted-loss-interpretation.json): prompt boundaries and the penultimate assistant boundary have different downstream contexts; no target labels are deleted.
- [Published/preliminary audit equality](token-audit-custody.json) and [portable custody](worker-custody.json): public numeric IDs, sizes and hashes, with no signed download URLs.
- [Independent input acceptance](../input-acceptance.json): shared unchanged full release/Git/weight-byte verification, SHA-256 `c6f882d2755f293cfd236452619ff039f161843e0b99aabdf44ab5958fbeefc3`.

Native source evidence remains in the [immutable published inputs](https://github.com/techrote/neurounner/tree/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/inputs). This review directory contains derived numeric findings and read-only analysis scripts, without model weights or duplicate archive payloads. [input-bundle-manifest.json](input-bundle-manifest.json) distinguishes byte-identical copies from portable derivatives and binds their original hashes.

## Reproduce the input analysis

Run from the repository root with Python 3.12 and NumPy installed. The raw-weight comparison needs the exact accepted archive filenames in the local input and reference directories. They are listed with release tags and SHA-256 digests in [adapter-distances.json](adapter-distances.json) and the [frozen worker plan](https://github.com/techrote/neurounner/blob/783a73ca7c8e97c3e03ef4e56fb5a5b125d58785/experiments/village-001/diagnostic-20261119/worker-plan.json). Output paths must be new and outside the checkout.

```bash
python experiments/village-001/diagnostic-20261119/review/worker/rederive_worker_evidence.py \
  --repo "$PWD" --phase inputs --out /tmp/village001f-worker-input-review

python experiments/village-001/diagnostic-20261119/review/worker/compare_adapter_distances.py \
  --repo "$PWD" \
  --acceptance experiments/village-001/diagnostic-20261119/review/input-acceptance.json \
  --input-assets /path/to/verified-input-assets \
  --reference-assets /path/to/verified-reference-assets \
  --out /tmp/village001f-adapter-distances.json
```

These commands read saved evidence and raw tensor bytes; they do not load a model, retokenize data or call a provider. Full input acceptance authenticates the release/archive/Git relationship separately.

## Complete results after judging

The same `rederive_worker_evidence.py` accepts `--phase final` only when the complete native results exist. It independently reconstructs the 72 blinded pair identities, corrected/reproduced orientation and exact rational primary means, plus the 156 contextual identities. It preserves the frozen overall +0.5 gain, four positive members and −0.5 member harm bound. Six members are the replication units.

`plot_worker_results.py` accepts a complete final `results/summary.json` and new SVG/PNG output paths. The member harm boundary is drawn only across member rows; the overall-gain target is drawn only across a separate population-mean row. It adds no confidence interval or significance claim. Its optional Sprocket panel is a separate descriptive comparison. No final score or figure is included in this input-only bundle.
