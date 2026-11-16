# Cross-generator closed-loop CPU model transfer — 16 Nov 2026

**Source:** techrote/cybersand PR #638, Draft/HOLD, `experiments/driv3r/corpus_factory/**` only.

**Executed:** two genuinely fitted 24-epoch CPU scikit-learn SGD/log-loss classifiers (v1 TRAIN-only source; unexecuted family-contrast TRAIN probe source), then host-applied up to 128 decisions on an **independent new deterministic synthetic** 96×96 generator (world seeds 3, 7, 11, 13, 19, 23). Three matched family labels per world and three policies per family: **54 closed-loop rollouts, 6,841 executed host-legal decisions, 54 original PNGs**. This is NOT native CyberSand, official DEV/heldout, or trained Sprocket/LFM.

| Policy | Training source | Runs | Mean distinct visited | Goal arrivals | Early STOP | Illegal/protected hits |
|---|---|---:|---:|---:|---:|---:|
| v1 synthetic-teacher SGD | 4,313 TRAIN messages / 144 source worlds | 18 | 107.56 | 0 | 1 | 0 |
| Artificial family-contrast SGD | 1,440 unexecuted counterfactual TRAIN messages / 131 source worlds | 18 | 80.61 | 1 | 0 | 0 |
| Host-legal seeded random | no optimizer | 18 | 61.11 | 0 | 0 | 0 |

**Novel visited-cell arrivals per 32-decision window:** v1 = 21.17 / 30.94 / 29.39 / 25.06; family-contrast = 22.06 / 19.94 / 18.89 / 18.72; random = 19.33 / 16.22 / 11.89 / 12.67. V1's classifier maintained novelty through later windows on these cross-generator worlds, so long-history collapse is **not universal**. The artificial family-contrast classifier does not prove compositional drawing style despite strong single-step TRAIN-internal accuracy; it transferred with less novel-cell exploration than v1. The single goal arrival (contrast seed 13, pipework, tick 62) is too sparse for a performance claim. Random excludes voluntary early STOP; this is not a fully parity-controlled benchmark.

Both models used `SGDClassifier(loss='log_loss',alpha=.0001,max_iter=24,tol=None,random_state=17,shuffle=True,learning_rate='optimal')`, with an exact-source `DictVectorizer` from `learnability_probe_scoped.features`. Genuine optimizer epochs: 24 each, `t_` counters: v1 103,513 and contrast 34,561. The training datasets are pinned ZIP artifacts rather than Git binaries: v1 SHA-256 `80b11c731e29306dd04c0b09539ab8fc7796592933276a13e7e44e83f17f6dbc`; contrast SHA-256 `ae3bcb2c9d06fca4ab69f427dba5e859da4638882850a26fe14a4103b97c2d2e`.

All actions passed the host legal menu. BRANCH chooses a fresh arm (RIGHT/LEFT/STRAIGHT), RETRACE pops a real history stack, and STOP ends a rollout. All obstacles/protected cells remain nontraversable. The original host-rendered PNGs depict actual proposed-and-applied synthetic paths on material grids, not one-step counterfactuals or native artwork. Original PNGs and their per-file SHA-256 hashes are in the independently generated evidence archive and `report.json` inventory, not Git.

**Source reproducibility:** two independent full executions produced byte-identical reports, SHA-256 `4cccee7ed03d8f425043e767efbab76747f2b01716405f54270d87cfc6f0b4dc`. Five focused local tests PASS, covering deterministic seeded world and reachability, immutable-cell host application, STOP legality, bad source pin rejection, and source hashes. Python environment: scikit-learn 1.8.0, NumPy 2.3.5. 54 PNGs all have source digests in JSON. No model/checkpoint binary saved in Git.

Reproduce with previously published source ZIPs:

```sh
python3 experiments/driv3r/corpus_factory/closed_loop_transfer_probe.py \
  --v1 /path/to/cybersand_game_corpus_v1_train_only.zip \
  --probe /path/to/cybersand_corpus_family_contrast_probe_v1.zip \
  --output /new/output/directory --budget 128
python3 -m unittest discover -s experiments/driv3r/corpus_factory \
  -p 'test_closed_loop_transfer_probe.py' -v
```

**Limits:** new independent synthetic generator, NOT official DEV/heldout and not native readback. Only six worlds, two different training sources, and a simple random control mean this is exploratory transfer evidence, not proof of comparative model quality, general artistic intent or a production policy. Artifact not yet published as immutable GitHub Actions evidence; local results do not imply hosted CI qualification. All PRs stay Draft/HOLD; do not merge, mark Ready, or modify game files.
