# CyberSand GAME Corpus Factory — standalone research, v1

**Authority:** owner-approved Game Corpus Factory lane in `experiments/driv3r/corpus_factory/**`. **Draft/HOLD.** No modifications to `native/**`, `godot/**`, Workbench, production policies, source assets or maps. No native runtime evidence or checkpoint claims.

## What is actually implemented

- Reproducible 96×96 **external/synthetic** local material grids, obstacles, immutable protected cells, reachable spatial goals, corridors, deceptive choices, dead ends, motif steering and long histories.
- Three matched motif/family trajectories per world, with W0-shaped `x/y/heading/local_5x5/legal/last_actions/family/tick` observations and **additional** `local_material_5x5`, `local_protected_5x5` and goal channels. Legal choices: STRAIGHT, LEFT, RIGHT, BRANCH, RETRACE, STOP. The host, never a proposed policy, controls legality and application.
- Synthetic *heuristic teacher* actions, host receipts, legal-but-less-preferred and explicitly illegal hard negatives; this is **not** a trained policy, an optimal-route certification, or authentic native PAINT execution.
- TRAIN/DEV/heldout partitioned **by underlying scenario+world seed, not decision**. Three motif families on each world stay in the same split. Near-identical world rejection, local observation-input dedup, cross-split collision exclusion and post-export integrity auditor.
- Human-inspectable original PNGs, immutable SHA-256 inventories and machine-readable JSONL shards with source, split, family, difficulty and evidence class.
- Optional **read-only** importer for original WORM W0 Action ZIPs. Preserves original JSON and PNG digests; imported observations remain **evaluation-reference-only** and **never become training targets**. Neither a model trace nor a Python W0 run is silently reclassified as native.

**Pinned donor contract**: [W0 `worm.py`](https://github.com/techrote/cybersand/blob/research/612-worm-common/experiments/agent_painter/worm/common/worm.py), Git blob `8979b6f0296e81051194bacc907992d38e006605`. This corpus extends W0's *external research* observation and protection semantics; its material/protected features are **not supported by vanilla W0**, so it is not byte-for-byte W0 replay.

## Reproduce (Python 3.11+ standard library only)

```sh
python3 -m unittest discover -s experiments/driv3r/corpus_factory -p 'test_corpus.py' -v
python3 experiments/driv3r/corpus_factory/corpus.py --output /tmp/cs-game-corpus-v1 --worlds-per-scenario 50 --budget 160 --png-limit 12
python3 experiments/driv3r/corpus_factory/corpus.py --output /tmp/cs-game-corpus-v1 --audit-only

python3 experiments/driv3r/corpus_factory/ingest_w0.py /path/to/original_sprocket_F3_actions_artifact.zip --output /tmp/cs-game-corpus-references
```

The source `corpus.py` is deterministic with fixed scenario templates, local PRNG namespaces and JSON serialization. Do **not** feed DEV, heldout, or `w0-reference.jsonl` into a model optimizer. Trainers must **not** feed raw `train.decisions.jsonl` into model prompts: it contains candidate scores, receipts and teacher labels. Consume only the validated `train.messages.jsonl` from `export_train.py`; keep DEV separately bounded and heldout untouched.

`manifest.json` records exact generator version, reference blob pin, parameters, hashes, counts and discard statistics. `*.worlds.jsonl` provides per-world obstacle/protection/seed/provenance and material-grid digest. `*.episodes.jsonl` carries complete host-applied trails; `*.decisions.jsonl` carries supervised observation/choice/receipt pairs with hard negatives. Source-derived references belong only in the separate ingestion manifest. The generator does not download models or start optimizers; no paid compute.

## Evidence, limitations, and handoff

Initial local dry run: 10 world groups per scenario, 60 groups / 180 family episodes / 24,855 retained decision inputs, 6 raster PNGs; integrity audit PASS; six focused unit tests PASS. This is **synthetic-only**, and does not prove any model capability. Independent GitHub Actions qualification and a larger hosted artifact remain to be completed; generated binary corpora are not committed to Git history.

For WORM model workers (#575 LFM, #576 Sprocket), a practical next step is to consume a finite TRAIN slice in existing primary-owned optimizer lanes, evaluate on DEV by independent world group, preserve heldout unchanged, and publish *equal-seed, equal-budget* visual comparisons and checkpoint pins. For DRIV3R #570, supply untuned inference examples and family-sensitive motif tests, not competing optimizer dispatchers. No safety admission, production model promotion, native bridge change, merge or Ready transition is implied.

### Known limitations

- Protected material is immutable and non-traversable **only in this synthetic extension**; it is not a claim about current native material interaction.
- The teacher uses goal-distance and texture heuristics, not a globally optimal traversal planner. Legal lower-scoring alternatives are **heuristic hard negatives**, not authoritative incorrect actions.
- World material layouts are deterministic analytical fixtures; physical sand, water and Rapier simulation are **not** run.
- W0 references are preserved separately and designated evaluation-only to prevent self-training on model evaluation traces.


## Safe TRAIN-only handoff to model owners

`train.decisions.jsonl` is **not itself a safe prompt corpus**: each row contains teacher labels, whole-world/source identifiers, candidate scores and action receipts. Never interpolate an entire source row into the model input.

Use the separate positive-allowlist exporter. It verifies the full original manifest and corpus audit, checks exact synthetic TRAIN lineage and each pre-action observation fingerprint, then emits only observable host inputs in the user message and the one synthetic teacher token in the assistant message. A distinct, analysis-only sidecar preserves record/group/SHA provenance. **No DEV, heldout or W0-imported evaluation reference is loaded into the exporter.**

```sh
python3 -m unittest discover -s experiments/driv3r/corpus_factory -p 'test_*.py' -v
python3 experiments/driv3r/corpus_factory/export_train.py \
  --corpus /tmp/cs-game-corpus-v1 \
  --output /tmp/cs-game-corpus-v1-train-800   --per-action-cap 800 --per-world-cap 80 --sampler-seed 0
```

The output `train.messages.jsonl` is an action-token chat-message TRAIN slice; `train.provenance.jsonl` and `export-manifest.json` are verification/analysis, not prompts. The selected samples are deterministically hash-ranked and capped per target action and source world, reducing the original STRAIGHT-heavy label skew without falsifying the source distribution. The receipt pins the source manifest SHA-256 and output hashes. Export refuses to write inside the source tree or overwrite an existing export. Models must record their own exact root, adapter and training-job hashes; this exporter launches **zero** optimizers. This is a synthetic curriculum handoff, **not** admission to native gameplay or qualification of an intelligent policy.

New exporter tests check deterministic bytes, true TRAIN-only group provenance, target metadata exclusion, tamper-resistant per-observation fingerprints and rejection of model-derived rows even if someone rehashes the corpus manifest. Hosted qualification of the new exporter has **not** yet been established.

### Original W0 long-horizon diagnostics (read-only)

`w0_reference_diagnostics.py` accepts only the quarantined, evaluation-only
`w0-evaluation-decisions.jsonl` ZIP produced by `w0_eval_decisions.py`.
It validates the provenance inventory, split quarantine, host legality, tick
continuity and four matched controllers before reporting unique cells, action
histograms and novelty in 32-decision windows. It **never exports training
labels**. Run `python3 -m unittest discover -s experiments/driv3r/corpus_factory
-p 'test_w0_reference_diagnostics.py' -v` and then
`python3 experiments/driv3r/corpus_factory/w0_reference_diagnostics.py
/path/to/quarantined_eval.zip --output /tmp/w0-diagnostics.json`.

The original Sprocket F3 four-world 128-decision evaluation averaged 71.0
unique cells for model-assisted steering versus 97.0 for random legal,
104.75 for fidget and 129.0 for wall. Model novel arrivals per 32-decision
window averaged 22.75, 17.50, 16.75 and 13.00. Those are descriptive
comparison results, not trained-teacher instructions, native acceptance, or
evidence of statistical generalization. Exact Git blobs for diagnostics and its
tests passed 2/2 local tests. Separately, strict deterministic replay of every
TRAIN episode and candidate action is **implemented but not yet executed**
against exact committed exporter source; do not infer exporter qualification.


## Family-conditional steering signal: measured TRAIN export, 16 Nov 2026

The v1 teacher's **observable family signal is nearly absent**. Exact matched-state analysis of the independently reproduced 4,313-message **TRAIN-only** export found 815 within-world pre-action contexts repeated across at least two family tokens. Only **3/815 (0.368%)** had different recorded teacher action labels; 812 had the same label. This is a selected TRAIN slice, not proof about all unexported producer decisions. It reinforces the full-source visual finding that 122/180 worlds rendered identically across families.

New scoped [family_contrast_probe.py](family_contrast_probe.py) accepts **only the already verified TRAIN-only export ZIP**, verifies its member hashes, row counts, source manifest class and per-input SHA, and produces a separate family-signal audit. Optional `--emit-probes` creates deliberately counterfactual, host-legal **single-step** preference triples: same source state with greebles→BRANCH, pipework→STRAIGHT, architecture→LEFT/RIGHT alternating by 8-tick phase. These are **NOT original teacher actions, NOT host-applied trajectories, NOT native receipts, NOT success/quality evidence**. They are an opt-in diagnostic or experimental TRAIN curriculum, never a DEV/heldout benchmark; never evaluate on their original source worlds.

A real run used `--cap 480 --per-world 6 --seed 0 --emit-probes`: **480 distinct source states / 131 TRAIN source worlds / 1,440 host-legal single-step prompts**, balanced across three family tokens; bounded quotas and deterministic hash-based sampling. Independent second-run file hashes and a separate native-safety boundary remain required for any later promotion. This is explicitly a new experimental data class, not a replacement for v1's 4,313 authentic synthetic-teacher examples.

Reproduce from the previously issued TRAIN-only ZIP:

```sh
python3 -m unittest discover -s experiments/driv3r/corpus_factory -p 'test_family_contrast_probe.py' -v
python3 experiments/driv3r/corpus_factory/family_contrast_probe.py \
  /path/to/cybersand_game_corpus_v1_train_only.zip \
  --output /tmp/cs-family-contrast-v1 --emit-probes --cap 480 --per-world 6 --seed 0
```

Five exact-Git-blob-matched local tests PASS; GitHub Actions corpus-specific qualification **not claimed**. The output report and two optional JSONLs carry their own SHA-256 manifest, source export ZIP hash, original corpus manifest hash and non-execution provenance. Trainer consumption requires explicit WORM/DRIV3R model-owner opt-in; do not silently combine counterfactual one-step labels with the original teacher or held-out evaluation. No optimizer was started.
