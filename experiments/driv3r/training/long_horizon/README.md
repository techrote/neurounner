# DRIV3R long-horizon game policy training — tiny candidate v1

**Research only. Draft/HOLD. This lane is `experiments/driv3r/training/long_horizon/**`, plus the single opt-in dedicated workflow. No native game, Workbench, provider, #568 painter, D-COMMON or D-1A modification.** The target is a slightly deranged but controllable CyberSand line worm; A* shortest-path optimisation is emphatically not the objective.

## What is genuinely implemented

- **Actual optimizer:** 625-parameter NumPy MLP candidate ranker with 24 local/host-computed action features, tanh hidden layer, masked legal-action categorical likelihood and manual Adam. Three curriculum stages at **36 -> 72 -> 128** sequential decisions, typically **180 optimizer updates each**. This is a genuinely trained **tiny decision model**, not d1-2B, LFM2.5, Sprocket or Smol, and not their LoRA.
- **Host-like synthetic world:** 96×96, Empty=0, Solid=1, material-Z=37, freshly painted worm material=58, hard protected rectangles. A one-cell cardinal MOVE is offered only for Empty, non-protected neighbors. A paint MOVE changes exactly one cell. STOP is **always** offered, including at turn 0; no host-forced 128-turn survival. Brush-up BACKTRACK and source-labelled `RELOCATE_START` are synthetic history operations with no painting. This emulates the original read-only M3 host candidate semantics at [CyberSand source `313b836...` `m3_model/short.py`](https://github.com/techrote/cybersand/blob/313b836b3c9ce2c60e4659e3763df140f84347a6/experiments/driv3r/m3_model/short.py), with historical recovery referencing [M4 `compare.py`](https://github.com/techrote/cybersand/blob/313b836b3c9ce2c60e4659e3763df140f84347a6/experiments/driv3r/m4/compare.py). The Python synthetic world is **not** a native observation adapter, validated M1 bundle, native World epoch or permission grant.
- **Teacher mixture:** contour proximity, line continuity, novelty, soft scent, route-like motion, motif-turning and source-visible recovery; intentional probabilistic small steering mistakes. Structural `trap_corridor` vignettes force recovery from a plausible bad direction. Training includes off-policy interventions, whose actions are logged separately from teacher labels, not retrospectively relabelled successful.
- **Actual matches:** learned tiny model vs the imperfect teacher mixture vs `random_legal` on *identical per-arm worlds and horizons*. Logs complete source-replay-verified original movement, material edits, menus, per-decision learned action probabilities, per-arm action distributions, cumulative ink, turns, branching, recovery, STOP, latency, RSS, source/World/checkpoint SHA-256. Original 384×384 material PNGs are rendered solely from synthetic material states and appropriately labelled.
- **Leakage separation:** optimizer consumes only TRAIN 1000-series seeds in four families; DEVELOPMENT uses different 3000-series seeds; final HELDOUT uses genuinely distinct **archipelago and staggered-vault structures** at 9000-series seeds, once after the last optimizer step. HELDOUT never generates training examples, teacher labels or decisions to tune optimizer parameters. Sources and masks are hashed. Earlier experimental `rooms`/`funnel` images were inspected while tuning v1; those families are **excluded from final heldout**, rather than falsely called blind. This contamination is recorded in `FIRST_PASS.md`.
- **Resource boundary:** no net, model hub, real SLM loading, native PAINT, paid GPU/API, or production setting. CPU bounded by 3×400 maximum optimizer updates, 128 maximum synthetic steps per episode, 12 maximum synthetic recoveries, 30 maximum training worlds per family. **No relaxation of native M3 64-logical-move limit.** A 128-decision synthetic session may consume far more than 64 cursor moves, especially after relocation: inadmissible to native without fresh owner-gated qualification.

### Exact local trial (pre-GitHub-commit; not yet an Actions artifact)

Executed Python 3.13 / NumPy 2.3.5 with seed 570, nine training worlds per family and 180 Adam updates/stage. `unittest` 11/11 passed. **540 real optimizer updates, 625 trainable parameters; ~5.6s training, ~15.8s including full trace/replay/gallery generation, 124,612 KiB peak measured process RSS**. Local model file SHA-256: `457c1debbf51bf4deaae1a328292ee676d434f563610e2df565262cfccf2f620`. Original local checkpoint is deliberately *not committed*.

| Final independent split | 128-step teacher | 128-step trained tiny | STOP-inclusive random |
|---|---:|---:|---:|
| DEVELOPMENT (12 scenarios) — mean ink cells | 126.33 | **126.75** | 2.75 |
| DEVELOPMENT — mean branch cells | 31.17 | **33.17** | 0.08 |
| DEVELOPMENT — early STOP episodes | 0/12 | 0/12 | 12/12 |
| HELDOUT (6 structurally distinct scenarios) — mean ink cells | 126.00 | **126.33** | 3.17 |
| HELDOUT — mean branch cells | 39.17 | **39.83** | 0.33 |
| HELDOUT — mean recovery actions | 2.00 | **1.67** | 0.83 |
| HELDOUT — early STOP episodes | 0/6 | 0/6 | 6/6 |

**Interpretation:** real optimizer plus generalisation to *synthetic teacher features*, not CyberSand artistic skill or d1/LFM/Sprocket improvement. STOP-inclusive random is intentionally a weak survival comparator; the strong teacher is the appropriate imitation reference. Learned teacher-like path length is not sufficient evidence of an intentional or amusing line. Recovery and structure need independent visual review and source-qualified native receipt experiments. No new native receipts were generated.

## Running

From repository root with Python >=3.11 and `numpy==2.2.6` (exact pinned Actions dependency):

```bash
python -m unittest -v experiments.driv3r.training.long_horizon.test_long_horizon
python -m experiments.driv3r.training.long_horizon.train --output /tmp/driv3r-lh-run-001 --seed 570 --n-worlds-per-family 9 --updates-per-stage 180 --batch 64
```

The output folder must **not already exist**, preventing accidental overwrite. `report.json` and `SHA256SUMS.json` bind results and actual checkpoint bytes, while `trajectories/*.json` plus `*.png` contain every original synthetic choice and replayed cell edit. The GitHub workflow stores all these bytes, including `tiny-ranker-36-72-128.npz`, as an immutable per-run **Actions artifact**, never in source Git history. Its dedicated workflow uses CPU and requires no secrets, tokens, model hubs, paid resources, native engine or PAINT.

## Next model scale-up (not implied as already executed)

First inspect independent original gallery and raw STOP/recovery distributions. Then adapt exactly the same host menu and structural split to pinned d1-2B-Q8, LFM2.5-350M, Sprocket-500M and tiny variants in **isolated** CPU optimizer jobs. The 625-parameter model is a useful smoke/provenance benchmark, not a substitute for a source-pinned SLM adapter. Current source model names mentioned here are **future targets, not loaded model identities**. For genuine native experiments, pass the shared source-fenced observation/receipt contract and separately obtain owner authorisation; synthetic `RELOCATE_START` is not a licence to teleport or exceed the native move ceiling.

**Acceptance/status:** DRAFT/HOLD, never merge or mark Ready without owner review. No production defaults changed. Per-test/optimizer failures remain visible in `FIRST_PASS.md` and the per-run Actions artifact; green general CI is not accepted as evidence for this lane.


## Development-only matched curriculum ablation (16 Nov 2026; local, not yet GitHub Actions)

A separate **predeclared** two-seed comparison trained four original 625-parameter checkpoints.
For each seed, two arms start from **identical initial weights** and receive **540 Adam
updates with identical learning-rate and sampling RNG schedules**. Only the corpus horizon
differs: staged `36 -> 72 -> 128` versus flat `128 -> 128 -> 128`. Source-pinned
DEVELOPMENT comprised four normal and four trap-corridor worlds; the structurally
sealed original heldout was not opened, and no 128-turn synthetic result authorises
native play beyond the existing 64 logical-move ceiling. Checkpoint SHA-256 and
complete source-replayed 128-turn traces are stored in the **external** run bundles.

| Seed | Model | Teacher-on-teacher-state agreement at 128 | Mean ink / 128 | Mean branch cells / 128 | Early STOP / 8 |
|---|---|---:|---:|---:|---:|
| 570 | staged | 97.27% | 126.12 | 26.00 | 0 |
| 570 | flat-128 | 96.78% | 126.12 | 24.88 | 0 |
| 571 | staged | 97.75% | 125.62 | 27.75 | 0 |
| 571 | flat-128 | 97.46% | 126.12 | 26.00 | 0 |

**Critical negative/control:** persistent host-legal random (which does not voluntarily
STOP while an ink MOVE remains) painted **91.75** and **108.62** cells with
**6/8** and **2/8** early STOP episodes, despite forming **55.88** and **66.38**
branch cells. Original STOP-inclusive random painted only **3.75** / **5.25**
and stopped early in **8/8** for each seed. Thus persistence, branching and
control are distinct: branch count by itself is not coherent line character.
These are two-seed DEVELOPMENT results, insufficient to establish statistical
superiority or neural artistry. The flat-128 arm is a valid negative/ablation,
not an inferior model by fiat; both are near-identical on some metrics.

Commands, from the repo root after publishing this source:

```bash
python -m unittest -q experiments.driv3r.training.long_horizon.test_long_horizon
for seed in 570 571; do
  python -m experiments.driv3r.training.long_horizon.compare_curricula \
    --output "/tmp/lh-ablation-seed${seed}" --seed-start "$seed" --seed-count 1 \
    --train-worlds-per-family 9 --updates-per-stage 180 --dev-worlds-per-family 1
  python -m experiments.driv3r.training.long_horizon.verify_ablation \
    "/tmp/lh-ablation-seed${seed}"
done
```

The verifier checked **80/80 original 128-step development trajectories** across
both seeds, including per-step material/legality replay and exact trained
checkpoint logits/probabilities; all four checkpoint byte hashes passed.
It also rejected a forged probability distribution even after its SHA manifest
was recomputed. Full lane tests **16/16 local PASS**. These are not exact
GitHub-HEAD or Actions qualifications until sources are actually committed and
run on the hosted runner. No real d1/LFM/Sprocket model was loaded.


## Source-replayed bad-move intervention experiment (16 Nov 2026; new local execution)

New `intervention_lab.py` produces **two genuinely trained, independently checkpointed
625-parameter candidates** with matched initial weights, 36→72→128 curriculum,
identical 300-update Adam schedules and separate source-logged **13% vs 40%
off-policy intervention** TRAIN corpora. The 40% arm is not a model-adapter
fine-tune and is not a learned WORM/d1/LFM/Sprocket result. The objective is
robustness to earlier *legal but ill-chosen* actions, not A* route quality.

On 8 predeclared DEVELOPMENT worlds (four source families; four normal and
four trap-corridor variants), four policies × three horizons × intact/forced
produced **192 original synthetic trajectories**. The forcing schedule is
zero-based decisions **12, 35, 71**. At each scheduled point, apply the
lowest *current local teacher-score* alternative **legal MOVE**, excluding the
controller's preferred choice. If no alternative exists, record a skipped
attempt; STOP is never suppressed in the real menu. This is an adversarial
*local score heuristic*, not a native host oracle, and the altered episode is
not an identical-world paired run after first divergence.

**Run seed 572 / 4 TRAIN worlds per source family / 100 updates per stage**:

| 128-decision DEVELOPMENT (8 per arm) | Intact mean ink | Forced mean ink | Intact mean branch cells | Forced mean branch cells | Early STOP intact/forced |
|---|---:|---:|---:|---:|---:|
| 13% off-policy tiny | 125.625 | 125.750 | 15.125 | 22.375 | 0 / 0 |
| 40% off-policy tiny | 126.125 | 125.875 | 27.375 | 25.375 | 0 / 0 |
| Imperfect teacher mixture | 125.625 | 123.750 | 25.875 | 23.875 | 0 / 0 |
| Persistent random legal | 73.125 | 79.750 | 38.250 | 48.750 | 6 / 6 |

Actual forcibly applied alternatives at 128: **23/24 schedule attempts** for
each model/teacher, **17/24** for persistent random. Every recorded forced
checkpoint retains the original position, preferred choice, actual legal
choice and progress for the next 16 decisions. Mean 128-step final-ink
Jaccard intact vs forced: **0.1801 / 0.2409** for 13% / 40% tiny, **0.2724**
teacher, **0.2979** persistent random. These show **trajectory sensitivity**, not
proof of better or more intentional recovery. The high-intervention model is
not uniformly better, and the small development set/one seed prohibits a
robustness-superiority claim.

**Actual resource receipts:** new experiment 20.0359 seconds, 129,780 KiB
process peak RSS; 2 × 300 updates. Per-decision measured latency, action
probabilities, distributions and 192 final material PNGs are kept per original
trajectory. `verify_intervention.py` independently checked 192/192 against
*restored checkpoint logits, original menu/state/material replay, evidence
manifest, source file hashes and PNG bytes* (387 files); forged neural
probability, forged post-force progress and held-out identity were rejected
by adversarial tests. Complete lane test run: **25/25 PASS** locally.

**Native boundary is material:** all 128-turn trained/teacher trajectories
had `logical_cursor_moves > 64`, as did all eight 72-turn worlds per arm,
so **none may be presented as admitted under the existing 64-move native
ceiling**. These are CPU-only DEVELOPMENT simulations; 0 native receipts,
0 actual game-side source changes, 0 held-out worlds revisited.

Exact reproducible commands:

```bash
python -m unittest -q \
  experiments.driv3r.training.long_horizon.test_long_horizon \
  experiments.driv3r.training.long_horizon.test_intervention_lab
python -m experiments.driv3r.training.long_horizon.intervention_lab \
  --output /tmp/lh-stress-seed572 --seed 572 \
  --train-worlds-per-family 4 --dev-worlds-per-family 1 \
  --updates-per-stage 100
python -m experiments.driv3r.training.long_horizon.verify_intervention \
  /tmp/lh-stress-seed572
python -m experiments.driv3r.training.long_horizon.paired_geometry \
  /tmp/lh-stress-seed572 --output /tmp/lh-stress-pairs.json \
  --gallery /tmp/lh-stress-gallery.png
```

The opt-in workflow now also executes this curriculum ablation and uploads
its checkpoints, receipts and original galleries **if and when** these
sources are published to GitHub. This experiment is locally executed and
independently checked, **not** an installed Actions workflow run or immutable
hosted artifact. The frozen structural heldout was not opened.

## Model-only three-intent controllability, local qualification 16 Nov 2026

`intent_lab.py` exercises a deliberately narrow question: can a 625-parameter
host-legal action ranker actually *change its drawing* when its requested
`wall`, `route`, or `motif` intent changes, instead of merely enduring 128 turns?
The command is temporarily overlaid on model-side input features only and
removed before the host-authoritative `World.apply`. Every original World
state, legal menu and material edit remains replayable under its original spec.

Four truly trained, matched 36→72→128 CPU policies (seed 573) compare (1) ordinary
single-intent labels, (2) mode-blind ablation, (3) three counterfactual teacher
labels on identical TRAIN host states, and (4) the same counterfactual corpus
with explicitly **hand-engineered action×intent feature interactions**. The last
is a deliberately easier representation control, NOT an emergent art policy;
its wall/route/motif interactions use local geometry and candidate direction,
not a hidden future-route oracle. They never authorize material edits.

Exact bounded local run: `--seed-start 573 --seed-count 1
--train-worlds-per-family 2 --updates-per-stage 25 --dev-worlds-per-family 1`.
Each arm: 75 genuine Adam optimizer updates, 625 parameters, checkpoint
SHA-256, original per-decision probabilities, full original material/host-action
replay. Four DEVELOPMENT structural families (one seed each), with three
commands per base World; separate fixed and predeclared switched 128-turn
commands (route→wall at turn 36; motif→route at turn 72). 176 synthetic episodes,
15,424 originally recorded model choices, 0 native receipts. Native 64-logical-
move limit unchanged; 32/44 episodes per arm exceeded it.

| 128-decision DEVELOPMENT, seed 573 | Mean pairwise ink Jaccard across intents | Distinct ink geometries / 3 | Motif turn fraction | Teacher-local argmax agreement, all horizons |
|---|---:|---:|---:|---:|
| Ordinary intent | 1.0000 | 1.00 | 0.0276 | 0.6053 |
| Blind intent | 1.0000 | 1.00 | 0.0276 | 0.6126 |
| Counterfactual labels | 0.4201 | 3.00 | 0.0909 | 0.6800 |
| Counterfactual + action×intent features | **0.1415** | **3.00** | **0.9960** | **0.9606** |

The principal negative result is important: the *ordinary conditioned* ranker
painted exactly the same geometry regardless of command, despite receiving mode
bits. The paired counterfactual teacher differs on ~90–92% of TRAIN states,
so the signal is present but not easily expressed in this representation.
The hand-engineered interaction representation makes short-horizon command
control visibly effective, with almost every action turning in motif mode.
That is an extreme zigzag, not necessarily good composition: a clear new
failure/limitation rather than a claim of beautiful artistic behaviour. The
worlds are tiny synthetic approximations, and the controlled ablation has
**one seed and four DEVELOPMENT worlds**, so it is not a generalization claim.

`verify_intent.py` independently replays each host-menu/material step and
recomputes every decision's neural choice and probability distribution from
restored checkpoint weights, hashes and the exact source. It refuses held-out,
foreign world identities or inconsistent source pins. The SHA-256 manifest
binds bytes but is not an authenticated signature. These outputs are local
until the GitHub workflow runs; checkpoint binaries remain out of Git.

Use the dedicated CPU workflow or run locally:

```bash
python -m unittest -q experiments.driv3r.training.long_horizon.test_intent_lab
python -m experiments.driv3r.training.long_horizon.intent_lab \
  --output /tmp/driv3r-intent-seed573 --seed-start 573 --seed-count 1 \
  --train-worlds-per-family 2 --updates-per-stage 25 --dev-worlds-per-family 1
python -m experiments.driv3r.training.long_horizon.verify_intent \
  /tmp/driv3r-intent-seed573
```

**Limits:** No sealed HELDOUT read; no actual d1-2B, LFM2.5 or Sprocket
weights; no sequential native PAINT receipts; 128 synthetic decisions are not
an extension to the existing 64-logical-move native ceiling. Further work:
independent multi-seed testing, preference-conditional re-steering at late
turns, and geometry-based human review to filter trivial jagged motifs.
