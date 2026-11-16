# DRIV3R — D-STUDENT-001 real d1 anchor / synthetic-game distillation

**Research Draft/HOLD only. No game code, Workbench, native engine, production provider or default is changed.** Student checkpoint files stay in immutable GitHub Actions artifacts, never Git history.

## What is genuinely sourced

`prepare_corpus.py` validates two independently archived **real d1-2B Q8 System One Frankenlayer inference** artifacts, with exact ZIP SHA-256, original parent observation, raw selected action and per-action probability. The 144 original calls comprise **48 stock**, **48 GREYHOUND** and **48 BRANCHWITCH** decisions. All original parent calls used **source seed 570** and three closely overlapping observation families; there is NO independent parent generalization result.

- GREYHOUND + stock: [original Actions artifact 14181164165](https://github.com/techrote/cybersand/actions/runs/43764117645/artifacts/14181164165); ZIP SHA `e4bf6b63997967cb0ccc1b4a34e517c7abb62a79d5dc8069e93b9a02371ecff3`.
- BRANCHWITCH: [original Actions artifact 14182441899](https://github.com/techrote/cybersand/actions/runs/43764117645/artifacts/14182441899); ZIP SHA `0ca2cf83eca44e3462a29ba0a27ae41e62d342bb8a4224f677539a26b68bc181`.
- Deterministic extracted original-teacher JSONL SHA `140342c16a0b8c6c8d2c3bc6071767885b15096d5a53bbf304979f3015c86522`; source schema `cybersand.driv3r.distill.original-teacher/1`.

**The parent is NOT run on any new synthetic situation.** Original parent distributions are probability anchors for deliberately synthesized, obstacle-conditional policies. `aerohound_proxy` is a named *synthetic oscillation approximation*, NOT an observed/checked AEROHOUND model.

## Actual student training and comparisons

External CPU-only, 24×24 obstacle/protected/painted-cell host uses six W0-like actions: STRAIGHT, LEFT, RIGHT, BRANCH, RETRACE, STOP. The *external simulator does not reproduce native W0 execution*. The student is a ~5,142-parameter float32 100→48→6 tanh MLP, genuinely trained with 450 Adam optimizer updates against a soft-label teacher mixture. Train obstacle seeds 1–64, validation 1001–1016, final worlds 2001–2016. The synthetic states/trajectories are not genuine parent inference, and synthetic structural decisions do not replace real student-vs-parent replay.

One 450-step local CPU preflight: 5,519 synthetic train examples; held-out synthetic teacher top-1 agreement 84.7%, held-out soft-target cross-entropy 1.092 versus raw untrained 1.826. (Training seed and optimizer order are explicitly reproducible; the hosted source HEAD must be separately qualified.) Matched final 40-turn synthetic test: 16 obstacle worlds × four styles × six controllers = 384 world-controller episodes. Trained student averages 32.09 unique cells and 17.19 branches. **Flat legal random including STOP** quits after 5.11 turns on average and is an inappropriate coverage comparator; **flat legal with optional STOP excluded** averages 23.09 cells and 8.22 branches, while fidget averages 30.94 cells and 5.75 branches. These are illustrative synthetic trajectories, not proof of useful real CyberSand steering.

The report preserves voluntary STOP rate, model-vs-untrained controller, host-invalid actions, visits, retraces, repetition, and original gallery PNGs. The student's deliberately branch-heavy outputs often make crowded blobs; subjective amusement and held-out teacher CE are distinct.

## Reproduction

```sh
python3 -m pip install numpy pillow
python3 -m experiments.driv3r.training.distillation.prepare_corpus GREYHOUND.zip BRANCHWITCH.zip original_teachers.jsonl
export DRIV3R_TEACHER_CORPUS="$PWD/original_teachers.jsonl"
OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s experiments/driv3r/training/distillation -p 'test_*.py' -v
OPENBLAS_NUM_THREADS=1 python3 -m experiments.driv3r.training.distillation.lab --out ./evidence-student --steps 450
```

Source-only files live under this path; the dedicated `driv3r-game-distillation-curriculum.yml` GitHub Actions workflow retrieves/verifies immutable parent artifacts, trains a new student, records `student.npz`, `metrics.json`, `rollouts.jsonl`, gallery PNG, original teacher JSONL, source-HEAD identity and artifact SHA256SUMS. No paid API, GPU or native authoring. **Do not promote a faithful-parent or trained-model freeze:** original 16-turn parent evidence shares one seed; new source-pinned parent episodes in obstacle-adverse worlds, 64–128 turn W0 closed loop, and real expressive-recovery comparisons remain the next falsification gate.


## D004 — trap-aware style retention, still synthetic only

The `trap_rebalance` lane starts from the immutable D003 style-restored
checkpoint. It generates new seed-disjoint on-policy states with stock/GREYHOUND/
BRANCHWITCH/synthetic-AEROHOUND styles and protected cells, walls, revisits,
dead ends and dynamic trap closures. It combines *checkpoint-derived* D003 style
probabilities, D002 recovery probabilities and explicitly synthetic one-step
novelty targets. No new d1 parent, LFM, Sprocket or AEROHOUND forward is run.

Each of three mixing fractions (0.15, 0.30, 0.45) receives 180 genuine
float32 Adam updates; the winner is selected only on obstacle seeds 1017–1024
before evaluation on untouched 3001–3016. Training uses seeds 65–112.
D004 additionally archives original replayed 80-decision trajectories,
seed-block paired confidence intervals, CPU checkpoints, a reproducible
soft-target corpus (including legal menus), protected-cell adversarial tests,
and actual synthetic-model PNG comparisons. STOP remains legally available.

Reproduce after the D001 → D002 → D003 source-matched steps:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m experiments.driv3r.training.distillation.trap_rebalance \
  --base evidence-student/student.npz \
  --repair evidence-student/repair_student.npz \
  --restored evidence-student/style_restored_student.npz \
  --style-metrics evidence-student/style_rehearsal_metrics.json \
  --out evidence-student --steps 180
```

Interpretation: *synthetic exploration/style experiments*, not evidence of
faithful Frankenbrain trait transfer or a native CyberSand-compatible policy.
Keep Draft/HOLD, no game/production changes; archived model binaries remain
in Actions artifacts only.

## D005 — frontier rescue (source + hosted Actions qualification)

`frontier_rescue.py` extends D004 with bounded short-horizon synthetic
frontier/recovery labels, protected-cell exclusions, hard negative states and
D003 behavioural rehearsal. The teacher oracle may look ahead four cells
through painted corridors, but the 5,142-parameter student observes only
the existing **100-element local observation**; those oracle-only variables
must never be presented as student observations or genuine parent logits.

**Disjoint seed ranges:** train 113–176, validation 1025–1032,
test 4001–4016. Three candidate mixing fractions (0.15/0.30/0.45) each
receive 160 real CPU Adam updates (480 total); choose on validation only.
All checkpoints are kept in Actions artifacts rather than Git. Original
D001–D004 model pins are checked against D004 metrics before training.

Prior **local-only** D005 preflight (2026-11-16, not yet source-HEAD Actions
qualification): 64,361 synthesized training observations, 3,391 harder
squeeze/dead-end states, and 1,152 matched 80-decision held-out episodes.
Validation selected 0.30; the local checkpoint SHA-256 was
`c2fc59df4e49e4ded91cb26533c3cc55e1dfeba8accb1963e738e9f45c2c82ad`.
Paired test mean (D004→D005): normal cells 70.22→72.14 and branches
25.92→28.39; dynamic traps 62.78→63.89 cells and 18.34→22.19 branches;
protected squeeze 16.89→17.08 cells and 5.69→7.27 branches. Coverage
intervals overlapped zero; squeeze revisits remain severe. Local original
action receipts and 24 PNGs replayed byte-for-byte, but these are **synthetic
external-host student decisions, not native CyberSand nor source-parent
forwards**. Source/environment variation may change exact hosted hashes.

To rerun after D004 in the dedicated workflow:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m experiments.driv3r.training.distillation.frontier_rescue \
  --base evidence-student/student.npz \
  --repair evidence-student/repair_student.npz \
  --restored evidence-student/style_restored_student.npz \
  --rebalance evidence-student/trap_rebalanced_student.npz \
  --source-metrics evidence-student/trap_rebalance_metrics.json \
  --out evidence-student --steps 160
```

The stage saves `frontier_corpus.npz`, three distinct candidate checkpoints,
the validation-selected student, independent rollouts, adverse PNG galleries,
original action/PNG replay evidence and a recursive SHA-256 manifest.
The workflow is scoped to this research branch and cancels obsolete
push-triggered runs; it must not modify or merge game code.

**Research gate remains HOLD:** the 144 archived stock/GREYHOUND/BRANCHWITCH
parent decisions all derive from source seed 570; no fresh parent execution,
true AEROHOUND or LFM/Sprocket teacher mixture, native host qualification,
or faithful parent-trait transfer is claimed. The next falsification priority
is seed-independent real teacher trajectories and protected-squeeze recovery.

## D-STUDENT-006: explicit exhaustion versus premature STOP (16 November 2026)

The external protected-squeeze environment frequently consumes ~56 of its
80 decisions after exhausting the local reachable pocket. D005 lacks effective
STOP preference. D006 tests whether a 5,142-parameter MLP can learn
**conservative early termination** instead of retracing/overpainting forever.

Implementation: exhaustion_recovery.py; regression: test_exhaustion_recovery.py.
Inputs are the source-matched D005 trained checkpoint and frontier_metrics.json,
with exact SHA-256 checks. The oracle/evaluator computes an external BFS through
traversable cells, **also treating STOP-only host menus as exhausted**.
Geometric reachability can differ from future six-action reachability
(heading/BRANCH/RETRACE), so this is a conservative synthetic diagnostic,
NOT a native CyberSand API. The MLP receives only the original 100 local
features; full-map reachability is a teacher label, not an inference feature.

Corpus: 48 training seeds 177..224 x four styles x three scenarios x
D005/flat-legal controllers; hard negatives include paint-filled local
windows with still-reachable distant frontier. Validation seeds 1041..1048
select a mixture; test seeds 5001..5016 remain disjoint from earlier tests.
Three genuine 140-Adam-update candidates have mixtures 0.35/0.60/0.85.
All action choices obey the synthetic host legal menu.

Local research preflight against the previously hosted D005 checkpoint:
91,596 examples, 12,522 exhausted positives, 16,244 difficult live
negatives, 420 real Adam updates. Validation selected 0.85. Across 64 unseen
squeeze episodes, D005 -> D006: unique cells 17.02 -> 15.45, branches
7.38 -> 6.62, justified STOP 0% -> 84.4%, and wasted decisions after
exhaustion 55.69 -> 6.83. D006 prematurely STOPs in **15.6%** of squeeze
episodes. Normal coverage increases 69.78 -> 72.13 while dynamic-trap
coverage decreases 64.02 -> 63.00. The premature STOP and lost squeeze
coverage are negative findings; do not claim a coverage improvement.
These are local preflight results, not hosted/native acceptance.

Reproduction: python3 -m experiments.driv3r.training.distillation.exhaustion_recovery
--base /path/to/frontier_student.npz --metrics /path/to/frontier_metrics.json
--out /path/to/new_evidence --steps 140 (put arguments on the same shell line).

The EXISTING single read-only distillation Actions workflow now runs D006
after D005 and uploads action receipts, original PNGs, candidate/checkpoint
weights, corpus, metrics and SHA256SUMS. Model binaries stay in immutable
Actions artifacts, NEVER Git. No new parent d1, AEROHOUND, LFM/Sprocket
or native CyberSand inference took place. Draft/HOLD, no merge.