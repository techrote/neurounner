# DRIV3R Population Lab — P0 neural policy ecology

**Target:** eventually supply small, weird, useful host-authorised CyberSand game/painting controllers. **External research only.** Do not edit game code, native physics, scenes, Workbench, materials, maps, assets, or production providers.

## P0 executable population

A separate 22×22 synthetic world offers a finite legal action catalog: cardinal single-cell paint into empty/unprotected cells, logical brush-up RETRACE along a known parent tree, and STOP. Synthetic obstacles, a protected rectangle, a narrow wall, and a mid-episode goal change exercise game-facing control.

Six training niches: navigator, branch artist, recoverer, texture worm, material cautious, chaotic brush. These are **20-feature / 24-hidden-neuron neural MLP policies**, NOT pretrained language models. They receive real Adam gradient updates on niche utility targets from sequential on-policy/teacher-augmented game episodes. This cheap, falsifiable proxy infrastructure precedes actual SLM adapter/student training.

Default: 6 niches × 3 initial policies + 1 mutated/retrained child each = 24 candidates, 36 training seeds, 8 selection-only seeds, and 12 untouched final holdouts. PBT chooses using selection worlds only. Weights, optimizer-step counts, rejected candidates, parent lineage, split-hash manifests and final comparisons are retained in original experimental outputs.

## Run

Requires Python 3.11+ and NumPy 2.3.x.

    python -m unittest discover -s experiments/driv3r/training/population -p 'test_*.py' -v
    python experiments/driv3r/training/population/population_lab.py --output /tmp/population-619 --seed 619

Output directory must not exist. Original synthetic research artifacts are produced under results.json, models/*.json (real trained weights), gallery/*.svg and gallery/*-trajectory.json. Each run is seed-replayable. Generated weights are intended for immutable GitHub Actions artifacts, not Git history.

## Controls and interpretation

- Uniform legal random: STOP is sampled, so this is a deliberately weak control.
- Nonstop random: uniform among legal nonterminal choices, STOP only if stuck; **critical fairer random control**.
- Persistent: deterministic heading-first non-neural controller.
- Synthetic oracle: greedy scorer using the same handcrafted reward proxy; a privileged local control, NOT a game quality metric.
- Host prefilters illegal actions, so 100% menu legality does **not** establish learned safety.
- Niche rewards are synthetic handcrafted proxies, not human judgements or demonstrated gameplay improvement.
- Descriptor diversity uses painted coverage, retrace fraction, turns, branching-junction frequency, net displacement. Do not infer aesthetic quality from a scalar alone.
- No native Workbench PAINT, real model-language inference, source-faithful Godot material readbacks, or production integration occurs in this experiment.

## Next concrete waves

1. Held-out observation-sensitive versus frozen-action-prior counterfactuals; delayed obstacles, misleading goals, dead-end recovery.
2. Novelty-search survivor archives and true population diversity, compared against reward-only PBT; carefully evaluated adapter/weight mixing.
3. Train real sub-1B language-model adapters on the legal-candidate-choice contract using this population as one of several teachers. Measure genuine model inference and generalisation; do not rename MLP results as SLM results.
4. Extend materials/observations using **READ-ONLY** real CyberSand schemas. Any new game-side interface must be proposed as a handoff, not implemented here.

Draft/HOLD only. No merge, game edits, paid API/GPU spending or production provider modifications.


## P1 — actual retraining, novelty archive, and observation ablations

Run the P0 command first, then run:

    python experiments/driv3r/training/population/population_p1.py --p0-output /tmp/population-619 --output /tmp/population-620 --seed 620 --offspring 4 --train-episodes 18 --selection-episodes 8 --holdout-episodes 12

P1 reloads all six exact P0 champion neural checkpoints, builds four new offspring per niche (mutation + retraining, substantial parameter perturbation, and cross-niche weight mixing + retraining), and **executes actual Adam optimizer updates**. It retains all 30 evaluated policy weights and their lineage, selection metrics and model digests, not just six champions. Mixing identical-shape MLP weights is an exploratory control; it does not establish that arbitrary transformer checkpoints/adapters can be merged.

Selection and held-out seeds are newly derived and disjoint from P1 training seeds. Each candidate is evaluated on **selection-only** worlds, with a 14-dimensional movement/topology/spatial descriptor. k-nearest-neighbour novelty and discretized behaviour bins form a conservative archive. Two survivors per niche are compared on fresh held-out worlds: reward-only selection, and 55% within-niche reward + 45% novelty selection. A larger descriptor distance does not imply higher human artistic preference. Any diminished reward is reported, not concealed. This is an *archive over candidate evaluations*, not a claim of population-wide evolutionary convergence.

An observation-dependence ablation replays each selected policy with its initial action-score prior held fixed while preserving the current host-legal action menu. Reports include choice divergence on **identical encountered states**, probability total variation, the post-goal-switch divergence and a separate frozen-prior rollout reward. It asks whether changing observations influence the policy's actions; it does **not** prove the model learnt goal reasoning, and the static prior maps previously unseen actions to the initial mean score.

The original held-out trajectories and SVGs for BOTH selection policies, original per-candidate weight JSON, seed-hash manifest and failed/weak candidates are retained inside the same immutable Actions evidence bundle. The P0 baselines (nonstop random, persistent and privileged synthetic oracle) are rerun on new P1 holdouts. Nothing here uses native Workbench, performs real-game training, imports Godot code or changes game content.

P2 priority: train a true sub-1B SLM/adapter as a student on legal-choice trajectories; compare frozen-input/teacher-distilled samples with train/selection/heldout controls. Keep P1 MLP evidence separate from claims about actual language-model capabilities.


## P3 — factorial rarity/morphology causal dissection

Run P0 first, then:

    python experiments/driv3r/training/population/population_p3.py --p0-output /tmp/population-619 --output /tmp/population-627 --seed 627 --members 3 --train-episodes 18 --selection-episodes 8 --holdout-episodes 12 --long-horizon 60

P3 tests four **matched, real-Adam-trained** policy populations: reward-only,
**rarity-only**, **morphology-only**, and the **combined** P2-inspired teacher-target intervention.
Each matched member starts from the identical mutated P0 champion and shares the
same seed streams, optimizer and selection rule. The only controlled difference
is the teacher-score intervention. Each arm has a separate training-only
state/action visitation archive; no held-out information is allowed into it.
Selection-only worlds decide champions before untouched held-out worlds are
opened. All offspring weights, optimizer steps, training bins, selection
scores, 30-step reward/diversity, original 60-step synthetic gallery SVG/JSON,
and long-horizon measures are retained. Two fixed independent cohorts
(627 and 628) run in GitHub Actions. The 60-step stress is **synthetic only**:
its longer clock does not correspond to a qualified native PAINT operation.

**Interpretation warnings:** the six niche utilities are handcrafted proxies,
rarity is a coarse visit-count bonus and morphology uses hand-coded styles;
novelty/diversity does not imply human preference or useful CyberSand art.
Four-arm teacher targets isolate direct interventions, but different generated
on-policy state distributions may mediate their effects. Within-arm archive
updates are sequential across members. These models are tiny neural MLPs,
**not** pretrained language models or learned SLM adapters. Native game code
and content remain completely untouched. Draft/HOLD only.

## P4 — online trajectory-level feedback (synthetic only)

`population_p4.py` tests whether an **observed completed-episode phenotype**
is more useful than sham feedback for learning distinctive CyberSand-style
game decisions. Four matched arms: reward-only supervised Adam; shuffled
sham feedback; observed trajectory-novelty feedback; previously successful
hand-authored morphology teacher. All run with fixed P0 parents, the same
member seeds, training-world splits, finite host-legal action menus and
reward-based selection. Report 30- and 60-decision held-out dispersion,
nonstop-random and P0 controls, original gallery traces, model lineage and
replayable exact-source results. Entire feedback archive is TRAIN-only.
Novelty is derived from observed trajectories, although its numeric
descriptor is still engineered; post-episode updates use supervised targets
and are **not** presented as policy-gradient RL. Sham controls random
feedback, not all possible sources of model stochasticity.

P4 does **not** establish learned aesthetic preference, generalization to
real CyberSand, native Workbench execution or any pretrained SLM adaptation.
Do not claim that trajectory-level novelty works unless it beats its sham
control on independent held-out game trajectories. The workflow retains all
weights and negative cases, without changing game code/content.

## P5 — matched-feedback offline causal ablation, selection archives, and recovery stress

`population_p5.py` tests whether observed novelty *corresponding to a training
episode* contributes more than the same feedback amplitudes assigned to different
episodes. Each mutated P0-source policy generates a **frozen TRAIN-only rollout
dataset** with finite host-legal game actions; four matched MLP copies receive
identical features, teacher targets, optimizer state and episode order. Arms:
reward only, measured trajectory novelty, **permuted exactly-matched sham
feedback**, and an explicit hand-designed morphology target. Signals use
centered novelty ranks, preserving the complete signed feedback multiset in
measured and sham arms. This controls signal magnitude and sign frequency
within each member; it does not imply live on-policy feedback or causal effects
in the native game.

Champion selection is performed only on SELECTION worlds; a small MAP-Elites
archive retains best-reward candidates within predeclared coarse topology bins.
We also evaluate reward-ranked equally sized subsets, retaining single-cell
archive failures explicitly. Final frozen models run on untouched 30/48-step
holdouts and a deterministic *synthetic* adversarial world with a one-cell
east cul-de-sac requiring RETRACE, added protected material and a mid-run goal
switch. Original weights, split hashes, exact feedback signals, selection
bins, held-out comparisons, SVGs and trajectory JSON are retained.

Reproduction after P0, for two independent cohorts:

    python experiments/driv3r/training/population/population_p5.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p5_632 --seed 632 --members 3 --train-episodes 12 --selection-episodes 6 --holdout-episodes 6 --horizon 48
    python experiments/driv3r/training/population/population_p5.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p5_633 --seed 633 --members 3 --train-episodes 12 --selection-episodes 6 --holdout-episodes 6 --horizon 48

This is **real Adam training on frozen trajectories**, not reinforcement
learning, not a pretrained SLM/adapter, not native CyberSand painting, and
not human-rated aesthetic improvement. A synthetic novelty scalar should
only be credited if it beats the amplitude-matched sham on untouched worlds.
No CyberSand game code or production content changes; Draft/HOLD only.


## P6 — genuine multigeneration MAP-Elites parent selection, with matched random/reward controls

Run P0 first, then `population_p6.py --p0-output /tmp/population-evidence
--output /tmp/population-evidence/p6_634 --seed 634`. The dedicated workflow
executes independent seed cohorts 634/635.

Three real-evolution treatments each train 3 successive generations × 4 offspring
in all six niches, starting from the exact same P0 model. At generation zero,
mutation seeds, source parent, train-world seeds and optimizer targets are
identical across arms. At subsequent generations the parent-selection rule
differs: **reward PBT** breeds the best selection-reward ancestor;
**MAP-Elites** samples the reward-best occupant uniformly across
predeclared *selection-only* topology archive cells; **random-parent control**
samples from the previous offspring generation independently of reward/bins.
All three receive genuine on-policy/teacher-augmented Adam updates (no post-hoc
weight-labeling). Their common curriculum alternates ordinary synthetic game
worlds with an adversarial cul-de-sac/protected-material scenario.

Track every mutated source checkpoint, exact initial and final model hashes,
actual Adam-update count, training decision count, archive occupancy by
generation, reward-ranked and archive-selected portfolio IDs, original synthetic
SVG/JSON trajectories, untouched held-out 48-decision pairwise diversity,
challenge retrace rates, held-out phenotype bins, and explicit weaknesses.
Each portfolio contains the same number of candidate policies, padded from
selection reward if too few archive cells emerge; report that undercoverage
rather than claiming bins that do not exist. Distinct train, repeated-selection,
and holdout seed streams have SHA-256 manifests. All trained checkpoints
remain in immutable Actions artifacts, never Git history.

**Important:** This is *not* emergent proof of learned aesthetics or a native
CyberSand training run. Phenotype bins are handcrafted and selection data is
reused across generations, potentially overfitting. MAP-Elites is evolutionary
archive pressure; it is NOT self-play, policy-gradient RL or pretrained SLM
training. After generation zero the ancestral choices change the training
state distributions, so results are a comparative algorithm experiment, not
a tightly controlled single-causal-variable ablation. Host masking guarantees
legal actions and says nothing about learned safety.

P7 candidate: distinguish archive-parent coverage effects from stochastic
ancestry with independent cohorts and evolve real learned trajectory embeddings;
continue the separately owned student-SLM distillation lane. Never touch
CyberSand game code/content; PR remains Draft/HOLD.


## P7 — fitted spatial/action embeddings versus raw MAP, random ancestry and reward PBT

`population_p7.py` adds a **four-arm, three-generation, genuine Adam-trained**
breeding comparison across the same six game-facing synthetic-policy niches.
Controls are reward-winner PBT, raw P6 topology MAP-Elites, uniformly random
previous-generation ancestry and **learned-embedding MAP-Elites**. Every
generation trains four mutated offspring per niche/arm and alternates standard
and dead-end/protected-region synthetic game curricula. The two independent
cohorts (636/637) train 576 small neural candidates. They do not mutate the
actual CyberSand game or claim native integration.

The **train-only fitted encoder** measures each policy's synthetic 5×5 spatial
paint footprint plus its six action frequencies, learns three PCA directions
from independent P0/mutated-probe TRAIN worlds, and freezes quartile cell
boundaries before any selection/holdout decisions. This is an unsupervised
*linear* learned projection of deliberately chosen descriptors, **not** a
neural reward model, learned aesthetics, pretrained SLM, reinforcement learning
or self-play. The raw-MAP control uses the P6 fixed handcrafted topology bins.
All arms use identical generation-zero weights, mutation seeds and
training-world seeds, though differing ancestry subsequently changes
on-policy state visitation.

Selection/parent choice and archive construction use exclusively selection
worlds; the fitted encoder uses an independent train-only seed stream.
Held-out 48-step reward/diversity and adversarial recovery are computed only
after choosing each final four-member portfolio. Archive cardinality is
reported as coverage, never as proof of visual interestingness. P0, training,
encoder, selection and holdout streams are checked for disjointness.
Preserve weights, exact lineage, hashes, PCA parameters/quantiles, occupied
cells, original trajectory SVG/JSON and negative results in Actions artifacts.

After P0:

    python experiments/driv3r/training/population/population_p7.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p7_636 --seed 636 --generations 3 --offspring 4 --train-episodes 8 --selection-episodes 5 --holdout-episodes 6 --horizon 48 --portfolio 4
    python experiments/driv3r/training/population/population_p7.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p7_637 --seed 637 --generations 3 --offspring 4 --train-episodes 8 --selection-episodes 5 --holdout-episodes 6 --horizon 48 --portfolio 4

**Interpretation gate:** compare learned MAP against raw MAP *and* random
ancestry on identical held-out horizons, not just against reward-PBT.
Higher selection archive occupancy is not necessarily higher held-out
diversity. Retain negative results. Next work can attempt richer learned
temporal encoders or actual student-SLM training, but do not call this PCA
experiment neural aesthetic learning.

All implementation remains in `experiments/driv3r/training/population/**`
and its research-only workflow. No game-side file may be modified. Draft/HOLD;
do not merge or promote Ready.

## P8 — train-only neural temporal-order encoder versus four ancestry controls

`population_p8.py` trains a three-dimensional *nonlinear neural temporal
convolutional* representation with NumPy Adam. It sees only ordered sequences
of six host-legal game actions, including STOP and RETRACE, over a fixed
30-decision window. TRAIN-only challenge/ordinary synthetic game trajectories
are contrasted against **action-count-preserving reversed and shuffled
transcripts** in a three-class supervised classification task. All temporal
weights, optimizer steps, training losses, original/permuted classification
accuracy, embedding variances, and quartile boundaries are frozen before
opening selection or held-out worlds. Encoder VALIDATION worlds are independent
of its TRAIN worlds and all P8 policy splits. The fitted P7 PCA projection is
also fit on distinct TRAIN-only probe worlds.

Five comparators start from matched P0 niche champions and identical
generation-zero mutation/training seeds: reward-best PBT, handcrafted topology
MAP, TRAIN-only PCA MAP, the new frozen neural temporal MAP, and random
previous-generation ancestry. They train via real teacher-augmented Adam on
mixed standard/dead-end/protected/goal-switch synthetic worlds. Parent choice
changes after generation zero. Final portfolios have matched size and are
frozen before holdout evaluation. Preserve all lineage, mutated source SHA,
final model SHA, per-candidate training steps, selection archive cardinality,
untouched held-out 48-decision diversity/reward, challenge RETRACE counts,
frozen-encoder-space diversity, and original synthetic SVG/JSON trajectories.

The bounded independent cohorts 638/639 each use two generations × three
offspring × five arms × six niches = 180 policies, 360 combined, plus 160
encoder Adam updates per cohort. The two local source-matched campaigns
produced 51,468 **policy** optimizer steps. Mean held-out trajectory diversity:
reward-PBT 0.481132; raw MAP 0.487688; linear PCA MAP 0.517058;
neural temporal MAP **0.525994**; random-parent 0.498964. However, the
neural MAP result differed substantially between cohorts, and encoder
independent validation accuracy was only 0.3741 and 0.4407 against a
one-in-three nominal chance baseline. These are **exploratory synthetic
results only**; modest average benefits are not evidence of reliable
emergent aesthetics, temporal reasoning, or general game capability.

Reproduce after P0 (NumPy 2.3.5; BLAS one thread):

    OPENBLAS_NUM_THREADS=1 python experiments/driv3r/training/population/population_p8.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p8_638 --seed 638 --generations 2 --offspring 3 --train-episodes 5 --selection-episodes 4 --holdout-episodes 4 --horizon 48 --portfolio 3 --encoder-steps 160 --probe-count 6 --probe-worlds 4
    OPENBLAS_NUM_THREADS=1 python experiments/driv3r/training/population/population_p8.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p8_639 --seed 639 --generations 2 --offspring 3 --train-episodes 5 --selection-episodes 4 --holdout-episodes 4 --horizon 48 --portfolio 3 --encoder-steps 160 --probe-count 6 --probe-worlds 4

**Limitations and guardrail:** a three-class ordering discriminator can exploit
trivial action-transition statistics; it is neither an aesthetic judge nor a
pretrained language model. Higher archive occupancy does not establish more
useful game behaviour. Synthetic actions are host-masked, not internally
safe. This is not reinforcement learning, self-play, native CyberSand, or
SLM-student training. Retain bad and quirky models equally. Everything stays
inside the population research lane and its isolated workflow; Draft/HOLD,
no merge or game-side modifications.

## P9 — adaptive scenario-adversary curriculum tournament (research only)

`population_p9.py` implements **external synthetic game-environment adversaries**.
Six niche-specialised small neural action-selection policies compete against
deterministically proposed obstacle, protected-material and changing-goal scenarios.
Variants include dead ends, slalom obstacles, protected choke points, blocked goal
approaches and goal reversals. Cards 6/7 are **heldout geometry templates** never
offered during training. All actual game actions remain host-mask legal
(N/E/S/W/RETRACE/STOP); no Godot/native/Workbench code or game assets are touched.

Three independent cohorts (640, 641, 642), each with two generations and three offspring per niche per arm; every mutated offspring
is actually trained with teacher-augmented on-policy-state Adam. Three controlled
scenario-assignment treatments are compared: uniformly sampled candidate card;
maximal frozen-parent hardness; and maximal *current-policy* hardness. Every arm
probes the same offered TRAIN-world cards under both its live and frozen model,
with the same number of training-world episodes and matched probe budgets.
Deterministic hardness is a normalized difference from a privileged handcrafted
utility oracle, plus early-STOP penalty. The experiment is *not* reinforcement
learning or true multi-agent self-play, and the environment opponent itself is
an adaptive scenario selector, not a trained neural generator.

Each of the 324 policies in cohorts 640/641/642 (108 per cohort) saves its actual
weights, parent/seed/optimizer lineage, per-episode candidate difficulty vector,
selected challenge card, and selection metrics. After selection/parent evolution
is frozen, untouched-world evaluations compare all three arms on familiar
and never-trained scenario geometries; selected-policy SVG/JSON traces capture
the latter. Numeric outputs and negative findings must be compared directly
against both random and fixed-hardness controls. Same reference world generator
underlies novel cards, and reward remains synthetic/handcrafted, so this is
not evidence of native game generalisation or human preference.

Reproduce from a P0 folder produced by the population baseline:

```sh
python experiments/driv3r/training/population/population_p9.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p9_640 --seed 640 --generations 2 --offspring 3 --train-episodes 7 --selection-episodes 4 --holdout-episodes 6 --horizon 40 --portfolio 2 --proposals 3
python experiments/driv3r/training/population/population_p9.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p9_641 --seed 641 --generations 2 --offspring 3 --train-episodes 7 --selection-episodes 4 --holdout-episodes 6 --horizon 40 --portfolio 2 --proposals 3
python experiments/driv3r/training/population/population_p9.py --p0-output /tmp/population-evidence --output /tmp/population-evidence/p9_642 --seed 642 --generations 2 --offspring 3 --train-episodes 7 --selection-episodes 4 --holdout-episodes 6 --horizon 40 --portfolio 2 --proposals 3
```

**Draft/HOLD, no merge, no production or CyberSand game/content changes.**

## P10 — CPU-only action-token sequence student distillation (synthetic research)

`population_p10.py` trains **from-scratch** recurrent action-token students on
finite host-legal action menus and teaches six niches simultaneously from the
original P0 optimizer-trained MLP teachers plus fixed-seed mutated teacher
variants. This is a real gradient-trained small recurrent sequence **student**,
not an adapted/pretrained text SLM. Unlike P0–P9's direct candidate-feature
MLPs, the token-GRU student consumes the most recent eight executed action
tokens as well as 20-feature legal candidate observations and niche labels.
Its output is constrained to the host-provided legal actions at every step.

Matched controls share identical initial parameter tensors and student
training data/order: (a) state-only candidate scorer; (b) actual past-action
GRU; (c) GRU with training-only action-history order randomly permuted, while
preserving each history's token multiset and padding. P10 compares two
holdouts: original geometry and a staggered-wall/goal-shift geometry family
not present in training. Original P0 teacher rollout is also evaluated on the
same held-out worlds. Models, distinct seed cohort splits, optimizer updates,
cross-entropy, per-niche synthetic utility, legal-action traces, SVG galleries,
and SHA-256 manifests are retained, including weak/negative controls.

Requires optional **CPU PyTorch**; the original NumPy-only population
training workflow remains independent. Exact-source P10 is qualified by
`.github/workflows/driv3r-population-students.yml`, which installs pinned
CPU PyTorch, rebuilds P0 teachers, and runs cohorts 643/644/645. Manual
reproduction after original P0 training:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/driv3r/training/population/population_p10.py --p0-output /tmp/population-p0 --output /tmp/population-p10-643 --seed 643 --train-worlds 24 --validation-worlds 6 --heldout-worlds 8 --epochs 6 --teacher-variants 3 --width 40 --horizon 40

Interpretation: P10 is **teacher imitation under handcrafted synthetic rewards**,
not RL, a pretrained language-model adapter, learned aesthetics, source-faithful
material physics, or native CyberSand gameplay. Host masking ensures menu
legality only, not a learned safety guarantee. Source-level independence of
TRAIN, validation and held-out world seeds is tested; dynamic student
trajectories are never fed back into optimization. Do not promote small
cross-cohort reward differences to genuine game or artistic superiority.
**Draft/HOLD; no game-side edits, no merge or Ready promotion.**

## P12 — Withheld cue grammar and long-gap memory (local research checkpoint)

P12 extends P11's short, supervised order probe into **six-branch-action paired histories followed by 9–17 common legal distractor actions**. Counterfactual worlds have identical painted/retrace geometry, host-menu observations and final eight history tokens, but demand opposite E/W decisions. The model is trained on *only next-action cross entropy* (no explicit auxiliary order classification). Its GRU sees 32 action tokens; matched last-eight, shuffled-order and state-only negative controls cannot distinguish the pairs.

Three cue grammars use the same E/W/N survey vocabulary; the N-first grammar is held out, as is the longest distractor chain. Two local scratch-trained cohorts (649 and 650, 4 models each, 4,608 AdamW updates combined) reached **100%** in-distribution full-history accuracy and **50%** for all information-blind controls. On the unseen N-first grammar, cohort 649 reached **100%**, but cohort 650 fell to **50%**. This inconsistent zero-shot grammar transfer is an important negative finding: do not claim reliable compositional generalization. All host-issued actions were legal by masking.

See `P12_REPORT.md` for cohort-level measurements and limitations. Run `python -m unittest discover -s experiments/driv3r/training/population -p test_population_p12.py -v` and the opt-in `.github/workflows/driv3r-population-long-memory.yml`. These are artificial single-decision synthetic memory probes, **not** native CyberSand, full missions, or pretrained SLMs. Keep Draft/HOLD; no game integration and no merge.
