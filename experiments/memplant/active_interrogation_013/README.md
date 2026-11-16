# DG013 — active neural donor interrogation and stateful game-policy trajectories

**Status:** Draft/HOLD; stacked on DG012 #245. **No CyberSand game/code/content changes.**

## Research question

DG012 transferred a neural donor's privately scrambled behaviour using fixed, randomly selected paired calibration worlds. Can **budgeted, sequential donor queries** extract more useful learned game laws, and do those inherited rules support useful behaviour after 36–128 consecutive decisions rather than single-action tests?

DG013 trains an original compact probe-ranking network using independently trained **narrow/wide neural specialists**. The ranking target is the disagreement of real specialist outputs across game-law families; its input is only ordinary canonical probe-world features. The new trained ranker is independently validated before transfer to **sealed gated-architecture specialists**.

The experiment compares four calibration protocols with matched query budgets of **8, 16 and 24 black-box donor calls per specialist**:
1. Random fixed probe selection.
2. Deterministic feature-coverage / regularized information-design selection.
3. Newly trained, source-agnostic probe ranking.
4. **Sequential adaptive interrogation:** begin with ranker probes; after each real donor response, update a posterior over *previously trained narrow/wide surrogate functions* and select the next world by posterior-weighted disagreement.

The adaptive method is constrained to **one gated-donor query per chosen world**. It never sees unselected gated-donor outputs. Surrogate-bank inference over all candidate worlds is permitted and **not counted as gated-donor calls**; the compute distinction is reported, not concealed. Calibration world features are canonical and known. Gated specialists' private feature maps are hidden. The learned ranker is not trained against the gated specialists or sealed query worlds.

## Sealed evaluation

- Rebuild DG012's **18 independently trained neural donors and three receivers** at source-pinned seed, rather than relying on an arbitrary online model.
- Retain **withheld (+,+,+) law** and independent mixed-law test worlds; 900 of each per seed.
- Use **three independently generated pools of 64 candidate probe worlds** for each budget and strategy. Only the selected 8/16/24 gated-donor outputs may be queried.
- Measure exact host-legal action agreement with the authored synthetic expert, not real-world gameplay success. Include the trained DG012 law-blind receiver and source-calibration baselines.
- Extend to **72 decision** *stateful* synthetic 13×13 grid rollouts, with blocked cells, wetness, protected paint cells, spatial goal, visitation memory, paint-once, and legal retrace. Record whole action traces, goal reach, unique cells, painting, retraces, repeated visits and illegal actions. Independently evaluate the **authored single-step expert** and a **BFS shortest-path goal planner** on exactly the same initial worlds to distinguish a weak expert/reward signal from unreachable goals. Every strategy sees the same initial world seeds.
- Preserve parent donor and receiver weights, new trained probe ranker, original action traces, metrics, source SHA and all per-file SHA-256 identities in immutable Actions artifacts.

The grid is an **external synthetic research harness**, not native CyberSand. The original DG012 receiver was trained for isolated action imitation; an anticipated failure is that it loops or paints too little during long rollouts. That negative result matters and must remain visible.

## Local pilot (pre-CI, torch 2.10.0 CPU)

Two independent selector seeds used verified, already-trained DG012 CI archives rather than retraining their parent models. This *local pilot is not exact DG013-head CI evidence*; CI is the qualification source.

Across three independent 64-world probe pools per strategy and 600 mixed-law test decisions per seed:

| Donor query budget, mixed-law exact action accuracy | Seed 26011622 | Seed 26011623 |
|---|---:|---:|
| Random, 8 | 70.39% | 79.67% |
| Learned ranker, 8 | 79.89% | 84.67% |
| Adaptive, 8 | **80.06%** | **83.56%** |
| Random, 16 | 85.33% | 91.00% |
| Adaptive, 16 | 85.39% | 88.78% |
| Random, 24 | 87.11% | 90.61% |
| Adaptive, 24 | 86.61% | 90.78% |

**Interpretation:** promising low-budget calibration improvement in both seeds, but **no reliable advantage at budgets 16–24**. The learned ranker correlation against held-out supervised informativeness was only **0.476/0.480**. This is not evidence of universal adaptive superiority.

The first stateful 72-decision runs were poor: mean unique visits ~9–13 cells, loop/revisit counts ~48–54, and goal reach 1–5 of 18 worlds depending on parent seed. The **authored one-step expert reaches the goal at the same low rates**; the independent BFS planner reaches 18/18 in both seed families. This identifies a weakness in the synthetic teacher and reward shaping, *not* evidence that the inherited neural receiver uniquely fails. No calibration mode rescues long-horizon planning. These are preliminary results in a version-equivalent local harness, not native gameplay.

## Running / CI

```bash
python3 -m pip install 'torch==2.5.1' 'numpy>=1.26,<3'
python3 -m experiments.memplant.active_interrogation_013.experiment --selftest
python3 -m experiments.memplant.active_interrogation_013.experiment \
  --seed 26011620 --donor-steps 220 --receiver-steps 900 \
  --train-cases 5200 --sealed-cases 900 --ranker-steps 450 \
  --trajectory-steps 72 --worlds 18 --out /tmp/dg013
```

The dedicated Actions matrix uses source seeds 26011620 and 26011621; the new ranker seeds are +2. The parent re-train step retains the exact DG012 architecture and seals the gated donor from selector training. No external paid API or GPU is used.

## Scientific boundaries

Neither a new joint physical law nor a new observation ontology is discovered. This remains **supervised synthetic transfer of behaviour from learned neural specialists**. Source models are small MLPs, not native transformer KV caches; sequential adaptation chooses calibration examples and is not adaptation of the receiver's weights. For game utility, **trajectory quality and recovery matter more than single-action imitation**. A strong navigation planner is included as a reachability control, not a neural inheritance baseline. If the inherited policy loops at long horizons, record that as a research limitation rather than an integration candidate. No CyberSand game edits, Workbench integration, merge or Ready promotion are authorised.
