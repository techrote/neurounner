# D-1A long-horizon direct-drive research — offline foundation

**Status:** independent **development-only** preparatory fixtures and oracle policies, not native or model qualification. #570; [owner D roadmap](https://github.com/techrote/cybersand/issues/570#issuecomment-6877408001). The five-record [D-COMMON-001 candidate](https://github.com/techrote/cybersand/pull/594) is **not accepted/frozen**. This lane intentionally does not import or implement that candidate as an approved protocol.

## What is implemented

- Deterministic versioned 36-decision fixtures: 3 distinct irregular selected-wall profiles and 4 material-Z avoidance geometries in **development**, plus 2 different wall profiles and 2 different routing topologies in **heldout**. Split is by hardcoded structural family, *not merely RNG seed*.
- Source hash for every complete fixture, manifest catalog hash, selected-wall identity held separately from distractor walls; all blocked/protected/Z coordinates enforce 4-connected movement.
- Strong conventional controls: selected-wall distance-band-penalised A* contour tracker versus shortest legal A* for A→B routing; seeded random legal policy receiving the same candidate set, including STOP.
- Optional separately labelled cardinal 1/2/4/8-cell action-macro arm. Every traversed cell is checked; an endpoint can never conceal collision.
- Strictly sequential **synthetic** observation → action → new observation loop (36 attempted decisions maximum), with original option-list hashes, immutable fixture hash, ordered events, deterministic path hash, honest STOP and first-pass failures.
- Geometry metrics: exact selected-wall distance, fraction within 6–10-cell band, mean/max deviation, wall-route progress, legal endpoint arrival, route inefficiency versus shortest A*, revisits/retraces, false STOP, collisions and material-Z/protected violations. Native counters are explicitly zero / not performed.
- Narrow staging view `observation_stub`: candidate IDs, geometry, budgets, simulated positions and explicit NULL native fences/receipts. It is deliberately **not** a D-COMMON or native observation.
- **Synthetic evidence integrity:** `paired.verify_episode` regenerates each legal candidate menu and full observation, confirms a sequential policy- and seed-reproducible choice, checks swept cells, STOP and arrival, all intermediate simulated positions and path digests, and independently recomputes every final metric. It explicitly **rejects claimed native receipts**, malformed/altered sequence, unbounded macros, forged choices, edited offered candidates, metric tampering, and held-out behavioral evaluation. JSON/JSONL round trips are supported.
- **Separately audited synthetic effect divergence:** the exact first differing selected ID is not asserted to be a consequential result. Replayed arms now distinguish first differing simulated cursor position, accumulated swept-path digest, episode length and shared cell prefix. Full post-divergence Worlds and native material are **not** compared; those require separately authorized sequential receipts. New `trajectory.py` has 9 source-hash-matched Python tests executed locally (9/9 PASS, including tuple/JSON roundtrip equivalence); the combined legacy lane suite remains pending independent qualification.

- **Matched development-only comparison:** `paired.compare_matched` requires the same fixture hash and identical macro/action menu; checks identical positions and legal candidate-set hashes **only until the first differing chosen action**. It records the first consequential decision-divergence index, avoids false post-divergence parity claims and retains full metric records of both strong geometric and seeded random-legal arms. No representation A1/A2/A3 model experiments are dispatched before M1.

## Commands

From repository root, with Python 3.11+ and no extra dependencies:

```sh
python -m unittest experiments.driv3r.stage_d.direct.test_offline experiments.driv3r.stage_d.direct.test_paired -v
python -m experiments.driv3r.stage_d.direct.offline --manifest
python -m experiments.driv3r.stage_d.direct.offline --policy astar
python -m experiments.driv3r.stage_d.direct.offline --policy random_legal --seed 17 --output /tmp/d1a-random.jsonl
python -m experiments.driv3r.stage_d.direct.offline --policy contour_astar --macro
```

To audit a source-pinned development pair programmatically:

```python
from experiments.driv3r.stage_d.direct import offline, paired
dev = next(f for f in offline.fixtures() if f.fixture_id == "route_gate_v1")
comparison = paired.run_development_pair(dev, seed=13)
assert comparison["native_qualified"] is False
```

**Qualification status:** The newly added `test_paired.py` methods are authored, but were not executed on a Python checkout in this run. Prior Stage A workflow success does **not** qualify these tests; that workflow discovers `experiments/driv3r/tests/**`, not this exclusive Stage D-1A lane. An independently implemented, offline JavaScript A*/contour cross-check of the seven development geometries returned wall steps **33/29/35** and A→B legal shortest paths **35/25/33/31** (all within 36); that is not equivalent to executing the Python suite or a native run. Run both unittest modules on a suitable Python environment and retain original failure output before any M1/integration claim.

Default runner executes **development fixtures only**. Held-out fixtures are manifested for source custody and cheap structural checks; no held-out model outcome is inspected. All JSONL data are offline simulation events and may not be described as native receipts, paintings or visual renderings.

## Explicit further gates

1. **M1 owner freeze:** pin exact accepted shared schema/source SHA and adapt `observation_stub` without rewriting native contracts outside this lane.
2. **Separate native execution authority:** integrate only through the centrally owned validated native bridge with fresh actual World fences, Stop, protection, swept-extent checking, material readback, native receipts, replay and cleanup. Never infer native success from these geometry fixtures.
3. **Preregister then dispatch:** matched controller/model Q8/random episodes on exact frozen sources; resident CPU cost/time; first-pass failures and complete sequential trajectories retained; holdout model outcomes kept sealed until the approved evaluation.
4. **36-decision limit:** subsequent 72/144-step stress cases need a reviewed D-specific adapter to avoid silently expanding prior strict Stage A/B caps. Real training, paid API/GPU, defaults and merges require separate authorization.

Do not edit `d_common/**`, other D streams, #568 or historical Stage A/B/C.
