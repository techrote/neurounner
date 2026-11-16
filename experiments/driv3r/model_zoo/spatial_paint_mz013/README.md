# MZ-013 — explicit 2-D PAINT, protected material and recurrent-memory onboarding

**Research only. Draft/HOLD. Do not merge, promote Ready or change CyberSand game code.**

This continues MZ-011/MZ-012's hidden-material instruction experiment beyond
automatic fork outcomes. A new, deterministic **2-D cell-grid host** executes
LEFT, RIGHT, UP, DOWN, PAINT, RETRACE, WAIT and STOP under a visible host-legal
mask. The agent traverses three spatially separated painting stations; each has
two opposite mineral cells. The early beacon and later inversion jointly select
the safe mineral, then disappear before the agent reaches the stations.
Choosing the protected cell causes an actual spatial detour and collision;
RETRACE physically returns to the previous cell. PAINT changes the cell from
mineral to painted material. All moves and paintings are recorded in original
trajectories. Training includes a deliberately wrong turn at the middle station
to teach recovery. Stopping early or failing to complete all three paintings is
a failed episode.

This is **not** a full free-roaming 2-D planner: the main route is a corridor
with procedural station spacing, and there is no native material simulation,
water, terrain solver or Workbench. The OOD probe changes station spacings to
5–8 from the training interval 3–5. The 5x5 egocentric terrain patch includes
obstacles and already-painted cells, and the model sees its action menu,
relative position, previous action and painting progress, but **not** the
historical beacon/inversion after the opening four briefing actions.

## New concrete model

A **randomly initialized 20M-class causal transformer**: four standard
TransformerEncoder blocks of width 656, eight attention heads, feed-forward
width 2624 and learned positional embeddings. Every attention matrix is
strictly causal. This is not a pretrained SLM or a language-model checkpoint.
It is a proper 20M-range transformer architecture trained directly on a common
host-legal decision curriculum.

A previous exact-HEAD 220-update pilot on commit 2a045a554641c9a593c7a6fbe7f72f26ed56959d is preserved as Actions run 43921685156 / artifact 14253737899. It failed to train the 20M transformer into competent hidden-cue branching (46.43% on TEST), while the diagonal recurrence reached 85.71%. This second 700-update run tests the finite-budget undertraining hypothesis without silently replacing that negative result.

Compare against a 2-layer input-gated diagonal recurrent controller,
a two-layer GRU and a current-state MLP. All four receive identical seeded
minibatches and **700 actual AdamW updates** on CPU. The curriculum has 96
train-seed families, disjoint DEV/TEST families of 12 each, a separately
shifted-spacing TEST, explicit injected-error trajectories, 8 actions and
the same weighted per-action loss. Checkpoints and error cases are preserved.

Measure held-out all-action, branch and PAINT classification, raw illegal
proposals, closed-loop painted-cell count, collision-free completion,
protected collisions, real action traces, ablated recent-context diagonal
performance, cached-versus-prefix parity, CPU latency and peak RSS.

A perfect teacher-forced PAINT score is not evidence of successful navigation:
the actual controller must choose its own actions through terminal STOP.
Likewise, matching updates/examples does not match parameter count, FLOPs,
inference software optimization or representation expressivity.

## Reproduce

From repository root:

~~~bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v experiments.driv3r.model_zoo.spatial_paint_mz013.test_spatial
python -m experiments.driv3r.model_zoo.spatial_paint_mz013.run --out mz013-evidence \
  --train-seeds 96 --eval-seeds 12 --rollout-seeds 3 --updates 700 --batch 12
~~~

The dedicated workflow source-fences every changed path against exact base
commit 7e6c3e416588549a2bfe6d83c5d75e948d64f7f3, reloads the entire
checkpoint, independently reproduces held-out TEST and shifted TEST metrics,
checks state-cached parity and SVG validity, and publishes checkpoint, source
SHA, dependencies, loss curves, metrics and trajectory gallery to an immutable
per-run GitHub Actions artifact (14-day retention).

**No edits** to native/**, godot/**, Workbench, game assets, scenes, materials,
simulation, maps or production provider settings are authorized.
