# MZ-011 — diagonal gated-state controller versus proven recurrent GRU

**Research only · Draft/HOLD · no CyberSand gameplay/Workbench modifications.**

## Research question

MZ-009's 177k two-layer GRU used disappearing material instructions to
complete synthetic material-fork levels without protected-material collisions.
MZ-010b's 20.67M wide GRU did not recover that skill under an exposure-matched
curriculum. MZ-011 tests a **different recurrent state transition** rather than
another GRU width increase.

The candidate is a ~141k-parameter **two-layer input-gated diagonal-state
recurrence**, with independent per-channel latent retention. It has constant
stored-state size and can issue actions by feeding just a new observation and
its cached recurrent state, rather than recomputing a full history window.
The input-only gates make it meaningfully different from a GRU's state-dependent
gates, while nonlinear stacking permits combining delayed clues. It is *not*
a pretrained transformer, Mamba, S4 or full selective-scan implementation.

## Exact inherited environment and controls

Stacked on MZ-009 Draft PR #654, source-pinned at
`3a4eded62900aad3df91fd9b2a1ae28bbb4a0c0e`, importing the
existing game environment **read-only**. The scripted host contains transient
material beacons, a disappearing hazard inversion, two visible material
candidates at each of three successive forks, protected collisions, RETRACE
recovery, STOP, legal-action masks and action-driven transitions.

Train candidate, 177,605-param GRU and ~4,933-param current-observation MLP
for 800 genuine optimizer updates each on the **same 48 sampled episodes per
step** with the same weighted action loss and learning rate. All
cue/inversion/injected-recovery variants belonging to a seed family remain
in a single split. Equal examples/optimizer steps is not equal FLOPs or
parameter count.

Report *held-out fork action accuracy*, raw unmasked illegality, actual
collision-free closed-loop completion separately from eventual completion,
protected-material collisions, cue-flip/deletion causality, historical
context truncation, cached-state versus full-prefix action parity, unbatched
CPU inference times, training step times and peak process RSS.

Training code saves a source- and upstream-pinned state dictionary, SHA-256,
per-step training loss samples, complete metrics, original SVG trajectory
gallery, and independently reproducible TEST predictions in an immutable
GitHub Actions artifact. No checkpoint is committed to Git.

## Local commands from this branch

```bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v experiments.driv3r.model_zoo.material_forks_mz009.test_world experiments.driv3r.model_zoo.diag_state_mz011.test_diag
python -m experiments.driv3r.model_zoo.diag_state_mz011.run --out mz011-evidence --train-seeds 160 --eval-seeds 24 --rollout-seeds 8 --updates 800 --batch 48
```

Do **not** promote on teacher-forced accuracy or eventual completion alone.
A viable follow-on must demonstrate material-cue causality and substantial
zero-collision completion advantages against the current-state controller.
The model remains an isolated proof-of-mechanism below the zoo's intended
20M–1B long-term target band; scale only if this architecture earns it.

**Hard ownership fence:** only
`experiments/driv3r/model_zoo/diag_state_mz011/**` and the single dedicated
workflow may change. Never touch native simulation, Godot, scenes, Workbench,
materials, maps, assets, gameplay content or default providers. No paid GPU
or API. PR Draft/HOLD; never merge or mark Ready without owner approval.
