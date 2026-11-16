# MZ-012 — rotary phase-gated recurrent game-memory policy

**Research only · Draft/HOLD · no CyberSand gameplay, Workbench or native edits.**

## Falsifiable purpose

MZ-011's 141k diagonal-state recurrence and MZ-009's 177k GRU achieved perfect
completion on fixed-length procedural material forks. Those tests could conceal
brittle dependence on fixed cue-to-fork timing. MZ-012 trains a distinct,
two-layer **phase/rotation-gated recurrent controller**, compares it against
both proven architectures and a current-state MLP, and probes variable horizon
and shifted-long corridors. It is a purpose-built mechanism probe, **not an
implementation or claimed benchmark of pretrained LRU/S4/Mamba**.

## Data and honest boundaries

Stacked on exact source `bcd759b8f8a351998d61829b3f527c965ca041d4`
(#656, itself stacked on #654). All inherited procedural host and architecture
code is imported **read-only**. Own code is confined to `phase_state_mz012/**`
and one dedicated GitHub Actions workflow.

Training corridors: first length 8–13, subsequent lengths 6–10.
A separate **OOD diagnostic** evaluates first lengths 14–20 and subsequent
lengths 10–14 with independent seed families. Each episode uses hidden early
material/beacon and inversion cues, three material forks, host-legal LEFT/RIGHT,
protected-material collision on wrong choice, automatic material painting on
safe choice, RETRACE and STOP. This is **not a brush-stroke painting study**:
the host increments a synthetic painted-cell outcome; explicit paint controls
and 2D terrain physics are future work.

Train 4 controllers for **500 real optimizer updates each, identical sampled
32-example minibatches** with action weighting: phase recurrence, diagonal
recurrence, GRU and reactive MLP. Equal examples do **not** mean matched
FLOPs/parameter count. Compare held-out ID and long-OOD fork correctness,
illegal raw proposals, collision-free versus eventual completion, material
collisions, beacon flipping, cached-versus-prefix parity, 1/8-observation
context ablations, CPU p50/p95 and peak RSS. The actual host executes each
predicted legal action, not the teacher action, during closed-loop evaluation.

Keep failed and successful results. A victory on easier ID but a collapse on
shifted OOD is a **failure of horizon generalization**, not a promotion.
No claims of native CyberSand game integration.

## Reproduction

```bash
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v experiments.driv3r.model_zoo.material_forks_mz009.test_world experiments.driv3r.model_zoo.diag_state_mz011.test_diag experiments.driv3r.model_zoo.phase_state_mz012.test_phase
python -m experiments.driv3r.model_zoo.phase_state_mz012.run --out mz012-evidence --train-seeds 128 --eval-seeds 24 --rollout-seeds 8 --updates 500 --batch 32
```

Checkpoint state dictionaries, exact source and parent SHA, scores, training
samples, raw protected-material trajectory gallery, Python dependency freeze
and SHA-256 are preserved as an immutable per-commit Actions artifact. Do not
commit model checkpoints or install paid external GPU services.

**Owner gate:** Draft/HOLD, never merge, enable auto-merge or mark Ready.
No files outside this experiment and its dedicated workflow may change.
