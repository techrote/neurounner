# TINY GAME POLICY TRAINER — synthetic M0 (Draft/HOLD)

This is a runnable **model-side research harness only** under the scheduler's
`experiments/driv3r/training/tiny_policy/**` ownership. It **does not execute
CyberSand or use native material readbacks**, does not alter gameplay/provider
source, and is **not SmolLM2, d1 or Sprocket training**. Results are not an
admitted original native TRAIN/FINAL corpus or native 128-turn qualification.

## Contract and authority

- 96×96 source-generated world; host supplies the finite, **cardinal Stage-A
  direct-drive action vocabulary** `UP DOWN LEFT RIGHT STOP`. Source reference:
  CyberSand `research/570-driv3r-stage-a@6067ea31adb8f3dfc76867676fc504078f5e9526`,
  `experiments/driv3r/contracts.py`; this **does not claim compatibility with
  the separate W0 SPROCKET action vocabulary**.
- Host rejects moves intersecting blocked/protected/out-of-bounds cells. Explicit
  STOP always legal; 96 actions maximum, no native writes, no inference-service
  permissions. Each action has original per-step before/after state hashes,
  feature hash, exact legal string array and receipt; replay reconstructs
  world bytes and fails closed if any field differs.
- Observation: three 7×7 binary local planes (walls, previous paint,
  protection), heading one-hot, family one-hot, bounded progress/geometry scalars:
  **162 float32** values. Unmodelled game dynamics and material species are
  deliberately excluded. Painting is host-owned and remains a single-cell line.
- Three synthetic families: Greebles, Pipework, Architecture. The teachers are
  small, imperfect deterministic/seeded geometric controllers, **not stronger
  pretrained model responses or native evidence**. An architectural label does
  not guarantee an architectural drawing. Distinct styles and failures remain
  in the gallery.

## Actual performed baseline

```bash
PYTHONPATH=. python -m unittest experiments.driv3r.training.tiny_policy.test_core -v
PYTHONPATH=. python -m experiments.driv3r.training.tiny_policy.train --out /absolute/new/evidence-directory --epochs 7
```

Requires Python 3.11+, NumPy, Pillow and CPU PyTorch. Output directory must not
exist. `train.py` runs actual AdamW updates with masked categorical cross
entropy on three competing *from-scratch small heads*: linear, two-layer MLP
and spatial CNN. It writes **real checkpoint files, split/hash identities,
source pixel PNGs, JSON rollouts, one contact sheet, and exact legal receipts**
into the selected **external** evidence directory, never Git history.

The original M0 execution used 50 **TRAIN** seeds / 10 **DEV** seeds / eight
**EVAL** seeds, with three family-labelled rows per world and 96 teacher steps
per row. Seed-group isolation is exact at the complete-world level. EVAL is
repeatedly inspected research evaluation, **not sealed FINAL**. Original W0
training and other native experiments are disjoint evidence domains even if
some descriptions resemble each other.

Every learned model gets exactly the same generated TRAIN/DEV/EVAL examples,
legal masks, optimizer family and epoch budget. Compared closed-loop against
teacher, random-legal and deterministic wall-following on identical obstacle
seeds/action budgets. Always show held-out teacher agreement **and** unique
cells, retraces, traversed-edge branch nodes, same-world family footprint IoU,
STOP and original drawings. Training agreement is not artistic usefulness.

## Historical M0 pretrained-model boundary (superseded by M1/M2)

This M0 **has not loaded or trained any 20M–300M pretrained LLM**. It is an
actual tiny-head training and host-mask smoke, not a fabricated SmolLM2 result.
The next, separate CPU-only cohort should compare source-pinned
`HuggingFaceTB/SmolLM2-135M-Instruct`, `facebook/opt-125m` and
`EleutherAI/pythia-160m` using identical observations, partitions and legal
options, with measured weight-load, cold/warm latency and real forward receipts.
Stage first as frozen-backbone decision-head training and include an untrained
backbone/head control; require separate evidence before LoRA and any inference
promotion. **Pin exact model revisions, tokenizer identities and licenses at
acquisition**; no dynamic `main` or vague model-family claims. Bound runner
resource/time budgets and retain every failed acquisition/training attempt.
Only write model weights, optimizer checkpoints and galleries into immutable
GitHub Actions artifacts when a hosted cohort is deliberately launched; this
source-only PR contains none of those binary outputs.

## Known negative findings / failure examples

A high imitation accuracy can coexist with very weak spatial composition.
The wall baseline is deliberately family-invariant. The linear head commonly
collapses to similar paths across theme labels. A 96-cell limit ensures finite
execution but does not mean early STOP/generalization is proven. Typical
synthetic worm output is a short scribble or a straight wall-following path,
not recognizable artwork or a production-quality CyberSand game policy.

## M1/M2 outcomes and M3 curriculum (16 November 2026)

M1 independently loaded frozen, revision-pinned pretrained SmolLM2-135M,
OPT-125M and Pythia-160M, trained actual linear action heads, and retained
checkpoints and rollouts as Actions artifacts. M2 applied those immutable heads
to four new world-seed groups with protected-gate/cul-de-sac obstacles.
**All three failed promotion** on matched original-host synthetic game paths;
better teacher agreement did not imply better artwork or new-world coverage.
See PR #634's M1/M2 evidence comments and immutable hosted artifacts. OPT's
license is research-only/non-commercial: do not promote to a product.

M3 adds `m3_curriculum.py` and `test_m3_curriculum.py`, a **new CPU-only
head training experiment**, not additional pretrained/LoRA work. It uses the
same imported `core.py` 162-feature, 96×96, host-legal action contract, four
small synthetic obstacle templates, real protected STOP-only fixtures, and
world-seed-disjoint TRAIN/DEV/EVAL groups. Each training seed contributes
oracle, random and wall behaviour states labelled by the same deliberately
imperfect 5×5-local recovery oracle. This reduces pure on-oracle covariate
shift. Its 64-action evaluation window is **not** native qualification.

Run from repo root with PyTorch, NumPy and Pillow installed:

```bash
python -m unittest experiments.driv3r.training.tiny_policy.test_m3_curriculum -v
python -m experiments.driv3r.training.tiny_policy.m3_curriculum --epochs 5 --out /new/nonexistent/path
```

The *local* CPU experiment used PyTorch 2.10.0+cpu (not a GitHub-hosted
qualification). It produced 23,160 TRAIN / 4,632 DEV / 4,632 EVAL
state-action examples, 230 real AdamW updates each for a linear, MLP and
convolutional head, and 24 matched original-pixel 64-action trajectories
per policy. Closed-loop unique-cell means were **57.79 linear / 56.12
MLP / 56.50 conv**, versus **27.08 random / 31.88 wall**; the
recovery oracle averaged 59.83. All applied actions were host-legal and
no trained head STOPped early in these 24 trials. **Counterexample: the
untrained conv baseline averaged 57.38 cells from mostly straight paths,
with zero branch nodes, showing that coverage alone is inadequate.**
M3 is synthetic, not a 20–300M backbone result, and cannot be compared
numerically with M2's different 48-action worlds. There is no game-side
change, original native corpus, admitted FINAL, production rollout, or
M3 GitHub Actions artifact from this local run. Code is retained to run
a later independently qualified hosted trial; checkpoints remain outside Git.
