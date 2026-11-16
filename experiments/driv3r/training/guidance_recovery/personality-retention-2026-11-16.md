# C13 — on-policy recovery and long-horizon personality (16 November 2026)

**Research-only outcome: REJECT C13 as a policy upgrade; retain frozen C12.** All experiments use an external synthetic CyberSand-like host. No game, Workbench, materials, runtime or production changes.

## Trained model
A frozen C12 policy plus a new **29→12→1** residual, permitted to adjust only the host-legal BACKTRACK logit by at most ±2 while a signed host advice event is active. No advice or expired/cancelled advice leaves every logit bit-identical; all INK/LIFT/STOP logits are invariant even with advice. The host retains material/action authority.

Collected **9,987 on-policy, host-executed training observations** across hairpin/staircase, both drawing personalities, no/weak/strong scalar guidance, and genuine/misleading BACKTRACK and CONTINUE instructions. A full-map BFS and cursor-history teacher supplied labels, *without passing global maps into the 29-feature policy*. The teacher prefers clean cursor-stack pops over equal-distance forward steps that would merely grow history. Distinct worlds can alias under this observation contract.

Ran **948 actual CPU AdamW updates** (weighted action loss plus residual-size penalty), training loss **0.167838 → 0.116853**; 5,593 positive BACKTRACK labels.

**New rejected checkpoint SHA-256:** `40491d69f8f66f339fe6c03799323f2c31e6b4bd98fa78848c631fdf7987056c`. Frozen C12 SHA-256: `180075041fdc61b8ec0e22216780b4a7c76169829dffd43734a9d46c19be80cf`.

## Independent 32-decision held-out synthetic trajectories

Evaluation: **2,304 episodes**, C12 vs C13; source-matched initial states, exogenous RNG, three hint strengths, both personalities, no/BACKTRACK/CONTINUE host cues, all four topology families. Hook/zigzag were excluded from training. Each row below has 96 trajectories *per model*.

| Withheld geometry goal + cue | Frozen C12 goals | Trained C13 goals |
|---|---:|---:|
| Reverse / no cue | 39 | 39 |
| Reverse / valid BACKTRACK | **73** | 56 |
| Reverse / misleading CONTINUE | **83** | 63 |
| Forward / no cue | 89 | 89 |
| Forward / misleading BACKTRACK | 87 | **90** |
| Forward / valid CONTINUE | 78 | **86** |

**Failure:** False-warning forward completion improves (+3/+8), but genuine recovery/regret under conflicting CONTINUE severely deteriorates (−17/−20). Thus a lower oracle BCE is **not** evidence of better sustained cursor-history recovery. C13 should not supersede C12.

## Personality measured along actual paths

No-guidance final ink hashes and actions were identical to C12 by construction. Under active advice, aggregate signed left/right gap and paired opposite-personality turn differences sometimes disagree:

- Withheld reverse / BACKTRACK: aggregate gap 0.07196→0.14140, but paired absolute bias contrast 0.22902→0.17973 among pairs where *both* turned.
- Withheld reverse / misleading CONTINUE: aggregate gap 0.53412→0.47018, paired contrast 0.77576→0.60606 (only 11/48 pairs had nonzero turns for both worms).
- Withheld forward / CONTINUE: aggregate gap 0.10563→0.08544; paired contrast 0.19433→0.08085.

These are limited turn-bias proxies, **not proof of preserved drawing personality or aesthetics**. Improved goal arrival can mask collapsing expressive choices.

## Qualification, reproducibility and guardrails

Two independent complete optimizer/evaluator runs reproduced **byte-identical** model, metrics, training receipts and full trajectory files. Safety guards: 48 absent-cue + 48 expired-cue observations exact; 1,728 non-BACKTRACK logits exact; zero synthetic host-rejected actions, zero illegal raw top-1 selections, protected-material hashes unchanged, painting monotonic. BACKTRACK never undoes ink. **106/106 locally available Python tests passed** (not a complete exact-PR-HEAD suite: C5/C10 tests were absent locally).

Source/reference archive, checkpoint binaries, all SHA receipts, complete trajectories and HTML gallery were generated *outside* Git history. Immutable GitHub Actions checkpoint custody is still missing because the separately owned workflow path was not modified. No model-native Workbench test, real material Undo, paid service or production dispatch occurred.

**Decision:** Keep C12 as comparator. Preserve C13 as a negative experiment; pursue a host-reviewed optional deeper-clearance/history observation or a directly multi-step, outcome-grounded training objective before another stronger correction gate. No production interface change is authorized by this research.
