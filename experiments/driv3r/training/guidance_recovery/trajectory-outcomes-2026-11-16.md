# C14 — counterfactual trajectory-outcome guidance training

16 November 2026 · [PR #639](https://github.com/techrote/cybersand/pull/639) · research-only · **Draft/HOLD**

## Hypothesis and experiment

C13's globally informed one-step oracle improved supervised loss but degraded long-horizon reversals. For C14, train from outcomes of **actual synthetic-host-applied trajectories** instead. Freeze C12 and learn only a bounded BACKTRACK-logit residual (29→12→1). With no active host advice, non-BACKTRACK actions and all uncued scores remain bit-identical to C12.

For each of **672 matched initial menus** from hairpin/staircase families, force one legal BACKTRACK and one legal forward action. Roll each out twice for up to 22 steps through `World.apply` with matched exogenous RNG, frozen C12, material legality checks and host advice expiry. That's **2,688 real optimizer-data suffix rollouts**, across no/weak/strong scalar guidance (0, 0.35, 1), weak/strong host advice strengths (0.35, 1) and BACKTRACK/CONTINUE signs. Label using *actual future* goal arrival, terminal distance improvement, decisions and ink expenditure; not a full-world shortest-path oracle. Positive vs negative preference margins: **318 vs 323**, 31 near ties.

Train a confidence-bounded correction using weighted binary cross-entropy plus predecessor anchoring. **220 CPU AdamW updates**, two complete independently repeated runs yielding byte-identical checkpoint, metrics and trajectories. The Git-committed [compact C14 optimizer](trajectory_outcomes_core.py) reproduces the checkpoint byte-identically from the seven exact prerequisite safetensors hashes. [C14 regression suite](test_trajectory_outcomes.py): five new host/legality checks; **115/115 locally available tests pass**.

**C14 checkpoint SHA-256:** `34153145f1de3f1ff7d9557e1d11113dde3f7b780241cd58940da71c3e9b0e26`.

## Held-out hook/zigzag stochastic trajectories

All numbers are successful full goal arrivals, each out of 144 episodes per model and condition. Training families are excluded from this table.

| Condition | 32 decisions: C12 → C14 | Separate seed, 48 decisions: C12 → C14 |
| --- | ---: | ---: |
| Reverse; valid BACKTRACK cue | **115 → 124** | **104 → 121** |
| Reverse; misleading CONTINUE cue | 121 → **115 (regression)** | 113 → **105 (regression)** |
| Ahead; misleading BACKTRACK cue | 129 → **117 (regression)** | 130 → **125 (regression)** |
| Ahead; valid CONTINUE cue | 114 → 117 | 119 → 121 |
| Reverse; no cue | 52 → 52 | 49 → 49 |
| Ahead; no cue | 130 → 130 | 126 → 126 |

Each full comparison has **3,456 source-seed/RNG-matched episodes** including trained and withheld families, two personalities, three scalar guidance arms and three advice modes. The second uses an independent seed and a longer horizon. No-cue trajectories are unchanged, as intended.

Signed left/right-turn personality gap, first panel: BACKTRACK cue 0.060709 → 0.056262, CONTINUE cue 0.176283 → 0.195308, uncued 0.242424 → 0.242424. Second panel: BACKTRACK 0.070160 → 0.075842, CONTINUE 0.204288 → 0.214444, uncued 0.204311 → 0.204311. This is **not** proof of style retention under active guidance.

## Verdict and custody

**Do not promote C14 over C12.** Trajectory-outcome optimization helps valid warned reversals on two unseen panels, but over-backtracks in viable passages and can accept misleading advice. Improvement on one recovery metric is not enough.

Synthetic host: zero host-rejected actions, unchanged protected-cell hashes, no new unmasked illegal top-one choices. This is **not native CyberSand acceptance**. The full C5/C10-inclusive exact-HEAD regression suite cannot be claimed from the locally available archive; Actions artifact custody for model binaries is still absent. Checkpoint and full trajectories are preserved outside Git in a SHA-manifest-verified research archive.

Next: train a cue-reliability discriminator from *observable* geometry/history rather than increasing backtracking strength. Use false-advice completion as a hard holdout gate. Do not change CyberSand native, Godot, materials, Workbench, production inference, maps, assets or provider defaults. No paid compute. Do not merge or mark Ready.
