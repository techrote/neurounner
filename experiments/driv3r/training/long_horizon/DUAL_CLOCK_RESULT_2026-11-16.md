# DUAL-CLOCK-001 — matched recovery memory ablation (16 November 2026)

**Status:** Completed locally, not committed or Actions-qualified. Synthetic DEVELOPMENT-only negative result. No game-side writes, held-out evaluation, native run or paid inference.

Two identically initialised 36-parameter action rankers compare executed-movement memory against executed + currently-live-path history after BACKTRACK/RELOCATE_START. Both receive identical TRAIN states/labels, host-legal menus, seeds and 90 Adam updates over 36→72→128 curricula. The teacher uses live path state, creating an expected agreement advantage for the dual-clock arm; that advantage must be tested against actual sequential progression and early STOP. Task variations: wave, jag and wave→jag at step 64. Evaluation at 128 and 160 decisions includes deterministic forced legal actions at decisions 14, 19, 32, 49, 68, 83, 109 when admissible. No hard intervention is simulated if not host-legal. STOP remains an ordinary candidate.

## Real local matched DEVELOPMENT evidence

| Seed | Arm | Forced 128 mean ink | Forced 128 early STOP | Mean post-recovery teacher agreement | Forced 160 mean ink | Forced 160 early STOP |
|---|---|---:|---:|---:|---:|---:|
| 582 | Executed-only | 108.00 | 3/12 | 50.71% | 132.00 | 3/12 |
| 582 | Dual-clock | 75.25 | 11/12 | 76.31% | 77.50 | 11/12 |
| 583 | Executed-only | 124.00 | 0/12 | 42.19% | 150.75 | 3/12 |
| 583 | Dual-clock | 87.00 | 11/12 | 72.27% | 87.50 | 12/12 |

**Finding:** Dual-clock improves teacher agreement in post-recovery windows, but often stops early and yields far less drawing. This is a negative result for useful long-horizon control. Training-target agreement is not a substitute for progress, recovery quality or artistic line character. Two seeds are exploratory; do not claim broad model superiority.

- Seed 582: 96 original sequential DEVELOPMENT episodes and 11,205 host-legal decisions; 22.516 seconds wall; peak 129,060 KiB RSS.
- Seed 583: 96 episodes and 11,829 decisions; 24.459 seconds wall; peak 130,180 KiB RSS.
- Two genuine 36-weight checkpoints per seed, with exactly 90 Adam updates per checkpoint.
- Six new tests + 25 existing tests PASS locally (31/31). Independent source replay of all 192 episodes PASS, artifact manifest verification PASS. Action distributions, per-decision probabilities, latency and full material/ink trajectories in original evidence.
- The earlier, non-switch seed 581 pilot is not pooled. One first-pass test raised IndexError by assuming an untrained policy survived 64 decisions; fixed to tamper with step 0 and independently assert the step-64 switch boundary. Both the error and correction are preserved in this report.
- Held-out structural worlds never inspected. No d1-2B, LFM2.5-350M or Sprocket-500M weights. No native receipts. The 64-logical-move native ceiling is unchanged.

## Source and checkpoint identity

`world.py`: Git blob c7e33b7801c3a9d6c1f00f11f466732b312ddc92.
`policy.py`: Git blob c61573bdd4dc603ef38594b5b764fec1ff2142ab.
`train.py`: Git blob 0e6da7de3ac52279049f0c7482d6c4754042e034.

Checkpoint SHA-256:

- 582 executed: 9576024d4d6e00560880cf3c2d97c2e7465d771ce2afae5595245c2f5f56325a
- 582 dual: b3273e5154812ac7a6e5c7912ef19146d1acf6bb25e9405a54953a373bc426ad
- 583 executed: 17ba4b22b33802ea39888594e5c2c0f68546ee83f9357161daffbbfdb4871728
- 583 dual: 6d1b90ec7f4a4ddaf37ed23de0f747e40b241cc1383dcdb574860276bc696304

Run: `python -m experiments.driv3r.training.long_horizon.dual_clock_lab --output /tmp/dual-clock-seed582 --seed 582 --worlds 1 --updates 30` then `--verify /tmp/dual-clock-seed582`. All outputs are created outside Git. GitHub publication and complete exact-HEAD Action qualification remain pending. Do not present local pilot hashes as Actions-derived artifacts.
