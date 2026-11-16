# WORM-001 SPROCKET W4: original partial PNG audit and bounded trained-only recovery

15 November 2026 UTC. Original W4 evaluation source 9dd2b7c620189b3b1f70d99c9150e4691ee15304; original GitHub-hosted [run #43715212047](https://github.com/techrote/cybersand/actions/runs/43715212047) / job 132815143384 **FAILED (exit 124)**, not SUCCESS. The unchanged original trained checkpoint was produced by [96-step run #43710874563](https://github.com/techrote/cybersand/actions/runs/43710874563), source c14795efe978dadeaa9e0da60f6273e18ab17b4d, checkpoint checkpoint-000096. This is WORM synthetic action learning, not whole-Art training or native Godot application.

## Immutable original evidence independently inspected

The original [artifact #14162494880](https://github.com/techrote/cybersand/actions/runs/43715212047/artifacts/14162494880) was freshly downloaded. Original ZIP SHA-256: 1b19a8cae48bd1396b547ba3181f46449a5f56efbce3c1755dea9f8158c8e614. **65/65 ZIP members** passed CRC and **64/64 listed internal SHA256SUMS entries** matched. No original files or receipts were rewritten or rebranded.

At 24 steps per rollout with three families and two seeds, the failed job completed **18/18 matched model-free controls** (random, wall, fidget), **6/6 untuned root** rollouts (each one invalid raw action, 0 EOS5, host STOP), and **5/6 trained W4** rollouts before the 1400-second timeout. Trained Architecture/seed223 is missing.

| Source-faithful controller | Completed | Unique cells per trajectory | Host-invalid raw choices |
| --- | ---: | --- | ---: |
| Random legal | 6/6 | 19 or 21 | 0 |
| Deterministic wall | 6/6 | 25 | 0 |
| Fidget | 6/6 | 21 or 22 | 0 |
| Untuned pinned root | 6/6 | 1, then host Stop | 6 |
| **W4 96-step trained greedy** | **5/6** | **4** | **0** |

Original trained model: **120/120 EOS5, 120/120 legal, 120/120 LEFT**. Its path covers a **2x2-cell box**, max Manhattan radius **2**, and yields only **two distinct original PNG byte hashes among the five captured cases**; within each seed, family labels do not change the image. No host fallback, script repair or teacher action was counted as model recovery.

I inspected the original PNGs. The trained image is a tiny turn-square around the start; matched random/fidget controls draw larger line segments. This is **artistic quality FAIL / no useful checkpoint freeze**, despite genuine optimization and valid action coding. Five completed trained trajectories cannot be reported as six.

## Distinct recovery instead of retrospective relabelling

The old evaluator placed six expensive, non-EOS untuned root generations *before* the trained arm. Its 1400-second timeout interrupted the final trained trajectory. New lane-owned worm_rollout_v4_recovery.py imports the exact original W4 choose/roll, verifies the immutable full-state source/model/numerics/W0 contract, and runs **only unchanged trained W4 first-pass action generation** paired to random/wall/fidget controls on identical seeds. The original slow untuned negative first passes remain separately preserved in the failed artifact. Recovery stores actual raw tokens/EOS, raw legality versus host-applied actions, W0 events, original PNG bytes, extent/radius, 24-action prefix and partial/failure receipts. It performs **zero optimizer updates**.

The opt-in, separately leased [recovery run #43721058235](https://github.com/techrote/cybersand/actions/runs/43721058235) was dispatched on ordinary GitHub-hosted CPU from exact producer source 7953f88bf9d62c116de36537c03f611ba8655517. Immutable original training artifact, 96-step checkpoint, pinned W0 and source lease verification passed. **Evaluation completion, PNGs and success remain pending.** The recovery is neither a new training run nor native Workbench paint nor a model freeze.

## Next

On recovery completion independently verify original artifacts, complete sixth W4 learned trajectory, control comparisons and all raw first passes. Then investigate a genuinely different host-seeded, model-conditioned action-selection mechanism or training objective, rather than repeating the same greedy W4 recipe. Leave #576 Draft/HOLD, original step96 full-Art negative result, #568 historical gates, support/common/peer lanes, sealed FINAL, native host authority and production providers untouched.
