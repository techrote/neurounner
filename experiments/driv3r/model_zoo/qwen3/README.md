# MZ-005 — Qwen3-0.6B instruction-pretrained game-policy probe

**Status: Draft/HOLD; research only. No CyberSand game code, content or production provider changes.**

New candidate: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), full pretrained immutable source revision `c1899de289a04d12100db370d81485cdf75e47ca` (Apache-2.0 model card). Unlike prior base Pythia-70M, this is an instruction-capable Qwen3 architecture with a chat template and non-thinking mode. No custom remote code is trusted.

**Training**: actual CPU pretrained hidden-state inference, then 140 real AdamW optimizer steps on a learned seven-action head. The Qwen3 backbone is **frozen**. A separately trained 114-feature structured policy receives the same 140 steps on the same TRAIN world seeds. The constant-majority controller is retained. Do not describe head training as end-to-end or LoRA/backbone training.

**Controlled source-faithful synthetic task**: same deterministic game-world TRAIN/DEV/TEST seed families and goal-swapped teacher as MZ-003/004. Worlds contain walls, protected material, drawn cells, host legal-action lists, cursor, facing, goals and sequential actions. Raw argmax legality is recorded independently of host masking. Output includes paired goal-change metrics, action metrics, model load and CPU p50/p95, RSS, original 4x4 trajectory gallery, and a checksum-protected trained head. Qwen's chat-template input means cross-model accuracy is not an identical input-encoding comparison.

Run using the dedicated Actions workflow `.github/workflows/driv3r-model-zoo-qwen3.yml`. The run executes source tests before downloading the immutable 0.6B pretrained model and retains research outputs as GitHub Actions artifacts for 14 days. No trained results may be claimed before a successful exact-head Actions run.

**Next decision**: promote only if it materially beats majority and shows observation-conditioned paired-goal responses, ideally competitive with or above the structured policy. Otherwise keep as a negative result and move to a new architecture or selective adapter study.

**Boundaries**: No edits to `godot/**`, `native/**`, Workbench, maps, scenes, materials, physics, gameplay logic, assets or production provider settings. This is external/synthetic GAME-policy research; never call its trajectories native Godot PAINT.
