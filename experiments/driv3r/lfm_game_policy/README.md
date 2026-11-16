# LFM-GAME-POLICY-350 — isolated CPU research (Draft/HOLD)

This is a **distinct** model-training lane from WORM-001's full multimodal LFM2.5-VL-450M PR #575. This experiment uses the actual text-only [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) at exact model revision `9921cd4d82c0e3986887aa4b376f9446729ef8a3`, under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-350M/blob/main/LICENSE). Model attribution: Liquid AI. No checkpoint may be substituted or represented as the VL-450M model.

## Experiment GP350-001

Run a single hosted **4-vCPU CPU** LoRA rank-8 training wave, 24 real AdamW updates/48 backward passes. Only verified full-attention `q_proj` and `v_proj` modules train; frozen base, embeddings and output head remain unchanged. Source pin, train-only synthetic W0 worlds, counterfactual matched-geometry family targets, real one-token constrained scoring, original model/W0 receipt trace, artifacts and adapter numerics are preserved.

A deliberately synthetic teacher makes local motif choices based on obstacle/visited status, history, heading, available legal actions, tick and family. TRAIN worlds 3101–3112; DEV worlds 9103, 9119, 9137; live on-policy W0 test seeds 9503, 9521. Models never execute native material writes; only W0 host may apply an action. Comparison: untuned full LFM350, same-model trained LoRA, random legal, deterministic wall and fidget procedural W0. The historical d1-2B experiments are a **separate context control** until matched replay is possible; no claim of a new d1 model decision.

Outcomes include actual optimizer progress, model exact model/revision/file hashes, model and adapter target inventory, full adapter checkpoints, frozen-sample integrity, original test PNG/JSON/scores, replayed action/PNG equivalence, actual STOP and illegal raw score rates, time and RSS. Results may fail artistically or technically. Raw-scored legal selection is **not** raw EOS-completed text. Negative and partial checkpoints stay intact as failure artifacts. No native acceptance, provider promotion or merge.

Execution: [workflow](../../../.github/workflows/driv3r-lfm-game-policy.yml) on the one immutable `gp350-001-request.json` source update. No automatic recurring optimizer is scheduled. Never run a second concurrent LFM350 optimizer. Offline first gate:

```sh
python3 experiments/driv3r/lfm_game_policy/lfm350.py --self-test --w0 /path/to/pinned-w0/worm.py
```

**Ownership boundary:** Only this research folder and its dedicated workflow. No edits to `godot/**`, `native/**`, game maps/assets, `experiments/agent_painter/**`, other DRIV3R lanes, production/default providers, or sealed FINAL. All PRs remain Draft/HOLD.
