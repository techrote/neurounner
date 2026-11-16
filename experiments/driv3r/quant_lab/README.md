# QL-001 — Extreme quantisation of a trained game-steering worm (research only)

**Draft/HOLD. No CyberSand runtime modifications, merge, Ready promotion or default-provider changes.**

This deliberately small first experiment trains a 27→64→64→6 neural action policy using original synthetic CyberSand-inspired world states (empty, rock, protected, visited), six host-legal commands (STRAIGHT, LEFT, RIGHT, RETRACE, BRANCH, STOP), three family prompts, and teacher+exploration trajectories. It is not an SLM, native Godot run, WORM-001 W0 byte-faithful replay, GGUF quantiser, architectural painter or gameplay integration. Its purpose is to get **real training, real low-bit packed weights, real repair, and original contrasting drawings** into the research pipeline before experimenting with larger SLMs.

## Exact runnable experiment

- python experiments/driv3r/quant_lab/quant_worm.py --self-test
- python experiments/driv3r/quant_lab/quant_worm.py --out quant-lab-out --steps 450 --repair-steps 100

Requires CPU PyTorch 2.5.1, numpy 2.1.3, Pillow 11.1.0 and Python 3.11. CI uses pinned dependencies and uploads all checkpoints, optimizer state, packed NPZ binaries, SHA-256 inventory, raw model-score/host-choice receipts, results.json and **144 original synthetic 25x25 PNGs** to an immutable Actions artifact (not git). Train seeds 1100–1163; held-out 8100–8123; nine visual cases: seed 8101/8107/8113 × three style prompts. Every model and control receives the same starting worlds, **identical host random-draw stream for a fixed world seed across families**, and 64-turn cap; subsequent states diverge, so trajectories are **not paired same-state counterfactuals**. Model-free RANDOM/WALL/FIDGET are family-label negative controls. An earlier provisional artifact used differing family RNG streams and must NOT be used to claim family-conditioned morphology; compare only new source-qualified receipts.

## Real precision ladder

An independently retained untuned FP32 and untuned Q1 root distinguish learned effects from quant-only random initialization. A single trained root is quantised per output row to Q8, signed uniform Q4/Q3/Q2, sign+scale Q1, mixed Q1/Q4/Q4 and Q4/Q1/Q4; also Q2 after 30% coherent first-layer channel pruning, and Q2 after 100 real straight-through QAT-style AdamW repair updates. Biases and per-row scales stay FP32. All low-bit arrays are **densely bit-packed across byte boundaries**, roundtrip-verified, and then **dequantised for FP32 Torch matmul** at inference. Record actual compressed files and theoretical packed payload separately. There are **no actual IQ3/IQ2/IQ1 GGUF codecs here**, and **no claim of low-bit execution acceleration or resident RSS reduction**. Do not relabel scalar Q1 as llama.cpp IQ1_S.

## Measurement and interpretation

Report unmasked top-action legality separately from host-applied legality (the host always masks), STOP/BRANCH/RETRACE histograms, 64-turn output and unique painted cells, held-out six-action teacher accuracy, time per warm CPU model-forward and raw RSS **for the shared process only**. CRITICAL: teacher supervision is class-imbalanced (STRAIGHT is 822/1477 held-out states; BRANCH 163, STOP 13 and RETRACE **not represented**). Record per-supported-class recall, unsupported classes and majority-teacher accuracy, not just overall accuracy. The initial source-qualified run revealed zero BRANCH teacher recall in many *trained* variants despite high overall accuracy; the untuned brain instead over-selected BRANCH. This is a creativity-relevant negative, **not a passing multi-action skill claim**. Unequal STOP durations can make painted-cell counts incomparable as aesthetic scores: do not turn the toy into an A* benchmark. Random control can STOP early, wall can produce bland large coverage, fidget can make repeated marks; all remain legitimate creative references. No human aesthetic ratings are manufactured. Source SHA, raw actions, model score vectors and PNG hashes are available for audit.

## Local preflight before Actions, 16 Nov 2026 (NOT exact-HEAD CI)

A local execution using the same algorithm with 450 AdamW updates and 100 repair updates gave FP32 73.8%, Q4 72.9%, Q2 71.0%, Q1 69.7%, and Q2+QAT repair 75.7% held-out teacher accuracy (1,477 synthetic state observations). FP32 and Q1 packed weight+bias payloads were 25,368 and 1,848 bytes respectively. Q2 permitted longer walks (30.1 average steps vs FP32 21.8 over nine worlds) but this is **not** evidence of creative preference or low-bit CPU speedup. CPU timings around 0.02 ms reflect a tiny model with FP32 execution; per-variant differences are noise-scale. Only the exact-source Actions artifact qualifies the published branch. The local preflight was executed outside GitHub Actions and is not called native, SLM or FINAL evidence.

## Next substantive step

Use the packed result to select *contrasting* personalities by inspecting real PNGs and raw decisions, not maximum coverage alone. Then move to actual llama.cpp GGUF Q4/IQ3/IQ2/IQ1 format-specific inference only after verifying matching architecture, supported loader/kernels, pinned immutable model bytes, RAM and same-runner throughput. Any larger student/trained LoRA repair must come from a known checkpoint with matching licensing and source hash. Do not accidentally use this pilot's tiny CPU timings as evidence about a 70M/350M/2B SLM. Preserve the host legal-action authority and all existing WORM/DRIV3R branches.
