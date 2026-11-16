# MZ-006 — ModernBERT-base (149M) real encoder-layer training

**Draft/HOLD; model research only, no CyberSand game edits.**

Hypothesis: a **bidirectional pretrained 149M encoder**, with an actually optimized pretrained final block, may learn counterfactual goal-conditioned actions better than frozen decoder/SSM/RWKV embeddings. Source [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) pinned to immutable Hugging Face revision `8949b909ec900327062f0ebf497f51aef5e6f0c8` (Apache 2.0); `trust_remote_code=False`.

Unchanged disjoint TRAIN/DEV/TEST source-generated 11x11 material-obstacle-protected-cell worlds, seven legal host actions, scripted teacher and paired goal swaps. Same 14/4/4 world seeds and 9 steps as MZ-003/004/005. Three fair control arms: (a) ModernBERT **frozen** pretrained encoder with trained classification head, (b) ModernBERT **last pretrained encoder block + classifier** with **real gradients and optimizer updates**, (c) 114-feature trainable structured controller; plus constant-majority action. Controls have equal combined update counts; tuned last-block arm uses minibatches while controls use full batches (important sample-budget asymmetry). No native PAINT training or claims.

Record actual model download SHA, encoder parameter/trainable count, DEV and held-out TEST accuracy/macro-F1, paired goal-swap correctness and switch rate, raw illegal proposals before host masking, CPU forward time/RSS, distinct 4-world trajectory gallery, optimizer loss and SHA-256 of independently retained updated pretrained weights. Checkpoint contains ONLY tuned last encoder block/classifier and smaller baseline heads, not the pretrained 149M backbone.

**Promotion gate:** beat the cheap structured controller at least on meaningful goal-swap accuracy or demonstrate new useful sequential behaviour, with controlled source hash and validated checkpoint. Otherwise retain as a negative or partial signal and change the training representation/approach. Do not interpret matching teacher labels as intelligent native gameplay.

Run with dedicated `.github/workflows/driv3r-model-zoo-modernbert.yml`. No paid APIs/GPUs, no merges, no production/game implementation files, no changes to simulation, Godot, Workbench, maps, materials, assets or gameplay content.
