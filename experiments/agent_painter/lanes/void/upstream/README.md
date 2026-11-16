---
library_name: transformers
pipeline_tag: text-generation
tags:
- qwen3
- byte-level
- pretraining
- symbolic
license: apache-2.0
datasets:
- karpathy/climbmix-400b-shuffle
- openbmb/Ultra-FineWeb-L3
- dotlabs/rewrite
- nvidia/OpenMathInstruct-2
language:
- en
---

### Updates
> - 🔹 best under 90m on sub-100m on [open slm leaderboard!](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard#dotlabs-void.1)
> - 🔸 void.2 paused to train better models focusing on stronger performance and usability.
> - ▫️ step 900,000 updated with better performance, lr decayed to zero on ~10b additional bytes.

# 👋 Meet void.1

State of the art, small language model pretrained from scratch on a diverse set of high-quality texts and an internal symbolic kernel. This is the first step into a series of models designed for fine-grained understanding of abstract/symbolic reasoning on text while being grounded on english.

<video src="https://huggingface.co/dotlabs/void.1/resolve/main/teaser.mp4" autoplay loop width="100%"></video>

## Comparison

| Model                                                             | Params | HellaSwag | PIQA   | ARC-E | ARC-C | ArithMark-3 | Intelligence Index |
| ----------------------------------------------------------------- | ------ | --------- | ------ | -------- | ------------- | ----------- | ------------------ |
| [void.1](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard#dotlabs-void.1)\*                 | **90.15M** | **39.46%**    | **67.19%** | 49.32%   | **29.35%**        | 43.70%      | **24.38**              |
| [palmer-006](https://huggingface.co/appvoid/palmer-006)           | 91.10M | 38.41%    | 63.60% | 44.82%  | 29.27%     |   **52.70%** | 23.33 |
| [100M-exp](https://huggingface.co/User01110/100M-exp)             | 98.16M | 37.78%    | 66.97% | **49.83%**   | 27.22%        | 40.00%      | 22.47              |
| [Rose-1.5-Medium](https://huggingface.co/GODELEV/Rose-1.5-Medium) | 98.28M | 38.09%    | 64.80% | 47.22%   | 27.13%        | 40.70%      | 21.07              |
| [tinctura-v1](https://huggingface.co/bench-labs/tinctura-v1)      | 96.2M  | 37.96%    | 65.61% | 47.98%   | 25.77%        | 38.40%      | 20.81              |
| [Surjo-100m](https://huggingface.co/SurjoLabs/Surjo-100m)         | 97.7M  | 35.05%    | 63.87% | 47.64%   | 25.85%        | 38.90%      | 18.86              |

## Details


We used the revision on step 900,000 (updated with ~10B decayed to 0) for evaluations which trained for ~12 days on a single RTX PRO 6000 gpu. It trained on 131.69 billion bytes which is between 32.92B and 43.90B standard tokens, around ~38 billion bpe tokens on average. For more details on the evals and inference, please have a look at the [official notebook](https://huggingface.co/dotlabs/void.1/blob/main/evals.ipynb). For precise training details, check the training's [config file](https://huggingface.co/dotlabs/void.1/blob/main/training_config.json) and [metrics](https://huggingface.co/dotlabs/void.1/tensorboard).
```bibtex
@misc{dotlabs,
  title  = {void: bytes is all you need},
  author = {appvoid},
  year   = {2026},
  url    = {https://huggingface.co/dotlabs/void.1}
}