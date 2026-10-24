# TRAIN-004 — SmolLM2-135M full-parameter CPU feasibility spike

Date: 25 October 2026 (Actions run started 2026-10-24T23:33Z)  
Repository: `techrote/neurounner`  
Issue: #5  
Actions run: `40143212748`  
Measured source head: `760c5709db158d227a12657e54a6caae3a34346c`

## Verdict

**Classification: practical but expensive.**

Full-parameter float32 AdamW training of the pinned SmolLM2-135M-Instruct model is technically comfortable on the standard four-vCPU / ~16 GiB `ubuntu-24.04` runner class. On the same AMD EPYC 7763 CPU model used by authoritative TRAIN-001, the measured training loop was only **1.10× slower** than LoRA and used **1.77× peak RSS**. The dominant penalty was the genuine resumable checkpoint: **1.614 GB / 1.504 GiB**, or **54.32×** the TRAIN-001 LoRA checkpoint.

That makes full tuning surprisingly competitive in raw CPU compute, but materially worse for storage, artifact transfer, modularity and repeated multi-run checkpointing. LoRA remains the default for current neurounner specialization/persona workloads. Full tuning is justified only when changing the entire base model is itself valuable enough to pay the checkpoint/storage cost.

## Experimental controls

The experiment kept the TRAIN-001 task shape fixed:

- base model: `HuggingFaceTB/SmolLM2-135M-Instruct`;
- exact revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`;
- dataset: `experiments/train-001/kernel_compiler_dataset.json`;
- dataset SHA-256: `db76aedf89eae32e4514c212b2feb21db7b2b46700ae5aabf6d35861a073cf41`;
- same chat-template construction and exact common-prefix assistant-only loss masking;
- float32 CPU training, four PyTorch threads, one interop thread;
- seed: `20261024`;
- AdamW betas `(0.9, 0.95)`, weight decay `0.01`, gradient clip `1.0`.

The full-model lane trained all **134,515,008** base-model parameters at a single conservative learning rate of **5e-5**. No learning-rate sweep was performed. The simultaneous LoRA control retained TRAIN-001's r8 / alpha16 / dropout0.05 recipe and **8e-4** learning rate.

Observed package stack for both primary lanes:

- Python 3.12.3
- PyTorch 2.14.1+cpu
- Transformers 4.57.6
- PEFT 0.21.2
- huggingface_hub 0.36.2
- safetensors 0.8.0
- psutil 7.2.2

## Primary full-parameter measurement

Runner:

- OS: standard GitHub-hosted `ubuntu-24.04`;
- CPU: **AMD EPYC 7763 64-Core Processor**;
- logical CPUs exposed: **4**;
- total parameters: **134,515,008**;
- trainable parameters: **134,515,008 (100%)**.

The initial bounded lane completed all **16 optimizer steps**:

| Metric | Full parameter |
|---|---:|
| Sequence tokens | 2,463 |
| Supervised/output tokens | 1,151 |
| Training wall time | 23.125 s |
| Sequence tokens/s | **106.51** |
| Supervised tokens/s | **49.77** |
| Seconds/step | **1.445** |
| First loss | 4.7135 |
| Final loss | 3.3857 |
| Mean loss | 3.7087 |

Per-step losses are preserved in `experiments/train-004/evidence/full.json`.

The loss moved downward over the bounded run, but TRAIN-004 is not a convergence or model-quality experiment. The lower 16-step final loss than the LoRA control should not be interpreted as a demonstrated downstream-quality advantage.

## Memory measurement

The full lane measured process memory at the requested phase boundaries:

| Phase | RSS / peak |
|---|---:|
| Before model load | 381,916 KiB RSS |
| After model load | 926,192 KiB RSS; 1,188,480 KiB peak |
| After AdamW construction | 926,192 KiB RSS; 1,188,480 KiB peak |
| After first optimizer step | 2,709,516 KiB RSS; **2,917,348 KiB peak** |
| After 16 steps | 2,804,236 KiB RSS; **3,087,556 KiB peak** |

Peak RSS was **3,087,556 KiB = 2.945 GiB**.

AdamW state was allocated lazily: constructing the optimizer did not materially increase RSS, while the first optimizer step raised peak RSS by about **1.65 GiB** over the post-load peak. After that first step, the peak increased by only another **170,208 KiB (~166 MiB)** through step 16. There was therefore no sign of uncontrolled per-step memory growth.

At peak training memory the runner still reported roughly 12 GiB of system memory available. The experiment did not approach its memory safety guards and produced no OOM/preflight failure.

## Checkpoint and restartability

The full lane wrote a genuine resumable checkpoint containing:

- full model weights;
- AdamW optimizer state;
- global step and sample cursor;
- Python RNG state;
- Torch RNG state;
- exact base model/revision, dataset hash, seed, optimizer and package/source metadata.

Measured checkpoint:

- total: **1,614,446,024 bytes = 1.504 GiB**;
- model payload: **538,091,413 bytes**;
- training state: **1,076,352,633 bytes**;
- local checkpoint save time: **1.812 s**;
- manifest SHA-256: `e6cb22371b8632c5286f5ccb375485e4bf7b8ce2bf46cc03e65ebe77196458b2`.

The large binary checkpoint was not committed to Git. It was uploaded as Actions artifact `train004-full-checkpoint-a1` in run `40143212748`; its file-level hashes and manifest are committed under `experiments/train-004/evidence/`.

A separate fresh GitHub-hosted runner downloaded the checkpoint, verified all **5 files** against the manifest, loaded the exact local model, restored AdamW and both RNG states, restored global step/sample cursor at **16/16**, and continued successfully to **32/32**.

Fresh-runner continuation:

- CPU: **Intel Xeon 6973P-C**, four logical CPUs;
- additional optimizer steps: **16**;
- sequence tokens: 2,445;
- throughput: 176.15 sequence tok/s (host-specific; do not compare directly with the EPYC lanes);
- loss: 2.2597 → 2.0045;
- peak RSS: 3,017,172 KiB.

This is stronger than a one-step reload smoke: restartability across runner boundaries is empirically demonstrated through another 16 optimizer steps.

## Direct LoRA comparison

A simultaneous 16-step LoRA control used the same first 16 examples in the same order. It landed on an **AMD EPYC 9V74**, so its throughput ratio is host-confounded; memory and checkpoint ratios nevertheless closely match the authoritative TRAIN-001 reference.

| Metric | Simultaneous LoRA | Full parameter | Full / LoRA |
|---|---:|---:|---:|
| Trainable parameters | 2,442,240 | 134,515,008 | **55.08×** |
| Sequence tok/s | 134.88 | 106.51 | 0.790× |
| Seconds/step | 1.141 | 1.445 | **1.266×** |
| Peak RSS | 1,748,916 KiB | 3,087,556 KiB | **1.765×** |
| Checkpoint size | 29,722,667 B | 1,614,446,024 B | **54.317×** |
| Loss movement, same 16 examples | 4.7135 → 3.6081 | 4.7135 → 3.3857 | larger full-model drop; feasibility-only |

For throughput, the cleaner comparison is the authoritative TRAIN-001 stage-1 result because both it and TRAIN-004 full parameter ran on an **AMD EPYC 7763 with four exposed CPUs**:

| Metric | TRAIN-001 LoRA | TRAIN-004 full | Full / LoRA |
|---|---:|---:|---:|
| Trainable parameters | 2,442,240 | 134,515,008 | **55.08×** |
| Sequence tok/s | 117.18 | 106.51 | 0.909× |
| Seconds/step | 1.310 | 1.445 | **1.104×** |
| Peak RSS | 1,748,660 KiB | 3,087,556 KiB | **1.766×** |
| Resumable checkpoint | 29,721,169 B | 1,614,446,024 B | **54.320×** |

Thus the best host-controlled slowdown measurement is **1.10×**, not an extrapolation from parameter count. The simultaneous control gives a more conservative **1.27×** slowdown on a different, faster host.

## Decision

1. **Does full tuning fit comfortably in runner memory?** Yes. Peak RSS was 2.945 GiB on a ~16 GiB runner, with no continuing memory-growth problem after AdamW state allocation.
2. **Measured slowdown relative to LoRA?** **1.10× on the same EPYC 7763 CPU model**. The simultaneous but different-host control measured 1.27×.
3. **Measured memory multiplier?** **1.77×** peak RSS.
4. **Checkpoint multiplier?** **54.32×**, from ~28.34 MiB to 1.504 GiB.
5. **Is a multi-hour full-training campaign realistic?** Yes. At the measured 106.51 sequence tok/s, the raw loop corresponds to about **1.53M sequence tokens in four hours** or **1.92M in five hours**, before setup/evaluation/checkpoint overhead.
6. **Is it competitive with LoRA for current neurounner workloads?** In raw CPU training throughput, unexpectedly yes. Operationally, generally no: current persona/specialization work benefits strongly from ~30 MB resumable LoRA checkpoints and ~10 MB swappable adapter payloads rather than ~1.6 GB full checkpoints.
7. **When should future work use full tuning?** When the intended change genuinely needs to alter the shared base model, when LoRA capacity has been measured as the limiting factor, or when a demonstrated quality gain justifies the storage/transfer cost. It should not replace LoRA by default for current modular persona, formatting, or narrow specialization jobs.

The bounded feasibility question is therefore answered: **full-parameter SmolLM2-135M CPU training is practical, restartable and memory-safe on this runner class, but expensive in checkpoint/storage terms.**

## Durable evidence

- `experiments/train-004/evidence/full.json` — complete primary full-lane telemetry and per-step history;
- `experiments/train-004/evidence/lora.json` — simultaneous LoRA control;
- `experiments/train-004/evidence/resume.json` — fresh-runner hash verification, state restore and step-16→32 continuation;
- `experiments/train-004/evidence/summary.json` — machine-readable ratios against both controls;
- `experiments/train-004/evidence/full-checkpoint-manifest.json`;
- `experiments/train-004/evidence/lora-checkpoint-manifest.json`;
- `scripts/train_004.py`;
- `scripts/train_004_compare.py`;
- `.github/workflows/train-004.yml`.

No failed preflight or OOM evidence exists for this run because all bounded lanes completed successfully.
