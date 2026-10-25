# RUNNER-001 — Cross-platform GitHub-hosted runner benchmark

Issue: #51

## Purpose

Neurounner has useful Linux-only measurements for llama.cpp inference and SmolLM2 CPU training. RUNNER-001 measures what changes when comparable work is executed on standard GitHub-hosted Linux, Windows, and macOS runners.

The experiment deliberately separates:

1. **runner-local work**, where CPU/OS/ISA differences can materially affect Neurounner throughput; and
2. **GitHub Models API calls**, where model inference is remote and runner differences mostly represent client/network/service variance.

## Lanes

| Lane | GitHub label | Role |
|---|---|---|
| Linux x64 | `ubuntu-24.04` | current Neurounner baseline |
| Windows x64 | `windows-2025` | x64 OS comparison |
| macOS Apple Silicon | `macos-15` | normal Apple-hosted arm64 lane |
| macOS Intel | `macos-15-intel` | partial x64 control |

Do not describe the Linux/Windows/Apple-Silicon comparison as an OS-only causal experiment. Standard hosted runner hardware is not identical. The workflow records the concrete CPU, architecture, CPU count, memory and image metadata for every run.

## Fixed identities

- local model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- local revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- corpus: `experiments/train-001/kernel_compiler_dataset.json` (SHA-256 recorded per run)
- GitHub Models endpoint: `https://models.github.ai/inference/chat/completions`
- default remote model: `openai/gpt-4.1-mini`

The remote model can be overridden on manual dispatch. Unavailable GitHub Models access is recorded as `unavailable`; it is never substituted with another provider or model.

## Measurements

Each lane records:

- JSON parse/serialize throughput for the existing TRAIN-001 corpus;
- SHA-256 throughput for checkpoint-like bytes;
- zlib throughput for artifact/checkpoint-like compression;
- bounded PyTorch CPU matrix multiplication throughput;
- cold pinned-model load time;
- deterministic SmolLM2 CPU generation throughput;
- four bounded LoRA optimizer steps over existing TRAIN-001 examples;
- three GitHub Models API round-trip measurements and token usage when available.

The aggregate summary reports raw metrics and ratios against `ubuntu-24.04-x64`.

## Interpretation

The most decision-relevant numbers for Neurounner are local model generation, LoRA step throughput, cold setup/load time, memory headroom and total job wall time.

GitHub Models latency is useful for orchestration planning, but it is not evidence that one runner executes the hosted model faster: the inference happens remotely.

## Reproduction

Run **Actions → RUNNER-001 cross-platform benchmark** or push this benchmark branch. Per-lane JSON and aggregate JSON/Markdown evidence are retained as Actions artifacts for 30 days.
