# RUNNER-001 — Cross-platform GitHub-hosted runner benchmark

Issue: #51

## Purpose

Neurounner already has useful Linux-only measurements for llama.cpp inference and SmolLM2 CPU training. RUNNER-001 measures the real hosted-runner choices for comparable Neurounner work on standard GitHub-hosted Linux, Windows, and Apple Silicon macOS runners.

This is a runner-choice benchmark, not a pure operating-system causal experiment: the standard hosted runner hardware differs.

## Comparable lanes

| Lane | GitHub label | Observed hosted shape |
|---|---|---|
| Linux x64 | `ubuntu-24.04` | 4 logical CPUs, x64 |
| Windows x64 | `windows-2025` | 4 logical CPUs, x64 |
| macOS Apple Silicon | `macos-15` | 3 CPUs, Apple M1 virtual / arm64 |

The workflow records concrete CPU identity, architecture, CPU count, memory, Python version, package versions, and runner image revision on every invocation.

### Intel macOS compatibility probe

An initial `macos-15-intel` control was attempted in run 40219158113. Python 3.12 was available, but the Intel macOS package index exposed only PyTorch 2.2.x while the three comparable lanes supported the same PyTorch 2.14.1 stack. The Intel lane is therefore retained as a compatibility finding rather than mixed into the performance table with a different ML runtime.

## Fixed identities

- local model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- local revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`
- corpus: `experiments/train-001/kernel_compiler_dataset.json`
- Python: 3.12
- PyTorch: 2.14.1 / 2.14.1+cpu
- Transformers: 4.57.6
- PEFT: 0.21.2
- huggingface_hub: 0.36.2
- psutil: 7.2.2
- safetensors: 0.8.0

The benchmark canonicalizes the JSON corpus before hashing and before byte-oriented microbenchmarks. This prevents Windows checkout line-ending conversion from masquerading as a different corpus or workload. The raw checkout hash is also retained for audit.

## Measurements

Each lane records:

- JSON parse/serialize throughput for the existing TRAIN-001 corpus;
- SHA-256 throughput for checkpoint-like bytes;
- zlib throughput for artifact/checkpoint-like compression;
- bounded PyTorch CPU matrix multiplication throughput;
- pinned SmolLM2 model load time and RSS growth;
- deterministic SmolLM2 CPU generation throughput;
- four bounded LoRA optimizer steps over existing TRAIN-001 examples;
- total inner benchmark wall time.

The aggregate step fails unless all three expected lanes are present and the canonical corpus SHA-256 is identical across them.

## GitHub Models finding

The originally planned remote GitHub Models latency control cannot be run. GitHub fully retired GitHub Models on 2026-07-30, including the playground, model catalog, inference API, and BYOK.

The pre-fix pilot deliberately attempted the former Actions inference endpoint and received unusable responses on the Linux and Apple lanes. The final benchmark records the service as retired and does not fabricate a remote-model latency figure.

Current GitHub-hosted AI model access is part of GitHub Copilot rather than the former GitHub Models inference API; it is not interchangeable with runner-local model compute.

Reference: https://docs.github.com/en/github-models

## Interpretation

The most decision-relevant numbers for Neurounner are local model generation, LoRA step throughput, memory headroom, setup/load behavior, and total job wall time. Generic JSON/hash/compression/matmul microbenchmarks are supporting diagnostics, not substitutes for the actual model workload.

In particular, the Apple runner has fewer CPUs than the x64 runners and a different ISA/microarchitecture. Its result answers “which standard GitHub-hosted runner is faster for this Neurounner workload?” rather than “which OS is intrinsically faster?”

## Reproduction

Run **Actions → RUNNER-001 cross-platform benchmark** or push the benchmark branch. Per-lane JSON and aggregate JSON/Markdown evidence are retained as Actions artifacts for 30 days.
