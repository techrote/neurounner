# Larger-model qualification and Foundry v1 — 24 October 2026

Source head: `f7d6d08b555498fb9fc1b24ba0bba8ea0d92d7e8`  
Actions run: `40093694143`  
Runner class: GitHub-hosted `ubuntu-24.04`, four logical CPUs.

Seven jobs ran concurrently: one three-Sprocket→LFM foundry experiment and six isolated model qualifications. All seven completed successfully on the first attempt.

## CPU qualification summary

| Candidate | GGUF size | Server RSS | Startup | pp64 | tg64 | Behaviour probe |
|---|---:|---:|---:|---:|---:|---:|
| LFM2.5-VL-450M Q4_K_M | 0.214 GiB | 0.447 GiB | 0.523 s | 292.98 t/s | 129.54 t/s | 121 tokens / 1.173 s |
| LFM2.5-1.2B-Instruct QAD-Q4_0 | 0.648 GiB | 1.299 GiB | 0.516 s | 317.07 t/s | 63.98 t/s | 99 / 1.817 s |
| LFM2.5-1.2B-Thinking Q4_K_M | 0.681 GiB | 1.260 GiB | 0.774 s | 118.10 t/s | 29.43 t/s | 192 / 7.015 s |
| LFM2.5-8B-A1B Q4_K_M | 4.801 GiB | 8.611 GiB | 3.966 s | 50.79 t/s | 31.52 t/s | 192 / 7.776 s |
| Qwen3.5-2B Q4_K_M | 1.222 GiB | 2.013 GiB | 1.276 s | 66.01 t/s | 23.95 t/s | 118 / 6.469 s |
| Qwen3.5-4B Q4_K_M | 2.592 GiB | 5.533 GiB | 1.788 s | 64.37 t/s | 6.59 t/s | 177 / 28.764 s |

These are single hosted observations on heterogeneous GitHub hardware, not stable comparative benchmarks. Model architecture, quantization family and host placement differ.

## Model-specific observations

### LFM2.5-VL-450M

Extremely cheap CPU inference. The text-only probe completed at roughly 103 completion tokens/s and the model occupies less than half a GiB RSS. Output quality was weak and schematic:

> Trigger: PUSH; State: MOVING/TURNING/JUMPING/WAITING; Action: PUSH.

Its main reason to keep investigating is multimodal use: if image observation proves useful for NPC/world-state interpretation, this is small enough to run almost as a sidecar. The current test did **not** exercise the vision projector, so only text-core feasibility is established.

### LFM2.5-1.2B-Instruct

Best current throughput/coherence balance for ordinary synthesis. Its QAD quant produced about 64 tg64 tokens/s and a real probe around 54.5 completion tokens/s. The generated kernel was generic but coherent and obeyed the physical-behaviour framing substantially better than Sprocket.

This remains the default selector/compiler candidate.

### LFM2.5-1.2B-Thinking

Technically easy to host but substantially slower than the instruct QAD build. Both probes consumed their full token ceilings inside `<think>` reasoning and failed to reach a final answer.

That is an important protocol result rather than a runtime failure: reasoning budgets must be explicit and much larger, or the thinking trace must be treated as the useful artifact itself. It is not suitable for tiny bounded final-answer slots without a reasoning budget policy.

### LFM2.5-8B-A1B

The sparse 8B model fits on the standard 16 GiB runner with real headroom:

- 4.801 GiB GGUF
- 8.611 GiB process RSS at 4K context
- 3.966 s startup
- 31.52 tg64 tokens/s

This is a strong feasibility result. Despite the 8B resident parameter count, token generation was slightly faster than the 1.2B Thinking Q4_K_M benchmark.

Both current probes spent their full 192/256 token limits in reasoning and emitted no final answer. Like the 1.2B Thinking model, it needs explicit reasoning-budget handling rather than ordinary short-response limits.

### Qwen3.5-2B

Fits very comfortably:

- 1.222 GiB GGUF
- 2.013 GiB RSS
- 23.95 tg64 tokens/s

The non-thinking behaviour probe completed coherently in 6.469 s. The reasoning-enabled probe hit its 256-token ceiling in the reasoning channel without a final response.

The output was more structured than LFM1.2-Instruct but still generic. It is worth testing as a critic/schema author rather than bulk mutator.

### Qwen3.5-4B

Fits in RAM but is dramatically less attractive for CPU throughput:

- 2.592 GiB GGUF
- 5.533 GiB RSS
- only 6.59 tg64 tokens/s
- 28.764 s for the concise 177-token behaviour probe
- 39.668 s for a 256-token reasoning probe

Its concise output was arguably the strongest single kernel from this suite: a bounded slope/velocity/energy mechanism with a concrete slippery-surface failure mode. However, its CPU cost is high enough that it should initially be reserved for occasional high-value critique or selection rather than continuous generation.

## Foundry v1: three Sprockets → LFM

Three one-thread Sprocket servers generated independent 48-token mutations concurrently.

- mutation 1: 48 tokens / 1.508 s
- mutation 2: 48 / 1.529 s
- mutation 3: 48 / 1.158 s
- parallel mutation wall: 1.531 s

The Sprocket outputs were still mostly malformed/meta-text. The LFM selector nevertheless repaired them into syntactically valid JSON:

```json
{
  "name": "PredictAndAvoid",
  "trigger": "Detected deviation from expected path",
  "state": ["Near obstacle", "Forwarding", "Avoiding obstacle"],
  "update": "Reduce speed and turn left",
  "action": "turn left",
  "interesting_failure": "Persistent loop due to incorrect prediction",
  "parents_used": [1, 2]
}
```

The pipeline therefore proves:

1. three concurrent Sprocket mutations are cheap;
2. a resident one-thread LFM can immediately consume them;
3. LFM can normalize noisy material into machine-readable output;
4. free-prose Sprocket mutation prompts remain the weak link.

The next Foundry protocol should constrain **Sprocket itself** to tiny fields or grammar-constrained fragments, e.g. `trigger`, `memory`, `action`, rather than asking for prose. It should also score whether the LFM output actually preserves any novel parent mechanism instead of merely producing a generic replacement.

## Storage strategy

The core Actions cache is currently about 962 MB and contains the pinned runtime, Sprocket and the 1.2B Instruct model.

The larger candidates are intentionally downloaded one-per-runner and discarded. This prevented the 4.8 GiB 8B model and 2.6 GiB Qwen-4B from consuming the core cache quota. The separate `techrote/ilikebigweightsandicannotlie` repository is reserved as a future model-storage/manifest location if persistent large-weight staging becomes useful; it was not needed for this run.

## Evidence

Run artifacts:

- `neurounner-foundry-a1` — artifact `12559803302`, SHA-256 `691a6921782d92b09944a1bb31662325b7103aed1867aa8558584f0ac004cf46`
- `neurounner-qualify-lfm25-vl-450m-a1` — `12560087096`, SHA-256 `9c8ff9d74f1d0e5a5f9a0c3ff04599836c30cfc39661abc659e450798b3ec3b9`
- `neurounner-qualify-lfm25-1.2b-instruct-a1` — `12559763472`, SHA-256 `4fe1cb5c5dab1cc907e759287a84282e0164a3353900bebc1ff159df5ad4ba83`
- `neurounner-qualify-lfm25-1.2b-thinking-a1` — `12559683850`, SHA-256 `71219507961a244df7ad3ae63f8df976f465036b36fb4b375eebae0174f9ae23`
- `neurounner-qualify-lfm25-8b-a1b-a1` — `12560301224`, SHA-256 `7eeb3f74c5d2ab248384ff21d23fdf565757d5cb2f0fa42b33a1e3b6ca0b73ad`
- `neurounner-qualify-qwen35-2b-a1` — `12559633991`, SHA-256 `63bea728cd831195147c428d3c4548b5a26e694764686a3d5f3a69ed3b4352fd`
- `neurounner-qualify-qwen35-4b-a1` — `12560381039`, SHA-256 `b069dcc560ddd3f79793bebf5305cffc330c1383ac78931a2acbd84849453815`
