# Second hosted experiments — foundry and larger-model qualification

**Date:** 24 October 2026  
**Source head:** `f7d6d08b555498fb9fc1b24ba0bba8ea0d92d7e8`  
**Actions run:** `40093694143`  
**Result:** all seven jobs passed on the first attempt.

This run tested two things in parallel:

1. a three-Sprocket mutation stage followed by an LFM2.5-1.2B selection/repair/recombination stage;
2. CPU qualification of six candidate models on independent standard GitHub-hosted `ubuntu-24.04` runners.

The model jobs ran independently and therefore do **not** constitute a same-host controlled benchmark. GitHub-hosted placement varied between AMD EPYC generations and llama.cpp selected different CPU backends. Treat the measurements as feasibility/first-order throughput observations, not fine-grained model ranking.

## Model qualification summary

All model downloads were SHA-256 verified before execution. Candidate GGUFs were deliberately left ephemeral instead of entering the shared Actions cache; the large candidates would otherwise consume most of the repository cache budget.

| Candidate | GGUF size | RSS after load | Startup | pp64 | tg64 | Behaviour probe |
|---|---:|---:|---:|---:|---:|---:|
| LFM2.5-VL-450M Q4_K_M | 0.214 GiB | 0.447 GiB | 0.523 s | 292.98 t/s | **129.54 t/s** | 121 tok / 1.173 s = **103.2 t/s** |
| LFM2.5-1.2B-Instruct QAD-Q4_0 | 0.648 GiB | 1.299 GiB | 0.516 s | 317.07 t/s | **63.98 t/s** | 99 tok / 1.817 s = **54.5 t/s** |
| LFM2.5-1.2B-Thinking Q4_K_M | 0.681 GiB | 1.260 GiB | 0.774 s | 118.10 t/s | **29.43 t/s** | 192 tok / 7.015 s = **27.4 t/s** |
| LFM2.5-8B-A1B Q4_K_M | 4.801 GiB | 8.611 GiB | 3.966 s | 50.79 t/s | **31.52 t/s** | 192 tok / 7.776 s = **24.7 t/s** |
| Qwen3.5-2B Q4_K_M | 1.222 GiB | 2.013 GiB | 1.276 s | 66.01 t/s | **23.95 t/s** | 118 tok / 6.469 s = **18.2 t/s** |
| Qwen3.5-4B Q4_K_M | 2.592 GiB | 5.533 GiB | 1.788 s | 64.37 t/s | **6.59 t/s** | 177 tok / 28.764 s = **6.2 t/s** |

### LFM2.5-VL-450M

The text backbone is extremely fast and tiny. Its first behaviour response was structurally weak and repetitive: it mostly echoed state/action labels rather than constructing a meaningful mechanism. That makes it interesting as an ultra-cheap classifier, filter, formatter or high-volume primitive mutator, but this one prompt does not support using it as the main behaviour designer.

This run exercised **text inference only**. It did not provision the vision projector or feed images, so the model's visual capability remains unqualified here.

### LFM2.5-1.2B-Instruct

This remains the strongest default general-purpose model in the current CPU ecology. It combined high throughput with a coherent, bounded behaviour answer and low resident memory. The first prompt produced a concrete obstacle/gap state machine and an explicit repeated-jump failure mode.

Its QAD quantization also makes direct throughput comparison against the Q4_K_M candidates approximate.

### LFM2.5-1.2B-Thinking

The Thinking model ran comfortably, but both probes spent their entire completion allowances inside explicit reasoning. Even the nominal "thinking disabled" concise probe emitted `<think>` content until the 192-token cap and never reached a final answer.

Observed benchmark decode was about 29.4 t/s; the two reasoning generations were approximately 27–29 t/s. It is technically viable but needs a deliberate reasoning budget/stop policy before unattended use.

### LFM2.5-8B-A1B

The sparse 8B model is the standout surprise.

The Q4_K_M GGUF was 4.801 GiB and the 4K-context server reported 8.611 GiB RSS, which fit comfortably on the hosted 16 GiB class runner. Decode reached 31.52 t/s in `llama-bench`, exceeding the first Qwen3.5-2B observation despite a much larger resident model.

Like the 1.2B Thinking model, both probes consumed their token budgets in reasoning before producing final content. Its reasoning itself was more promising: it converged toward storing a bounded last-push vector and applying a decaying/local response. The main requirement is therefore output-budget control, not feasibility.

### Qwen3.5-2B

The 2B model fit easily at about 2.0 GiB RSS. The concise non-thinking probe produced a valid-looking small state/action design, although it used four named states despite the prompt's maximum of three state values.

The reasoning probe used all 256 completion tokens before reaching final content. Its chain did accurately decompose the push mechanism into input force, bounded stored direction and deterministic action, suggesting potential as a critic when a larger reasoning allowance is acceptable.

### Qwen3.5-4B

The 4B model fit at 5.533 GiB RSS but was by far the slowest decoder in this first matrix: 6.59 t/s `tg64` and about 6.2 t/s for the concise behaviour probe.

Its concise output was arguably the best single free-prose kernel of the Qwen pair: a slope-triggered "Tilt-Seek" mechanism with bounded slope/velocity/energy, a deterministic update, visible downhill movement and a slippery-surface waiting-loop failure mode. The quality is interesting, but the throughput penalty is substantial.

The reasoning probe again exhausted 256 tokens without final content.

## Foundry experiment

The foundry topology was:

`Sprocket inverter ┐`
`Sprocket memory   ├─ parallel 48-token mutations → LFM2.5-1.2B selector/recombiner → JSON kernel`
`Sprocket contact  ┘`

All four servers remained resident. The three Sprocket mutations completed concurrently in **1.531 seconds**.

### Sprocket mutations

The 48-token cap successfully stopped the previous long degeneration but exposed a different failure mode: two of three mutations were truncated while still echoing/reframing the instruction.

The third candidate was the only clearly useful mutation:

> The aim is to predict the agent's future position, then to determine how to avoid that outcome. The plan is to predict where it is headed, then to select an appropriate control plan to steer it safely.

This is still too abstract for a final kernel, but it supplied a usable predictive/avoidance mutation.

### LFM recombination

The selector returned parseable JSON on the first attempt:

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

Mechanically, this is a success: parallel noisy generation can be normalized into a structured kernel.

Semantically, it exposes the next problems:

- the final kernel discarded most of the original pushed-direction mechanism;
- LFM sanitized the interesting mutation into generic obstacle avoidance;
- `parents_used: [1, 2]` is not credible because candidate 3 contained the predictive concept that dominates the result;
- model-declared lineage therefore cannot be treated as provenance;
- 48-token prose mutations are still too unconstrained for Sprocket.

The next foundry format should make each Sprocket fill a tiny fixed mutation record such as `trigger_delta`, `state_delta`, `action_delta`, and `failure_guess`. Parent IDs should be attached by the orchestrator, never generated by the selector. LFM should receive immutable candidate IDs and perform only selection/recombination.

## Initial role recommendation

Based on this single qualification pass:

| Role | Current first choice | Reason |
|---|---|---|
| high-volume chaos/mutation | Sprocket ×3 | cheap and fast; useful when heavily bounded |
| ultra-cheap text filtering/formatting | LFM2.5-VL-450M backbone | ~100 t/s real probe, tiny RSS |
| routine compiler/selector | **LFM2.5-1.2B-Instruct** | best current balance of speed, coherence and memory |
| deeper periodic reasoning | **LFM2.5-8B-A1B** | sparse model retains ~25–32 t/s despite 8B total parameters |
| alternate critic | Qwen3.5-2B | viable but slower; reasoning needs budgeting |
| expensive quality/diversity probe | Qwen3.5-4B | good single output, but ~6 t/s |
| specialized reasoning experiment | LFM2.5-1.2B-Thinking | viable, but currently reasoning-budget limited |

These are provisional roles, not model-quality rankings. The next controlled comparison should run identical structured candidate-selection tasks and score schema validity, constraint adherence, novelty retention, lineage correctness and throughput.

## Evidence

Run `40093694143`, source `f7d6d08b555498fb9fc1b24ba0bba8ea0d92d7e8`.

| Artifact | ID | ZIP SHA-256 |
|---|---:|---|
| `neurounner-foundry-a1` | 12559803302 | `691a6921782d92b09944a1bb31662325b7103aed1867aa8558584f0ac004cf46` |
| `neurounner-qualify-lfm25-vl-450m-a1` | 12560087096 | `9c8ff9d74f1d0e5a5f9a0c3ff04599836c30cfc39661abc659e450798b3ec3b9` |
| `neurounner-qualify-lfm25-1.2b-instruct-a1` | 12559763472 | `4fe1cb5c5dab1cc907e759287a84282e0164a3353900bebc1ff159df5ad4ba83` |
| `neurounner-qualify-lfm25-1.2b-thinking-a1` | 12559683850 | `71219507961a244df7ad3ae63f8df976f465036b36fb4b375eebae0174f9ae23` |
| `neurounner-qualify-lfm25-8b-a1b-a1` | 12560301224 | `7eeb3f74c5d2ab248384ff21d23fdf565757d5cb2f0fa42b33a1e3b6ca0b73ad` |
| `neurounner-qualify-qwen35-2b-a1` | 12559633991 | `63bea728cd831195147c428d3c4548b5a26e694764686a3d5f3a69ed3b4352fd` |
| `neurounner-qualify-qwen35-4b-a1` | 12560381039 | `b069dcc560ddd3f79793bebf5305cffc330c1383ac78931a2acbd84849453815` |

Artifacts were configured for seven-day retention; this committed record preserves the measurements, identities and archive hashes after artifact expiry.
