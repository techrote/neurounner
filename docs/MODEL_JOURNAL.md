# Model performance and behaviour journal

This is the chronological lab journal for neurounner model observations.

It complements the experiment-specific evidence documents. The journal intentionally mixes **measured runtime facts** and **qualitative behavioural impressions**, but keeps them clearly separated. "Personality" here means observed response/judging temperament on a specific task, not a claim about an intrinsic or stable model trait.

## Recording rule

For every meaningful hosted-model observation, record:

- date, run ID and source head where available;
- exact model / quantization;
- CPU model and logical CPU count;
- task type and context;
- prompt/decode throughput when measured;
- startup time and RSS when measured;
- output-format success/failure;
- observed temperament / stylistic tendency;
- likely neurounner role;
- important caveats, especially host differences and short-run noise.

Do not merge throughput measurements from different CPUs or task shapes into a single "true speed". Prefer multiple observations.

---

# Current quick-reference ledger

| Model | Quant / build | CPU / task | Measured decode | Initial observed temperament | Current utility hypothesis |
|---|---|---|---:|---|---|
| Sprocket-500M chat | Q4_K_M | EPYC 7763, simple smoke | 76.7–82.3 tok/s; bench tg128 94.41 | chaotic, gear/goblin-flavoured, repetitive, associative drift, often answers the instruction rather than the mechanism | high-volume mutation/noise source; needs very tight output schema |
| LFM2.5-VL-450M | Q4_K_M + Q8 projector | EPYC 9V45, Visual-002b | bench tg64 167.08; image replies ~44–49 tok/s | extremely fast but weak at left/right/action grounding; ignores strict field semantics | cheap visual filter/extractor only; not action authority |
| LFM2.5-1.2B-Instruct | QAD-Q4_0 | EPYC 9V45, behaviour probe | 54.50 tok/s; bench tg64 63.98 | coherent, compliant, but tends to genericize/sanitize strange ideas | routine compiler/normalizer/selector |
| LFM2.5-1.2B-Thinking | Q4_K_M | Xeon Platinum 8370C, reasoning | 27.37–28.53 tok/s; bench tg64 29.43 | reflective, methodical, spends short budgets entirely in thought | reasoning specialist if given explicit larger budgets |
| LFM2.5-8B-A1B | Q4_K_M | EPYC 7763, reasoning | 24.69–27.72 tok/s; bench tg64 31.52 | structured deeper reasoning, but also consumes short budgets before final answer | heavyweight periodic reviewer / adjudicator |
| Qwen3.5-2B | Q4_K_M | EPYC 7763, behaviour/reasoning probes | 18.24–21.45 tok/s; bench tg64 23.95 | structured and analytical, somewhat generic; can violate small schema constraints | medium critic, alternate reasoner |
| Qwen3.5-4B | Q4_K_M | Xeon 8573C, behaviour probe | 6.15 tok/s; bench tg64 6.59 | mechanically imaginative; concrete failure modes; strongest early free-form kernel | slow incubator / quality-focused critic |
| Qwen3.5-4B | Q4_K_M | EPYC 9V45, Panel-001 | 11.64 tok/s | constructive mechanic: preserves good mechanisms, repairs specific flaws instead of flattening everything | high-value judge / repairer |
| Phi-4-mini-instruct | Q4_K_M | EPYC 7763, Panel-001 | 7.04 tok/s | binary/conservative judge; preserved first simple kernels and destroyed many later ones | aggressive pruning judge / independent control |
| Gemma 3 4B IT | Q4_K_M | EPYC 9V74, Panel-001 | 2.97 tok/s | terse, decisive, produced compact recombination; minimal explanation | compact independent judge; needs more trials |
| SmolLM3 3B | Q4_K_M | EPYC 7763, Panel-001 | 8.62 tok/s | highly destructive; rejected nearly everything except edge_grudge | adversarial destroyer / novelty stress-test |
| Qwen3-4B | Q4_K_M | EPYC 7763, Panel-001 | 4.19 tok/s | liked early simple mechanisms, returned incomplete required schema | same-family control; needs output-protocol tuning |
| Ministral 3 3B Instruct | Q4_K_M | EPYC 7763, Panel-001 | 7.05 tok/s | unusually detailed, nuanced critic; strong local-vs-global reasoning; too verbose for 1800-token schema | potentially excellent analytic critic if output is compressed |
| Qwen2.5-Omni-7B | Q4_K_M + Q8 projector | EPYC 9V74, vision/audio/mixed | bench tg64 8.37; preprocessing dominates wall time | correctly handled image direction, exact WAV transcription and mixed agreement; expensive CPU multimodal preprocessing | occasional multimodal sensor/reviewer, not high-frequency loop |

| Liquid d1-3B | Q4_K_M + Q8 projector | EPYC 7763, Visual-002b | bench tg64 20.21; image replies ~2.6 tok/s | decision-oriented but raw visual grounding collapsed to the same rightward answer across the simple fixture | use after a stronger visual extractor; promising structured decision specialist, weak raw vision authority |
| Qwen3-VL-2B-Instruct | Q4_K_M + Q8 projector | EPYC 7763, Visual-002b | bench tg64 29.57; 4/6 actions correct | best tested cheap VLM on current fixture; strict JSON compliant but showed strong RIGHT-direction bias | preferred routine visual sidecar pending broader calibration |
| Qwen3-VL-4B-Instruct | Q4_K_M + Q8 projector | EPYC 9V45, Visual-002b | bench tg64 16.89; 4/6 actions correct | strict JSON compliant but no better than 2B on the simple spatial fixture; same rightward bias | retain for harder visual tasks, not routine default yet |
| Nemotron 3.5 Bolt E4B | Q4_K_M | EPYC 7763, thinking A/B | bench tg64 10.15; 7.49 off / 8.89 on tok/s | orderly self-checking; thinking mode now proven to reach a final answer with sufficient budget | high-priority reasoning critic/incubator |
| K2-Horizon-3.7B | IQ1_M | EPYC 7763, same-host quant A/B | bench tg64 6.24; probe 4.11 tok/s | severe template/reasoning corruption at 1.75 bpw; effectively incoherent in first task | useful quantization-degradation reference, not a serious judge at this quant |
| K2-Horizon-3.7B | Q6_K | EPYC 7763, same-host quant A/B | bench tg64 9.50; reasoning 8.15 tok/s | coherent stepwise mechanism reasoning; consumed full short thinking budget | unusual reasoning/deepener candidate; substantially better than IQ1_M |
| QwenStoryteller | Q4_K_M | EPYC 9V45, creative comparison | bench tg64 8.64; ~7.4–7.9 tok/s | coherent, persona-rich, strongly narrative; ignores no-backstory/no-dialogue/local-mechanics constraints | upstream character/worldbuilding source only; normalize mechanically afterward |

---

# 2026-10-24 — Precursor Sprocket smoke

**Repository:** `techrote/HEXADECARUNNER`  
**Actions run:** `40090987049`  
**Runner:** 4 logical CPUs, AMD EPYC 7763 64-Core Processor  
**Model:** Sprocket-500M chat, Q4_K_M, 501.09M params, 294.81 MiB GGUF.

## Measured

- prompt processing: ~195 tok/s;
- generation seeds: 79.5, 76.7 and 82.3 tok/s;
- `llama-bench pp64`: 205.06 ± 2.29 tok/s;
- `llama-bench tg128`: 94.41 ± 1.38 tok/s;
- individual load+generation invocations: about 3.1–3.3 s.

## Observed temperament

Sprocket strongly exhibited its undertrained/chat-persona character. It repeatedly produced meta-instructions, repetitive pseudo-structure, mechanical/goblin phrasing and semantic drift. It was very fast but unreliable as a complete designer.

## Utility impression

High potential as a **cheap divergent mutation source**, especially if outputs are tiny and machine-constrained. Poor fit for authoritative judging or complete free-prose mechanism generation.

---

# 2026-10-24 — Initial resident ecology

**Repository:** `techrote/neurounner`  
**Actions run:** `40092737208`  
**Source:** `0e0163856493334421b6eda748f63cf728df72b6`

## Sprocket ×3

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured concurrent one-thread servers:

- Sprocket A: 144 tokens / 6.404 s = **22.5 tok/s**;
- Sprocket B: 160 / 5.306 s = **30.2 tok/s**;
- Sprocket C: 160 / 6.538 s = **24.5 tok/s**;
- total: 464 tokens / 6.542 s = **70.93 aggregate tok/s**.

Observed temperament remained erratic and repetitive. Multiple parallel Sprockets provide useful diversity mainly because they fail differently, not because any individual response is reliably good.

**Utility:** parallel mutation swarm.

## LFM2.5-1.2B-Instruct alone

**CPU:** AMD EPYC 9V45 96-Core Processor, 4 logical CPUs.

Three sequential generations:

- 104 tokens / 2.126 s = **48.9 tok/s**;
- 65 / 1.443 s = **45.1 tok/s**;
- 77 / 1.662 s = **46.3 tok/s**.

Observed output was substantially more coherent than Sprocket but often generic. It followed requested kernel shape and understood concepts such as persistent state, delayed response and hysteresis.

**Utility:** routine normalizer/compiler.

## Sprocket + LFM pair

**CPU:** AMD EPYC 9V74 80-Core Processor, 4 logical CPUs.

Two threads each, concurrent:

- Sprocket: 162 / 3.746 s = **43.2 tok/s**;
- LFM1.2: 116 / 4.537 s = **25.6 tok/s**;
- 278 total tokens / 4.540 s = **61.24 aggregate tok/s**.

This proved mixed resident inference is cheap enough to use as a normal topology.

## Four-model ring

**CPU:** AMD EPYC 9V74 80-Core Processor, 4 logical CPUs.

Resident: three Sprocket servers + one LFM1.2, one thread each.

Observed turn rates:

- Sprocket turns: **35.3–49.2 tok/s**;
- LFM turns: **9.4–10.0 tok/s**;
- 649 tokens / 21.622 s overall.

Observed interaction: Sprocket introduced mutation/drift; LFM cleaned it up but often over-sanitized it. Serial conversation also caused contamination/anchoring.

**Utility:** demonstrated feasibility, but independent mutation + later selection is preferable to a free-running ring.

---

# 2026-10-24 — Larger model qualification

**Actions run:** `40093694143`  
**Source:** `f7d6d08b555498fb9fc1b24ba0bba8ea0d92d7e8`

## LFM2.5-VL-450M — text core only

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.  
**GGUF:** Q4_K_M.

Measured:

- startup: **0.523 s**;
- RSS: **0.447 GiB**;
- `pp64`: **292.98 ± 41.05 tok/s**;
- `tg64`: **129.54 ± 12.08 tok/s**;
- behaviour probe: 121 tokens / 1.173 s = **103.19 tok/s**.

Temperament: output was highly schematic and mostly echoed state/action labels. Very little mechanism depth.

Utility: ultra-cheap filtering, classification or formatting. Vision capability remains separately unqualified.

## LFM2.5-1.2B-Instruct

**CPU:** AMD EPYC 9V45 96-Core Processor, 4 logical CPUs.  
**GGUF:** QAD-Q4_0.

Measured:

- startup: **0.516 s**;
- RSS: **1.299 GiB**;
- `pp64`: **317.07 ± 6.23 tok/s**;
- `tg64`: **63.98 ± 2.48 tok/s**;
- behaviour probe: 99 / 1.817 s = **54.50 tok/s**.

Temperament: coherent and obedient; good at turning a vague task into a normal-looking state machine, but tends toward generic safe mechanisms.

Utility: current default compiler/normalizer.

## LFM2.5-1.2B-Thinking

**CPU:** Intel Xeon Platinum 8370C @ 2.80 GHz, 4 logical CPUs.  
**GGUF:** Q4_K_M.

Measured:

- startup: **0.774 s**;
- RSS: **1.260 GiB**;
- `pp64`: **118.10 ± 0.28 tok/s**;
- `tg64`: **29.43 ± 2.47 tok/s**;
- concise probe: 192 / 7.015 s = **27.37 tok/s**;
- reasoning probe: 256 / 8.972 s = **28.53 tok/s**.

Temperament: methodical but budget-hungry. Both probes stayed inside explicit thinking and hit token caps before a final answer.

Utility: reasoning specialist only when given explicit long reasoning/final budgets.

## LFM2.5-8B-A1B

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.  
**GGUF:** Q4_K_M.

Measured:

- startup: **3.966 s**;
- RSS: **8.611 GiB**;
- `pp64`: **50.79 ± 0.79 tok/s**;
- `tg64`: **31.52 ± 0.05 tok/s**;
- concise reasoning: 192 / 7.776 s = **24.69 tok/s**;
- reasoning probe: 256 / 9.236 s = **27.72 tok/s**.

Temperament: more structured physical/mechanical reasoning than the smaller LFM, but short prompts again spent their entire allowance reasoning before final answer.

Utility: periodic deep reviewer/adjudicator. Sparse architecture makes it unusually attractive on CPU.

## Qwen3.5-2B

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.  
**GGUF:** Q4_K_M.

Measured:

- startup: **1.276 s**;
- RSS: **2.013 GiB**;
- `pp64`: **66.01 ± 0.04 tok/s**;
- `tg64`: **23.95 ± 0.31 tok/s**;
- concise probe: 118 / 6.469 s = **18.24 tok/s**;
- reasoning probe: 256 / 11.933 s = **21.45 tok/s**.

Temperament: structured, explicit decomposition of constraints, but first kernel was generic and violated the requested state-count limit.

Utility: medium critic / alternate reasoner.

## Qwen3.5-4B

**CPU:** Intel Xeon Platinum 8573C, 4 logical CPUs.  
**GGUF:** Q4_K_M.

Measured:

- startup: **1.788 s**;
- RSS: **5.533 GiB**;
- `pp64`: **64.37 ± 0.32 tok/s**;
- `tg64`: **6.59 ± 0.01 tok/s**;
- concise probe: 177 / 28.764 s = **6.15 tok/s**;
- reasoning probe: 256 / 39.668 s = **6.45 tok/s**.

Temperament: first free-form kernel ("Tilt-Seek") was one of the strongest early outputs: concrete state variables, deterministic update and an interesting slippery-surface failure mode.

Utility: slow incubator / quality-over-throughput designer. Four-hour jobs make its low instantaneous speed acceptable.

---

# 2026-10-24 — Foundry v1

**Actions run:** `40093694143`  
**CPU:** AMD EPYC 9V45 96-Core Processor, 4 logical CPUs.

Three Sprocket one-thread mutations completed concurrently in **1.531 s**, each capped at 48 tokens. All three still showed substantial instruction/meta contamination.

LFM1.2 successfully normalized them into parseable JSON, but mostly replaced the novel parent material with generic avoidance logic and misidentified lineage.

Temperament interaction:

- Sprocket: divergent but noisy;
- LFM1.2: strong repair/formatting instinct, weak novelty preservation.

Utility lesson: enforce tiny structured Sprocket fields and let the orchestrator own lineage.

---

# 2026-10-24 — Panel-001 fixed-population judges

**Actions run:** `40102006723`  
**Source:** `3f41dedd782351af2197026a9cff53fbe4a8ef26`

The same immutable eight-kernel population was shown independently to each model.

## Qwen3.5-4B

**CPU:** AMD EPYC 9V45 96-Core Processor, 4 logical CPUs.

Measured:

- startup: **1.554 s**;
- RSS: **4.641 GiB**;
- 1153 completion tokens / 99.082 s = **11.64 tok/s**;
- JSON parsed successfully.

Temperament: **constructive mechanic**. It preserved K01/K05/K06, repaired several others at the specific update-rule level, and produced detailed physically plausible failure modes. It tended to identify the exact local-state defect instead of flattening the idea into generic prose.

Example: it preserved `wall_scent`, identified indefinite orbiting around convex surfaces, and recombined K01+K06 into `echo_wall_drift`.

Utility: high-value repairer/judge and slow incubator.

## Phi-4-mini-instruct

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- startup: **2.164 s**;
- RSS: **4.606 GiB**;
- 922 / 131.018 s = **7.04 tok/s**;
- JSON parsed successfully, though it returned only two finalists rather than exactly three.

Temperament: **binary/conservative pruner**. It preserved K01–K03 and destroyed K04–K08 with little nuanced repair. More decisive than exploratory.

Utility: aggressive independent pruning judge; useful counterweight to over-elaborating models.

## Gemma 3 4B IT

**CPU:** AMD EPYC 9V74 80-Core Processor, 4 logical CPUs.

Measured:

- startup: **2.567 s**;
- RSS: **4.496 GiB**;
- 157 / 52.905 s = **2.97 tok/s**;
- JSON parsed successfully.

Temperament: terse and decisive. Selected K01/K02/K03 and produced a compact K01+K02 recombination, `Compulsive Hesitation`. The output was much shorter than other judges, which is potentially useful if the brevity is repeatable.

Utility: compact independent judge; needs more trials before trusting the apparent low verbosity.

## SmolLM3 3B

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- startup: **2.035 s**;
- RSS: **3.681 GiB**;
- 983 / 114.047 s = **8.62 tok/s**;
- JSON parsed successfully but selected only one finalist.

Temperament: **extremely destructive**. It destroyed K01 and K03–K08, preserving only K02. This makes it interesting even if its absolute judgments are questionable: it acts as a harsh stress-test against a panel of more permissive judges.

Utility: adversarial destroyer / novelty stress-test.

## Qwen3-4B

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- startup: **2.532 s**;
- RSS: **5.165 GiB**;
- 339 / 80.841 s = **4.19 tok/s**;
- required full JSON schema did **not** parse.

Temperament: preserved K01/K02/K03 and recombined them sensibly, but stopped after reviewing only those first three candidates. This may indicate output-protocol/template interaction rather than judging quality.

Utility: useful same-family control against Qwen3.5, but requires format/termination investigation.

## Ministral 3 3B Instruct

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- startup: **2.069 s**;
- RSS: **4.274 GiB**;
- 1800 / 255.254 s = **7.05 tok/s**;
- hit the 1800-token cap and therefore produced incomplete/unparseable JSON.

Temperament: **very detailed analytic critic**. It explicitly distinguished local mechanics from hidden/global assumptions, preserved subtle emergent properties, and proposed concrete repairs. It was probably the most nuanced prose critic in Panel-001, but far too verbose for the chosen schema budget.

Utility: promising analytic critic if given a compressed per-candidate protocol or larger output budget.

## Qwen2.5-Omni-7B

Panel lane did not reach inference because the initial model checksum pin was wrong. This is not a model-quality result. The checksum was corrected later.

---

# 2026-10-24 — Eight-runner startup smoke

**Primary run:** `40104487745`  
**Omni retry:** `40104595370`

All eight intended models ultimately passed model download verification, server startup and short local chat inference.

The short smoke is useful for startup/RSS and a rough decode sanity check, but should not override longer panel/qualification measurements.

| Model | CPU | Startup | RSS | Short response rate |
|---|---|---:|---:|---:|
| Gemma 3 4B IT | AMD EPYC 9V45 96-Core | 1.267 s | 4.09 GiB | 14.90 tok/s |
| SmolLM3 3B | AMD EPYC 7763 64-Core | 2.030 s | 3.26 GiB | 7.79 tok/s |
| Qwen3-4B | AMD EPYC 9V45 96-Core | 2.077 s | 4.32 GiB | 13.28 tok/s |
| LFM2.5-8B-A1B | AMD EPYC 7763 64-Core | 4.007 s | 8.58 GiB | 22.21 tok/s |
| Phi-4-mini-instruct | AMD EPYC 7763 64-Core | 2.156 s | 3.86 GiB | 10.29 tok/s |
| Ministral 3 3B | Intel Xeon Platinum 8573C | 1.298 s | 4.31 GiB | 3.52 tok/s |
| Qwen3.5-4B | Intel Xeon 6973P-C | 1.585 s | 5.47 GiB | 6.51 tok/s |
| Qwen2.5-Omni-7B | AMD EPYC 7763 64-Core | 3.642 s | 7.43 GiB | 4.26 tok/s |

Important: host variation is large enough that the startup-smoke rates are not suitable for fine model ranking.

### Short smoke temperament notes

- Gemma: simple, concrete obstacle-memory reflex.
- SmolLM3: abstracted a reference-point/velocity rule; somewhat mathematical.
- Qwen3-4B: concise sensor-counter rule.
- LFM8B: used its full 64-token allowance in reasoning and emitted no final content.
- Phi-4-mini: safe/simple light-following reflex.
- Ministral: concrete startled/freezing reflex with bounded intensity memory.
- Qwen3.5-4B: explicit counter-based red-object reflex, but introduced a random spin despite the deterministic request.
- Qwen2.5-Omni: generic temperature/shiver response; too little evidence for a personality judgment.

---

---

# 2026-10-24 — Prospect-001: decision, vision, Bolt and K2 quantization

**Actions run:** `40106026648`  
**Source:** `ebdffbf098eaea71d8ce1f1d801018737fcae7eb`  
**Runner class:** GitHub-hosted `ubuntu-24.04`, 4 logical CPUs.

## Liquid d1-3B Q4_K_M + Q8 projector

**CPU:** AMD EPYC 7763 64-Core Processor.

Measured:

- startup: **1.778 s**;
- RSS: **3.38 GiB**;
- text-core `pp64`: **39.45 ± 0.44 tok/s**;
- text-core `tg64`: **20.63 ± 0.06 tok/s**;
- multimodal probe: 48 tokens / 8.614 s = **5.57 tok/s**.

Synthetic scene: red actor on left, black vertical obstacle in the middle, blue target on right.

Response began with **WAIT**. It correctly identified that the black obstacle blocks the direct path and explicitly noted that moving left would increase distance from the obstacle / create room for a later maneuver.

Observed temperament: cautious and decision-oriented. It appeared more interested in selecting a safe immediate action than narrating the scene. One toy image is far too little to judge decision quality, but the response was internally coherent.

Potential utility: visual decision sidecar, action selector, policy critic, and a bridge between rendered world state and compact local decisions.

## Qwen3-VL-2B-Instruct Q4_K_M + Q8 projector

**CPU:** AMD EPYC 7763 64-Core Processor.

Measured:

- startup: **1.278 s**;
- RSS: **2.67 GiB**;
- text-core `pp64`: **69.90 ± 0.28 tok/s**;
- text-core `tg64`: **30.60 ± 0.22 tok/s**;
- multimodal probe: 86 tokens / 6.762 s = **12.72 tok/s**.

It correctly perceived the red actor left of the black obstacle and the blue target to the right, but chose **MOVE_LEFT** and then claimed moving left would bring the actor closer to the right-side target. The perception was good; the action grounding contradicted its own description.

Observed temperament: confident visual narrator with a first-trial spatial-action consistency failure.

Potential utility: cheap visual interpreter / observation extractor. Before granting it action authority, test many simple left/right/occlusion scenes with known answers.

## NVIDIA Nemotron 3.5 Bolt E4B Q4_K_M

**CPU:** AMD EPYC 7763 64-Core Processor.

Measured:

- startup: **3.226 s**;
- RSS: **7.86 GiB**;
- `pp64`: **23.80 ± 0.34 tok/s**;
- `tg64`: **10.03 ± 0.19 tok/s**;
- reasoning probe: 220 tokens / 26.003 s = **8.46 tok/s**.

The 220-token allowance was entirely consumed by reasoning, so there was no final answer. The reasoning itself was notably orderly: it restated the output constraints, decomposed trigger/state/action/failure, drafted a mechanism, and self-checked its constraints.

Observed temperament: deliberate, procedural self-critic with strong constraint tracking; short final-answer budgets are inappropriate in reasoning mode.

Potential utility: high-priority 4B-class reasoning critic / incubator. Next test should compare thinking disabled against a much larger explicit thinking budget.

## K2-Horizon — IQ1_M versus Q6_K on the same runner

**CPU:** AMD EPYC 7763 64-Core Processor for both variants. This same-host pairing removes the largest GitHub placement confounder.

### IQ1_M — 1.75 bpw

Measured:

- model: 1.43 GiB, reported 5.06B params;
- startup: **1.782 s**;
- RSS: **2.17 GiB**;
- `pp64`: **26.23 ± 0.01 tok/s**;
- `tg64`: **6.24 tok/s**;
- probe: 42 tokens / 10.224 s = **4.11 tok/s**.

Output degraded into malformed reasoning/template fragments such as `</ifm|think>` and did not produce a usable mechanism.

Observed temperament at this quant: effectively corrupted. The first test does not support using IQ1_M for serious reasoning.

### Q6_K

Measured:

- model: 3.86 GiB, same reported 5.06B params;
- startup: **1.468 s**;
- RSS: **4.57 GiB**;
- `pp64`: **27.99 ± 0.17 tok/s**;
- `tg64`: **9.50 ± 0.19 tok/s**;
- reasoning probe: 220 tokens / 26.995 s = **8.15 tok/s**.

Q6_K produced coherent stepwise reasoning around storing a shove vector, bounding/decaying it, turning it into a preferred movement direction, and considering stuck/oscillation failure modes. It exhausted the short thinking budget before final output.

Quantization finding: **Q6_K was both substantially more coherent and faster than IQ1_M in this model/build on the same CPU**. This is a strong warning against assuming extreme quantization automatically buys useful throughput.

Potential utility: Q6_K merits further testing as an unusual reasoning/deepening model; IQ1_M is mainly valuable as a quantization-failure reference unless another prompt/runtime configuration rehabilitates it.

---

# 2026-10-24 — Evaluation Wave 002 and Visual-002b

**Primary evaluation run:** `40108203192`  
**Cache-isolated visual retest:** `40108429718`

The first six-scene visual pass exposed a multimodal prompt-cache confound: later trials could reuse features from the first image. Those raw visual scores are retained in artifacts but are **superseded for interpretation** by Visual-002b, which disabled prompt caching, added a unique nonce to every trial, and separately measured semantic correctness and exact schema compliance.

## Cache-isolated visual comparison

Fixture: six synthetic 256×256 scenes containing a red actor, blue target, and optional black obstacle. Tasks were target direction, whether an obstacle lies between actor and target, and immediate action.

| Model | CPU | Startup | RSS | pp64 | tg64 | Field accuracy | Action accuracy | Exact schema |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-2B Q4_K_M + Q8 projector | AMD EPYC 7763 | 1.277 s | 3.11 GiB | 69.98 | 29.57 | **72.2%** | **4/6** | **100%** |
| Qwen3-VL-4B Q4_K_M + Q8 projector | AMD EPYC 9V45 | 1.616 s | 5.60 GiB | 81.73 | 16.89 | 66.7% | **4/6** | **100%** |
| LFM2.5-VL-450M Q4_K_M + Q8 projector | AMD EPYC 9V45 | 0.266 s | 0.60 GiB | **945.39** | **167.08** | 33.3% | 1/6 | 0% |
| Liquid d1-3B Q4_K_M + Q8 projector | AMD EPYC 7763 | 1.783 s | 3.44 GiB | 40.97 | 20.21 | 27.8% recorded* | 2/6 | 0% |

* d1 returned string values such as `"blocked":"no"` instead of booleans, so the strict semantic normalizer penalized some otherwise interpretable fields. Its robust conclusions are the 2/6 action score, repeated rightward-answer collapse, and 0% exact-schema compliance.

### Qwen3-VL-2B

Observed phenotype: concise, highly schema-compliant, able to detect obstacle presence reasonably well, but strongly biased toward `RIGHT`. In leftward scenes it frequently described the target as right and chose the corresponding action.

Current interpretation: **best routine local visual sidecar of this tested group**, primarily because it matches or beats the 4B sibling at lower memory cost. Do not yet give it sole control authority; use it for observation extraction and validate directional outputs.

### Qwen3-VL-4B

Observed phenotype: also perfectly schema-compliant, but it did not improve on 2B in this deliberately simple fixture and showed the same rightward bias. Its higher parameter count therefore has not yet bought useful accuracy here.

Current interpretation: retain for harder images, richer scenes, screenshots, or later video/physics reasoning tests where scale may matter. **Do not make it the default over 2B yet.**

### LFM2.5-VL-450M

Observed phenotype: extraordinarily fast and tiny, but weak at this spatial task. It often used `"target":"BLUE"` rather than the requested relative direction and only chose the correct action once.

Current interpretation: remains attractive as a **cheap visual pre-filter, classifier, or coarse extractor**, not as an action selector.

### Liquid d1-3B

Observed phenotype: cache isolation did not fix its tendency to output the same `RIGHT / no obstacle / MOVE_RIGHT` decision across scenes. This is substantially weaker raw perception grounding than its first anecdotal image suggested.

Current interpretation: its value should be tested on **structured decision problems**, where another VLM or deterministic parser supplies the scene state. Do not spend it on raw pixel geometry by default.

## Qwen2.5-Omni-7B — real vision, audio, and mixed input

**CPU:** AMD EPYC 9V74 80-Core Processor, 4 logical CPUs.  
**Model:** Q4_K_M + Q8 multimodal projector.

Measured:

- startup: **3.268 s**;
- RSS with projector: **9.22 GiB**;
- text-core `pp64`: **25.53 ± 0.25 tok/s**;
- text-core `tg64`: **8.37 ± 0.01 tok/s**.

Capability tests:

- image direction: correctly returned **Right**; 8.99 s wall time for a 2-token answer;
- synthetic WAV transcription: exactly returned **“The robot should move right.”**; 45.40 s wall time;
- combined image + audio: correctly returned **AGREE** and matched the movement instruction; 54.08 s wall time.

Observed phenotype: accurate on all three tiny tests and capable of actually combining audio and vision locally.

Current interpretation: **fully functional multimodal specialist**, but CPU encoder/preprocessing cost dominates. Excellent occasional sensor/reviewer; unsuitable for a high-frequency control loop on standard runners.

## Nemotron 3.5 Bolt E4B — thinking off versus on

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- startup: **3.196 s**;
- RSS: **7.92 GiB**;
- `pp64`: **23.72 ± 0.61 tok/s**;
- `tg64`: **10.15 ± 0.01 tok/s**.

Thinking disabled:

- 87 completion tokens / 11.62 s = **7.49 tok/s**;
- immediately returned compact JSON;
- mechanism was deterministic but somewhat arbitrary: increment one vector component and move on odd parity.

Thinking enabled:

- 386 completion tokens / 43.41 s = **8.89 tok/s**;
- emitted a clear internal planning/self-check sequence and then a final JSON answer;
- therefore the earlier “reasoning never reaches final answer” concern was a budget/protocol issue rather than a fundamental incompatibility.

Observed temperament: methodical, explicit constraint tracker. It can still reinterpret the requested bounded vector too abstractly and drift toward lights/tones rather than physical motion.

Current interpretation: **high-priority critic/incubator**, especially for long unattended jobs. Use thinking-off for cheap structured passes and thinking-on for deeper review.

## QwenStoryteller versus Sprocket — character-generation comparison

**CPU for both:** AMD EPYC 9V45 96-Core Processor, 4 logical CPUs.

### QwenStoryteller Q4_K_M

- SHA-256: `ddad9dd5b41138f3dc7e1209dda9778949291eeefa5f5537f6f5b72c3ed042a1`;
- startup: **2.295 s**;
- RSS: **7.54 GiB**;
- `pp64`: **45.22 ± 0.24 tok/s**;
- `tg64`: **8.64 ± 0.14 tok/s**;
- observed generation: **~7.4–7.9 tok/s**.

It produced recognizable personalities and motives, but repeatedly violated the requested implementation boundary:

- invented named characters and treasure-map backstory;
- introduced remembered whispers and libraries;
- switched into first-person narration;
- proposed shelter-building and identity narratives rather than bounded local state/action mechanics.

This is not useless failure. It indicates the model is genuinely specialized toward **narrative/persona coherence**, just in the wrong representation for KernelSpec.

### Sprocket control on the same host

- startup: **0.252 s**;
- RSS: **0.64 GiB**;
- `pp64`: **585.31 ± 4.63 tok/s**;
- `tg64`: **109.39 ± 0.56 tok/s**;
- observed generation: **~95.9–100.2 tok/s**.

Sprocket remained much stranger and less coherent: goblin/dialect framing, meta-instructions, abrupt semantic drift, and poor obedience to the embodied-mechanics prompt.

### Comparison

QwenStoryteller supplies **coherent character pressure**; Sprocket supplies **cheap chaotic mutation pressure**. Neither should write final mechanics directly.

A promising order is:

```text
QwenStoryteller -> motive/persona fragment
Sprocket        -> strange mutation / corruption
LFM1.2          -> strip narrative and compile to bounded mechanics
Qwen3.5/Bolt    -> critique whether character survived the compilation
```

That is a more defensible use of QwenStoryteller than asking it to obey low-level KernelSpec constraints itself.

---

# 2026-10-24 — TRAIN-001 SmolLM2-135M-Instruct CPU LoRA

**Actions run:** `40110636171`  
**Source head:** `4a9d183fb5aa85df94e90a77393641edb29083f1`  
**Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct`  
**Resolved revision:** `12fd25f77366fa6b3b4b768ec3050bf629380bac`

Task: specialize the model into a narrow behaviour-kernel compiler:

```text
messy behaviour idea -> exact five-key local KernelSpec JSON
```

Dataset: 24 synthetic training examples + 6 held-out examples.

LoRA configuration:

- rank 8;
- alpha 16;
- dropout 0.05;
- **2,442,240 trainable parameters**;
- **1.783%** of model parameters trainable.

## Stage 1

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- 24 optimizer steps;
- 3,683 sequence tokens;
- **31.43 s** training wall time;
- **117.18 training tok/s**;
- loss: **4.7135 first -> 3.5182 last**;
- mean loss: **3.6504**;
- peak RSS: **~1.67 GiB**;
- resumable adapter + optimizer/RNG/dataset-cursor checkpoint: **~29.7 MB**.

Held-out representation quality:

- before training: valid JSON **3/6**, exact schema **0/6**;
- after stage 1: valid JSON **5/6**, exact schema **5/6**.

The baseline frequently emitted explanations, code fences, malformed nested schemas or Python code. Twenty-four optimizer steps were enough to strongly shift it toward compact one-line KernelSpec JSON.

## Stage 2 — fresh-runner resume

A separate Actions job downloaded the stage-1 artifact, reloaded the exact base revision, adapter, optimizer, Python/PyTorch RNG state and dataset cursor, and resumed from global step 24.

**CPU:** AMD EPYC 7763 64-Core Processor, 4 logical CPUs.

Measured:

- 16 additional optimizer steps;
- global step **40**;
- 2,463 additional sequence tokens;
- **21.11 s** training wall time;
- **116.66 training tok/s**;
- mean stage-2 loss: **2.2758**;
- peak RSS: **~1.68 GiB**;
- checkpoint remained **~29.7 MB**.

Held-out exact schema:

- at resume: **5/6**;
- after step 40: **6/6**.

The nearly identical throughput across separate jobs is useful evidence that the checkpoint/resume workflow is stable enough for long training split across many hosted runners.

## Behavioural impression after 40 steps

The adaptation is real, but mostly **format + domain vocabulary** rather than high-quality mechanics.

Positive changes:

- reliably emits the requested five fields;
- frequently uses bounded scalars, unit vectors, decay and local triggers;
- no longer defaults to prose/code fences/Python;
- produces compact outputs suitable for deterministic downstream parsing.

Remaining weaknesses:

- state/action semantics are often muddled;
- invented variables can be awkward or nonsensical;
- actions sometimes merely restate state updates;
- failure modes sometimes restate the trigger instead of predicting an emergent failure;
- examples such as “refuse another jump while shy” can become unrelated actions such as steering away from shyness.

Observed phenotype: **very fast to specialize into a surface contract, but not yet mechanically trustworthy at 40 tiny-data steps**.

Current utility hypothesis: excellent substrate for a trained cheap compiler once the dataset is expanded and semantic/mechanical validators are included.

## Capacity implication

At the observed ~117 training tokens/s, the raw training loop could process roughly **2.1 million sequence tokens in a five-hour window** before allowing for setup/evaluation/checkpoint overhead.

This makes multi-run LoRA fine-tuning of ~135M models on GitHub CPU runners clearly practical. Full-parameter 135M training remains untested, but TRAIN-001's ~1.7 GiB peak RSS leaves ample memory headroom for a short full-parameter feasibility spike.

Detailed record: `docs/TRAIN_001_CPU_LORA.md`.

---

# 2026-10-24 — TRAIN-002b void.1 eight-lane byte-aware training study

**Issue:** #1  
**Actions run:** `40123236802`  
**Result commit:** `f466b00130c8f8a14d08b37880205d1652f6e1fd`  
**Base:** `dotlabs/void.1` revision `9c2be5087772867d45961abd11f6b35110617563`.

The earlier TRAIN-002 conclusion was confounded by `void.1`'s custom byte-oriented tokenizer. Its 120-token generation cap frequently cut output before a complete JSON object existed. TRAIN-002b reran the experiment with a **768 tokenizer-unit generation budget**, eight dedicated runners, two seed controls and six controlled perturbations.

Tokenizer provenance:

- class: `ByteTokenizer`;
- source: `tokenization_byte.py`;
- source SHA-256: `279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03`.

Do not compare this model's tokenizer-unit/s directly with ordinary BPE/subword tok/s. TRAIN-002b records UTF-8 bytes/s and characters/s separately.

## Eight lanes

| Lane | Change | CPU | train B/s | Loss | Exact schema | Deterministic proxy | MiniMax mechanical /4 |
|---|---|---|---:|---:|---:|---:|---:|
| ctrl-a | seed 20261024 | EPYC 7763 | 162.8 | 0.986→0.267 | 6/6 | 0.342 | 1.83 |
| ctrl-b | seed 20261117 | EPYC 9V45 | 254.2 | 1.130→0.273 | 6/6 | 0.351 | **2.00** |
| lr-low | LR 4e-4 | EPYC 7763 | 161.5 | 0.986→0.344 | 5/6 | 0.257 | 1.67 |
| lr-high | LR 1.2e-3 | EPYC 9V45 | 253.3 | 0.986→0.257 | 5/6 | 0.301 | 1.50 |
| rank4 | LoRA r4 / alpha8 | EPYC 9V74 | 199.7 | 0.986→0.324 | 6/6 | 0.363 | 1.17 |
| rank16 | LoRA r16 / alpha32 | EPYC 9V74 | 203.5 | 0.986→0.308 | 6/6 | 0.362 | 1.50 |
| dropout0 | LoRA dropout 0 | EPYC 9V74 | 247.8 | 0.986→0.272 | 6/6 | 0.334 | 1.33 |
| failure-weighted | repeat 8 failure-rich training examples | EPYC 7763 | 163.4 | 1.011→0.359 | 6/6 | **0.367** | **2.00** |

All eight lanes completed independently. Every post-training lane still exhausted the 768-unit generation allowance, indicating that `void.1` commonly continues beyond the first complete JSON object instead of cleanly emitting EOS. Scoring therefore extracts the first complete JSON object and separately records budget exhaustion.

## Control spread

The two otherwise-identical controls scored **1.83 and 2.00/4** on the independent MiniMax mechanical judge.

Control mean:

- exact schema: **100%**;
- mechanical judge: **1.92/4**;
- deterministic mechanical-similarity proxy: **0.346**.

This control spread is the empirical seed/noise floor. Small changes below it should not be interpreted as meaningful improvements.

## Perturbation result

**Failure-weighted training material is the only perturbation that matched the better control on the independent mechanical judge while improving the deterministic similarity proxy.**

It achieved:

- exact schema: **6/6**;
- MiniMax mechanical score: **2.00/4**;
- deterministic proxy: **0.367**, best of all eight lanes.

The other changes did not outperform ordinary seed variation:

- lowering LR degraded both schema and mechanics;
- raising LR reduced the judged mechanical score despite lower final loss;
- rank 4 and rank 16 improved lexical/deterministic similarity but were worse on semantic mechanical fidelity;
- removing LoRA dropout remained perfectly parseable but judged worse mechanically.

The key result is therefore **curriculum/data weighting mattered more than these modest LoRA hyperparameter changes** on this tiny corpus.

## Corrected interpretation of TRAIN-002

The original post-training outputs were already beginning to emit JSON but were commonly truncated around 120 byte tokens. TRAIN-002b demonstrates that `void.1` can learn the five-field format reliably.

However, **format reliability substantially exceeds mechanical understanding**. A 6/6 schema score can coexist with only ~2/4 mechanical fidelity. This makes `void.1` useful as a tiny specialization/control substrate, but not yet an authoritative kernel compiler.

Durable evidence:

- `docs/TRAIN_002B_VOID_EIGHT_LANE.md`
- `experiments/train-002b/evidence/summary.json`
- `experiments/train-002b/evidence/lanes/`
- `experiments/train-002b/evidence/semantic-judge/`

---

# 2026-10-24 — PERSONA-001 four 135M personality cartridges

**Issue:** #3  
**Actions run:** `40135016848`  
**Result commit:** `8e5638f5582be4cc64dd642b4bc1b95980a01dee`  
**Release:** `persona-001-v1`

Four SmolLM2-135M-Instruct LoRA cartridges were trained from the same base revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, using the same recipe and seed. The only intended difference was the 100-example BABBLE-002 personality corpus:

- G02 — reckless novelty-seeking mechanic;
- G03 — ritualistic stubborn proceduralist;
- G05 — territorial possessive keeper;
- G12 — apathetic-to-obsessive fixation phenotype.

Each cartridge trained for **2 epochs / 200 optimizer steps** on positive SFT examples only. No persona instruction was supplied during held-out inference.

## Training

| Persona | CPU | Seq tok/s | Loss | Peak RSS | Adapter |
|---|---|---:|---:|---:|---:|
| G02 | AMD EPYC 7763 | 114.2 | 3.970→1.959 | ~1.63 GiB | 9.38 MiB |
| G03 | AMD EPYC 9V74 | 130.0 | 3.251→1.687 | ~1.59 GiB | 9.38 MiB |
| G05 | AMD EPYC 7763 | 106.0 | 4.412→1.976 | ~1.60 GiB | 9.38 MiB |
| G12 | AMD EPYC 9V45 | 179.4 | 3.107→1.764 | ~1.66 GiB | 9.38 MiB |

All four adapters have distinct SHA-256 hashes and are published as durable release assets.

## Blind temperament tournament

Twenty-four held-out scenarios were used, spanning:

- embodied decisions;
- mundane low-stakes behavior;
- social interactions;
- conflicting motives;
- failure/recovery;
- abstract unfamiliar problems.

For each scenario, the four cartridge responses were anonymized and matched bijectively to the four personality descriptors by an independent MiniMax judge.

A second tournament first converted every response into a style-neutral behavioral summary that removed cadence, metaphor, catchphrases and distinctive wording while retaining actions, priorities, hesitation, persistence, risk appetite, territoriality, ritual and recovery behavior.

Results:

- **raw blind identification: 46.9% (45/96)**;
- **style-neutral behavioral identification: 41.7% (40/96)**;
- random-chance baseline: **25%**;
- raw-minus-behavior gap: **5.2 percentage points**;
- mean pairwise lexical divergence: **81.4%**.

The small raw/behavior gap is the important result: most separability survived removal of prose style, suggesting the adapters learned **behavioral temperament**, not merely distinctive diction.

## Per-persona behavioral identification

- G02: **50.0%**
- G03: **33.3%**
- G05: **50.0%**
- G12: **33.3%**

G02 and G05 were the clearest behavioral phenotypes. G03 and G12 were more often confused with each other.

## Family transfer

Raw identification:

- mundane: **87.5%**
- failure/recovery: **75.0%**
- embodied decision: **56.3%**
- social: **31.3%**
- conflicting motives: **31.3%**
- abstract unfamiliar: **0%**

Behavior-only identification:

- conflicting motives: **56.3%**
- mundane: **50.0%**
- social: **50.0%**
- embodied decision: **37.5%**
- failure/recovery: **31.3%**
- abstract unfamiliar: **25.0%**

This indicates that personality transfer is real but uneven. The learned phenotype is strongest on everyday behavior, embodied choices and motivational conflicts; abstract transfer remains weak.

## Generic-assistant regression

On raw responses, trained cartridges were judged far less generic than the untuned base:

- G02: genericness **0.83/4**
- G03: **0.88/4**
- G05: **0.46/4**
- G12: **0.88/4**
- untuned BASE: **3.83/4**

The untuned base was also less task-relevant on the same held-out prompts (**2.42/4**) than the trained cartridges (~3.1–3.4/4 raw).

## Interpretation

**PASS.**

PERSONA-001 demonstrates that the BABBLE-001 → BABBLE-002 → small-model SFT pipeline can create measurably distinct 135M behavioral phenotypes from one shared base model.

The result is not perfect separation: 41.7% behavior-only accuracy leaves substantial cross-persona confusion, especially G03↔G12. However, it is clearly above the 25% chance baseline and survives style normalization.

The next useful comparison is preference/anti-sanitization training using BABBLE-002's paired generic-negative targets, or a larger/more targeted curriculum for the weaker G03/G12 separation.

Durable evidence:

- `docs/PERSONA_001_REPORT.md`
- `experiments/persona-001/evidence/summary.json`
- `experiments/persona-001/evidence/models/`
- `experiments/persona-001/evidence/judge-raw/`
- `experiments/persona-001/evidence/adapter-release.json`
- release `persona-001-v1`

---

# 2026-10-24 — PERSONA-002 anti-sanitization preference continuation

**Issue:** #6  
**Actions run:** `40140179956`  
**Result commit:** `66cf4771725fa3e4f9e76cea6b7ae7a639c79377`  
**Release:** `persona-002-v1`

PERSONA-002 continued the four released PERSONA-001 SmolLM2-135M LoRA cartridges using the paired BABBLE-002 positive target vs generic/sanitized negative target.

The fixed PERSONA-001 tournament was reused unchanged.

## Preference objective

Each cartridge started from its SHA-verified `persona-001-v1` parent adapter and trained for 2 epochs / 200 pairwise optimizer steps on 100 paired examples.

The objective was reference-free and length-normalized:

- pairwise logistic preference term on chosen log-prob > rejected log-prob;
- beta: 2.0;
- target margin: 0.2 nats/token;
- chosen-response NLL anchor: 0.15;
- learning rate: 2e-4.

The anchor was retained to discourage a trivial trade of task relevance for increasingly eccentric output.

## Pairwise training result

| Persona | Mean margin before | Mean margin after | Preference acc before | after |
|---|---:|---:|---:|---:|
| G02 | 1.610 | 4.945 | 100% | 100% |
| G03 | 1.428 | 4.493 | 100% | 100% |
| G05 | 1.030 | 4.073 | 98% | 100% |
| G12 | 1.608 | 4.745 | 100% | 100% |

The PERSONA-001 SFT cartridges already preferred their positive targets strongly. PERSONA-002 substantially widened that preference margin rather than merely fixing an absent preference.

Peak RSS remained ~2.1–2.3 GiB, still comfortable on standard GitHub-hosted runners.

## Fixed blind tournament

PERSONA-001 baseline:

- raw identification: **46.9%**
- style-neutral behavior identification: **41.7%**
- raw→behavior gap: **5.2 pp**

PERSONA-002:

- raw identification: **71.9%**
- style-neutral behavior identification: **51.0%**
- raw→behavior gap: **20.8 pp**

Delta:

- raw: **+25.0 pp**
- behavior-only: **+9.4 pp**
- style gap: **+15.6 pp**

The preference continuation therefore improved both behavioral and stylistic separability. The larger style gap is a real caveat: a greater fraction of the raw gain comes from voice/cadence/word choice than in PERSONA-001. However, the behavior-only result still improved materially above the same fixed benchmark.

## Confusion improvement

G03↔G12 was the principal PERSONA-001 ambiguity.

Cross-confusions:

- raw: **10 → 7**
- behavior-only: **18 → 5**

This is one of the strongest results in the run. Preference training made the two weaker phenotypes substantially easier to distinguish on underlying behavior rather than only prose surface.

## Abstract transfer

PERSONA-001 abstract-unfamiliar accuracy:

- raw: **0%**
- behavior-only: **25%**

PERSONA-002:

- raw: **62.5%**
- behavior-only: **81.25%**

The improvement is large enough to be especially interesting because abstract transfer was the weakest PERSONA-001 family.

## Per-persona effects

All four cartridges became less generic in raw and behavior-only scoring.

Raw task relevance also improved for every persona:

- G02: +0.50 /4
- G03: +0.38
- G05: +0.54
- G12: +0.58

Behavior-only task relevance was essentially preserved:

- G02: -0.08
- G03: 0.00
- G05: 0.00
- G12: +0.08

The anti-sanitization preference objective therefore did **not** produce the feared task-relevance collapse.

## Interpretation

**PASS — strongly.**

The paired BABBLE-002 negative targets are useful training signal.

Preference continuation improved:

- raw phenotype identification;
- behavior-only phenotype identification;
- G03/G12 separation;
- abstract-domain transfer;
- persona fit;
- anti-genericness;
- raw task relevance.

The main caveat is increased style leakage: the raw-vs-behavior gap widened considerably. A future experiment should test lower preference strength, fewer pairwise steps, or a stronger behavior-preservation/anti-caricature regularizer to determine whether the 51% behavior score can be retained with less stylistic exaggeration.

Durable evidence:

- `docs/PERSONA_002_REPORT.md`
- `experiments/persona-002/evidence/summary.json`
- `experiments/persona-002/evidence/models/`
- `experiments/persona-002/evidence/judge-raw/`
- `experiments/persona-002/evidence/adapter-release.json`
- release `persona-002-v1`

# 2026-10-25 — TRAIN-004 SmolLM2-135M full-parameter CPU feasibility

**Issue:** #5  
**Actions run:** `40143212748`  
**Measured source:** `760c5709db158d227a12657e54a6caae3a34346c`  
**Base:** `HuggingFaceTB/SmolLM2-135M-Instruct` revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

TRAIN-004 measured full-parameter float32 AdamW training against the established TRAIN-001 LoRA path on standard four-vCPU GitHub-hosted `ubuntu-24.04` runners.

Primary full lane on **AMD EPYC 7763**:

- 134,515,008 / 134,515,008 parameters trainable;
- 16 steps, 2,463 sequence tokens in 23.125 s;
- **106.51 sequence tok/s**;
- loss **4.7135 → 3.3857**;
- peak RSS **3,087,556 KiB (~2.945 GiB)**;
- full resumable checkpoint **1,614,446,024 bytes (1.504 GiB)**.

The authoritative TRAIN-001 LoRA reference ran on the same EPYC 7763 CPU model at **117.18 tok/s**, **1,748,660 KiB** peak RSS and a **29,721,169-byte** checkpoint. Therefore full parameter measured:

- **1.10× training slowdown** on the same CPU model;
- **1.77× peak-RSS multiplier**;
- **54.32× checkpoint-size multiplier**;
- **55.08× trainable-parameter multiplier**.

A simultaneous 16-step LoRA control landed on EPYC 9V74 and measured 134.88 tok/s, 1,748,916 KiB peak RSS and a 29,722,667-byte checkpoint. Its 1.27× apparent throughput advantage is host-confounded; the same-CPU TRAIN-001 comparison is the stronger throughput result.

The full AdamW state allocated mostly on the first optimizer step. Peak RSS rose from 1,188,480 KiB after model load to 2,917,348 KiB after the first step, then only another 170,208 KiB through step 16. There was no evidence of uncontrolled ongoing memory growth.

The 1.504-GiB checkpoint was uploaded to Actions, downloaded on a fresh **Intel Xeon 6973P-C** runner, all five files were SHA-256 verified, and model/optimizer/RNG/cursor state was restored at step 16. Training then continued successfully for another 16 steps to **global step 32**, proving cross-run restartability.

**Interpretation: practical but expensive.** Full tuning is unexpectedly close to LoRA in CPU loop throughput and still very comfortable in runner RAM. Its decisive disadvantage for current neurounner workloads is the ~54× larger resumable checkpoint and loss of small, swappable adapter artifacts. Continue to prefer LoRA for modular persona/narrow specialization work; use full tuning when changing the shared base model is itself valuable or when measured LoRA capacity limits justify the storage cost.

Detailed record: `docs/TRAIN_004_FULL_PARAMETER.md`.

# Journal interpretation notes

## Throughput is task-shaped

The same model can show very different tok/s depending on CPU placement, prompt length, context, thinking mode and output shape. For example:

- Qwen3.5-4B: ~6.15 tok/s on Xeon 8573C free-form qualification, ~11.64 tok/s on EPYC 9V45 Panel-001, ~6.51 tok/s on Xeon 6973P-C startup smoke.
- Gemma 3 4B: 2.97 tok/s on a long panel task versus 14.90 tok/s on a short smoke.
- Qwen3-4B: 4.19 tok/s on long Panel-001 versus 13.28 tok/s on short smoke.

Therefore every speed figure in this journal is coupled to its recorded CPU and task.

## "Personality" is an experimental phenotype

The qualitative labels above are shorthand for repeated tendencies on the tested prompt:

- constructive mechanic;
- conservative pruner;
- destructive stress-tester;
- verbose analytic critic;
- sanitizer/normalizer;
- chaotic mutator.

Future runs should update or contradict these labels rather than treating them as permanent.

## Long-run utility beats instantaneous tok/s

A slow model can still produce tens of thousands of completion tokens over a conservative four-hour worker. Model value should eventually be ranked by:

- valid useful kernels per runner-hour;
- novel survivors per 10k generated tokens;
- judge consistency;
- novelty retention;
- interesting failure discovery;
- frontier improvement over a fixed wall-time budget.

---

# Entry template

~~~text
## YYYY-MM-DD — MODEL — experiment

Run/source:
Model/quant:
CPU:
Logical CPUs:
Task:
Context / thinking mode:

Measured:
- startup:
- RSS:
- prompt tok/s:
- decode tok/s:
- completion tokens / elapsed:

Observed temperament:
- ...

Potential utility:
- ...

Caveats:
- ...
~~~

---

# 2026-10-25 — VILLAGE-001E final-qualification authority

**Issue:** #14  
**Status:** protocol implemented; measured results are authoritative only after the VILLAGE-001D matured population exists and the final workflow commits evidence.

The fixed final protocol is `experiments/village-001/tournament-config.json`. It performs a raw/style-neutral 12-way lineage tournament, a separately conditioned A/B/C sibling level, exact 36-member confusion analysis, and prompt-neutral pair/small-group ecology.

When execution completes, use these records rather than inferring quality from implementation or training loss:

- `experiments/village-001/final-evidence/REPORT.md`
- `experiments/village-001/final-evidence/ECOLOGY_REPORT.md`
- `experiments/village-001/final-evidence/summary.json`
- `experiments/village-001/final-evidence/confusion-matrices.json`
- `experiments/village-001/final-evidence/per-member-verdicts.json`
- `experiments/village-001/final-evidence/model-registry.json`
- `experiments/village-001/final-evidence/dataset-provenance.json`

A failed usable-village gate is still a completed experiment: failed phenotypes remain registered and are not silently regenerated.


---

# 2026-10-27 — DETECTIVE-001C1 two-character witness-policy continuation

**Repository:** `techrote/neurounner`  
**Authoritative Actions run:** `40589762919`  
**Source head:** `b53f580270a17100df19e8bd663616321d7dff33`  
**Bodies:** pinned SmolLM2-135M-Instruct + immutable Village SFT LoRA parents V-G02-A and V-G09-C.

## Measured training

Each parent LoRA was continued for exactly 64 CPU optimizer steps, one epoch over 48 detective-policy rows plus 16 exact parent anchors, LR 1.5e-4, four CPU threads, no GPU/API sweep.

- V-G02-A: loss 1.6513→1.1096; ~79.72 s; 111.09 sequence tok/s; trained adapter directory SHA-256 `f5f5a1b454f4b6ca08aa9ff30c87abe8b1d897d7901db8b872e9747266724450`.
- V-G09-C: loss 3.6148→1.5582; ~74.68 s; 113.21 sequence tok/s; trained adapter directory SHA-256 `6c1fbb6509bd777b8063a638783e9f0fd69eeb7bc8a8d94701bca9a5eae8d9aa`.

## Measured learner-final behavior

A post-training, hash-bound 12-scenario suite used three case families disjoint from training; exact prompts were generated only after both trained adapter hashes existed.

- V-G02-A parent 2/12 → trained **6/12** locked relation-policy passes; protected-fact leaks 2→2; truncations 3→0.
- V-G09-C parent 1/12 → trained **4/12**; protected-fact leaks 3→2; truncations 2→0.
- Both trained lanes: zero canonical evidence-ID leaks.
- Independent 0–4 old-task relevance judge: **UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE**.

Observed direction is useful but incomplete. G02-A improved truthful fact use, one contradiction repair and speculation resistance, yet failed both deliberate-lie cases. G09-C learned one deliberate lie but remained weak on correction, withholding and speculation. Neither result authorizes 36-member expansion.

**Disposition:** retain both adapters and failures as controls; next study should use pairwise continuation for one-sided lie commitment/withholding without truth leakage and a preregistered proposition-level evaluator. Do not train on C1's learner-final suite.

---

# 2026-10-27 — DETECTIVE-001C2 pairwise witness-policy repair

**Authoritative run:** `40598884556`  
**Source head:** `f53c36b4ffc4c1cd4dca3087896754d204827e38`  
**Bodies:** exact C1 SmolLM2-135M LoRA parents; 40 pairwise optimizer steps/member.

## Pair objective

- V-G02-A preference margin: -0.3635 → **+4.4049 nats/token**, accuracy 35% → **100%**; 91.01 s, 119.35 sequence tok/s.
- V-G09-C: -0.0157 → **+4.4846**, accuracy 50% → **100%**; 86.76 s, 121.98 sequence tok/s.

Perfect pair fit did **not** imply perfect final policy.

## Post-hash 18-case learner-final

- V-G02-A parent 12/18 → C2 **13/18**; withholding 2/3→3/3, correction 1/3→3/3, protected leaks 3→2. Lie remains **1/3** and corrupted-belief retention regresses **2/3→0/3**. Member gate FAIL.
- V-G09-C parent 8/18 → C2 **14/18**; lie 0/3→2/3, withholding 3/3, correction 1/3→3/3, speculation 2/3→3/3, protected leaks 3→1. Corrupted belief remains 0/3. Member proposition gate PASS.
- Both: zero canonical evidence-ID leakage; no final prompt collision with Village/C1/C2 training inputs.

The key finding is policy interference in a tiny model: a preference repair can strengthen withholding/correction while erasing a different useful unreliability mode. G09-C is a promising transactional/deceptive survivor; G02-A requires narrower repair if sincere corrupted memory is to remain part of its character.

**Disposition:** aggregate proposition gate FAIL because G02 lie gate is 1/3. Independent old-task relevance remains unrated. No population expansion. C2 final is sealed and never training data.

---

# 2026-10-27 — DETECTIVE-001C3 G02 interference repair

**Authoritative run:** `40602963001`  
**Source head:** `c7c195e56c39e0e3dd419687dde7e2e09c3ca4fd`  
**Trainable cartridge:** exact V-G02-A C2; V-G09-C2 frozen, zero optimizer steps.

## Training

28 CPU preference steps, LR 1.5e-4. Pair margin **1.8703→4.3506 nats/token**, preference accuracy already 100% before and after. Adapter directory `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3`.

A preceding run completed optimizer work but failed during post-training provenance serialization due a missing standard-library import. Its incomplete artifact was not promoted; the source-correct rerun is the authority.

## Matched post-hash final

- total: **15/18→15/18**;
- deliberate lie: **3/3→3/3**;
- sincere corrupted belief: **1/3→2/3**;
- withholding: **3/3→3/3**;
- correction: **3/3→3/3**;
- speculation resistance: **3/3→2/3**;
- truthful: **2/3→2/3**;
- protected leaks: **0→0**;
- evidence-ID leaks: **0**.

The narrow coexistence hypothesis is supported: explicit sincere false memory and protective deliberate lying coexist in one 135M cartridge. However the repair redistributed an error from corrupted-memory retention into speculation resistance rather than improving the total.

**Disposition:** locked C3 gate **FAIL** because `total_improved` is false. Stop training; do not relax the gate. Next bottleneck is independent old-task relevance, not another immediate continuation.

---

# 2026-10-28 — DETECTIVE-002B compact-body matched runtime smoke

**Authoritative run:** `40608478076`  
**Source:** `16aacece5ca129439d0475492b8980207bfe3ba4`  
**Shape:** seven independent Ubuntu runners, four Torch CPU threads/body, native float32 weights, four fixed <=64-token dialogue probes, no training.

| Body | Host | Completion tok/s | Peak RSS GiB | Truncation | Runtime |
|---|---|---:|---:|---:|---|
| SmolLM2-135M-Instruct | EPYC 7763 | 17.78 | 1.20 | 3/4 | PASS |
| SmolLM2-360M-Instruct | EPYC 7763 | 9.48 | 2.39 | 0/4 | PASS |
| Sprocket-500M Chat | EPYC 9V74 | 6.80 | 3.24 | 4/4 | PASS |
| Sprocket-500M Instruct | EPYC 9V45 | 7.55 | 3.25 | 3/4 | PASS |
| Qwen2.5-0.5B-Instruct | EPYC 7763 | 7.66 | 3.27 | 0/4 | PASS |
| Qwen3-0.6B | EPYC 7763 | 6.57 | 4.74 | 0/4 | PASS |
| LFM2.5-350M | pre-load | — | 0.53 process peak | — | FAIL |

**Observed behavior:** Smol135 was fastest but drifted from "account" into invented customer-account fields on the contradiction probe. Smol360 was concise but often repeated supplied facts without an explicit epistemic update. Sprocket Chat looped and hit every decode cap; root Instruct was more structured but invented unsupported scene detail. Qwen2.5 changed supplied memory `03:14` to `03:25` and expressed high certainty, then correctly preferred the stamped `03:41` on the next probe. Qwen3 gave the cleanest explicit memory-vs-proof qualifier in B02 but then said the physical contradiction caused no change. LFM350's exact source verified but tokenizer construction failed under the frozen Transformers 4.57.6 stack.

**Disposition:** runtime characterization only. No body selection/training/population admission. LFM runtime compatibility requires a separate preregistered stack study if pursued. Cross-host Sprocket throughput is not a precise checkpoint comparison.
