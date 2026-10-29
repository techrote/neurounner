# DETECTIVE-001B2 — measured JSON versus natural-language witness presentation

**Original native run:** [Actions 40582407781](https://github.com/techrote/neurounner/actions/runs/40582407781), source head 258980aae4d2930f52f2e87a020aaad6db333a2d.

**Verification:** native base and SFT release pins PASS; independent archive readback PASS; 33 offline tests PASS; **3 model lanes, 12 first-turn responses, 730 generated tokens** against 960 maximum. No training, GPUs or paid model APIs.

## Preregistered matched-format diagnostic

All three source-matched lanes answered the same two Bay 17 witness situations. JSON and natural-language prompts contained the **same case, fact IDs, admitted fact texts and player question**; the system instruction, native weights and greedy decode remained unchanged. No followups were included in this A/B, eliminating previous-answer confounds.

| Lane | JSON replies starting with object brace | Natural replies starting with object brace | JSON tokens | Natural tokens |
|---|---:|---:|---:|---:|
| Untuned SmolLM2 BASE | 2/2 | 0/2 | 149 | 116 |
| V-G02-A SFT | 1/2 | 0/2 | 149 | 120 |
| V-G09-C SFT | 1/2 | 0/2 | 125 | 71 |
| **Total** | **4/6** | **0/6** | **423** | **307** |

Literal echoed schema-key labels appeared in **3/6 JSON outputs**, **0/6 natural** outputs. Generation time summed 27.58 seconds across JSON outputs versus 20.04 seconds for natural outputs, on the same CPU; unequal output lengths preclude a pure speed comparison. These are **mechanical presentation** observations, not measurements of truth, helpfulness, deception, style-neutral identity or game solvability.

## Representative source readback

- Untuned BASE, clock witness: JSON output began by copying the admitted-memories object and stopped mid-question at 80 generated tokens. Natural output attempted witness prose with the nine-minute clock difference and 02:14 reroute, but repeated the same battery clause multiple times. Natural format changed turn-taking but **did not eliminate repetition**.
- V-G02-A: both clock responses contained obsessive clock-and-evidence language, with natural formatting producing a somewhat shorter, direct version. A JSON prompt did **not always** induce schema echo in this adapter.
- V-G09-C: the JSON clock version repeatedly mentioned nine minutes; natural returned terse phrases about a missing battery. The repeated clock/battery motif survived while JSON markup did not.
- Camera witness: JSON responses included straight JSON echo (BASE, G02-A) or corrupted JSON-like repetition around a "stripped gear" (G09-C). Natural prompts produced prose instead, but G09-C still hallucinated associations such as a "body" and a machine.

Natural presentation removed brace-prefixed output from all six sampled responses but was **not** a reliable conversational-policy fix. Nothing here proves strategic lying, character motivation, coherent memories, or that the game's mandatory evidence routes are traversable.

## Evidence and source custody

- Full pinned base model: HuggingFaceTB/SmolLM2-135M-Instruct revision 12fd25f77366fa6b3b4b768ec3050bf629380bac; native model.safetensors SHA-256 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c. Same two independently trained Village SFT LoRAs as the first smoke.
- AMD EPYC 7763, four inference threads, Python 3.12.15, torch 2.14.1+cpu, Transformers 4.57.6, PEFT 0.21.2; peak process RSS 1,303,524 KiB.
- [Raw evidence](../../datasets/detective-001-presentation-ab/) contains all exact request messages, special-token and ordinary decoded outputs, token IDs, SHA hashes, source head, release identities, timings, mechanical metrics and file inventory.
- Independent verifier recreated all 12 inputs from the frozen case and presentation contract, checked release/adapter references, grade recomputation and integrity inventory, then permitted promotion. Original [DETECTIVE-001B](REPORT_001B.md) archive remains unchanged.
- Source is one already-observed **development case family**, greedy decoding with one seed. There was no sealed case-family-disjoint holdout, script-model execution, new training, special tool-call model, or permission to change canonical case facts.

## Decision

**Source and evidence infrastructure: PASS. Detective NPC / deliberate deception / scripter admission: NOT EVALUATED.**

The natural renderer is a suitable *candidate* for additional NPC testing because this small paired comparison found fewer structured echoes. Before training, lock a separate case-family-disjoint holdout, synthetic independent belief/intent labels (truthful, corrupted belief, deliberate lie, concealed fact, uninformed invention) and a small per-character LoRA budget. Train selected characters against the same original SFT parents, compare to base and to prompt-only controls and preserve regressions.

The distinct **Sprocket-500M root Instruct** tool/memory checkpoint deserves its own source-correct fictional evidence scripting test; do not silently substitute the Chat subfolder checkpoint or a frozen NCR-001 experimental adapter. The original untuned SmolLM2 scripting result remains **0/2 strict JSON-valid outputs** and is not reclassified by this presentation study.
