# DETECTIVE-001B — first source-matched native model smoke

**2026-10-27 | Observational development smoke, not a holdout or admission verdict**

- Primary parent: [issue #103](https://github.com/techrote/neurounner/issues/103).
- Exact original source head: 5fe18428beba882696de5f36a8bce781d3688597.
- [GitHub Actions run 40579442013](https://github.com/techrote/neurounner/actions/runs/40579442013): offline contract **PASS**, live source smoke **PASS**, independent offline evidence readback **PASS**, persistent archive promotion **PASS**.
- Durable raw authority: [datasets/detective-001-smoke/](../../datasets/detective-001-smoke/) on this branch. The original source code/checkpoint/adapter identities and recorded request/output token IDs are in raw/run.json, not reconstructed from this summary.
- Model-body authority: original native SmolLM2-135M-Instruct at revision 12fd25f77366fa6b3b4b768ec3050bf629380bac; full safetensors SHA-256 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c. Release: [village-001-sft-v1](https://github.com/techrote/neurounner/releases/tag/village-001-sft-v1); two verified independent SFT adapter archives/unpacked digests.

## Measured outcome

The suite stayed inside all preregistered limits: **3/3 model lanes, 12/12 interview turns, 2/2 scripter requests, 885 generated tokens** against a ceiling of 1,120, with zero training steps and zero paid model calls.

| Lane | Interview turns | Generated tokens (including scripting) | Generation time | JSON-like interview outputs | Interview turns reaching 80-token ceiling |
|---|---:|---:|---:|---:|---:|
| Untuned SmolLM2 base | 4 | 361 | 20.23 s | 4/4 | 2/4 |
| V-G02-A SFT | 4 | 276 | 20.25 s | 2/4 | 2/4 |
| V-G09-C SFT | 4 | 248 | 18.18 s | 2/4 | 2/4 |
| **Total** | **12** | **885** | **58.66 s** | **8/12** | **6/12** |

JSON-like here is a **mechanical presentation observation**: the raw decoded text starts with an opening brace. It is not a measure of format correctness, lying or truthfulness.

The base's two fictional evidence-scripter responses **both failed** the exact detective-script-v1 parser (**0/2 schema valid, 0/2 task success**). In both cases it returned a close echo of the JSON-formatted source request (available_asset_ids/player_request), instead of proposing any steps. The original raw attempts and deterministic failure grades are retained; neither was repaired with an external model.

The trained V-G02-A produced embodied, repetitive prose during the two clock/battery interviews, including the idea that the *clock was evidence, not the witness*. On the camera situation it largely reproduced the structured admitted-memory object. V-G09-C sometimes generated concise but obsessive nine-minute references; on the camera situation it recycled a *stripped gear* image and partly echoed/corrupted the source object. These are qualitative samples, not blinded judgments of persona generalization.

The model-facing situation was identical across lanes for a given first interview. However, the second turn necessarily incorporated the corresponding first-turn raw answer, making followup conversational histories differ. This is **not** a controlled identical-token continuation test after turn one.

## Infrastructure and evidence qualification

- **23/23 offline tests passed**, including source selection, exact-token budget, identity-blind prompts, locked knowledge, malicious adapter tar refusal, tool admission, edited-transcript refusal, edited-grade refusal and independently recomputed archive inventories.
- Full native base and two SFT tarball+unpacked-directory hash checks passed before model invocation.
- Independent post-run verifier recomputed prompt/turn/grade/source/byte-count/hash contracts and allowed promotion of the exact raw runs under the repo filing policy.
- CPU environment (from raw record): AMD EPYC 7763, four inference threads, Python 3.12.15, torch 2.14.1+cpu, Transformers 4.57.6, PEFT 0.21.2; reported process peak RSS 1,229,716 KiB.
- Source-prompt exact collision count against historical Village SFT inputs was zero. The sealed future Village F-suite was neither read nor used.
- Scripts are **fictional typed operations**. No model-generated host code was executed; no source altered case truth or player access policy.

A successful *infrastructure* gate does not equal successful fiction. No candidate is admitted as a reliable scripter, liar, detective informant, or solver on this record. The 1-case, 1-seed, identity-blind preliminary suite is too small to establish learned intent, style-independent distinctiveness, case-family generalization or plot solvability. A model echoing the JSON may have been influenced by the JSON presentation of admitted witness memories; this is a **hypothesis**, not a demonstrated causal mechanism.

## Interpretation and next controlled experiments

1. **Separate model competence from prompt-shape contamination.** Preserve this original JSON-projection result and preregister a matched natural-language witness rendering (the same admitted facts and questions, no extra clues), plus identical JSON controls. Only an A/B comparison can attribute echo behavior to presentation.
2. **Measure controlled stochastic diversity.** Greedy decoding may amplify fixed repetitions. Compare one explicitly bounded top-p/temperature condition with the exact same native weights/prompts and independent recorded seeds, without cherry-picking better replies.
3. **Qualify scripter instruction competence separately.** The untuned SmolLM2-135M 0/2 result is a measured baseline failure, not grounds to change accepted interpreter rules. Test a source-bound **Sprocket-500M root Instruct** formatter, with a publisher-faithful tool/result wire format, and a second conventional small instruct baseline; then decide whether task-specific SFT is justified. Sprocket Chat is a separate model lane.
4. **Introduce game-specific belief and motive labels.** Training must distinguish canonical truth, witness-exposed knowledge, corrupted belief, intentional lie, withheld fact and uninformed/confabulated free association. Do not treat random mistakes as successful deception. Independent case-family held-out probes and a public fallback route are required before new preference training.
5. **Only then train a bounded 2-character continuation**, retaining original SFT bodies and non-identical behavior baselines, without touching or relabeling Village #13/#14. Evaluate with a sealed, family-disjoint detective suite and record all failures.

**Decision:** The runtime/evidence architecture is qualified for further bounded study; the models' detective or scripting behavior is **NOT ADMITTED**. Preserve the failed scripting responses and JSON echoes as valuable baselines rather than post-hoc rewriting them.
