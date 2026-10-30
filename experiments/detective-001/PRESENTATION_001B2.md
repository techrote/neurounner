# DETECTIVE-001B2 — witness presentation A/B (development-only)

**Parent issue:** [#106](https://github.com/techrote/neurounner/issues/106). **Predecessor:** [first source-verified smoke, DETECTIVE-001B](REPORT_001B.md) (885 generated tokens, 3 model lanes, 0/2 script format-valid).

## Research question

The original real-native smoke used a strict JSON witness view, and **8/12 interview responses began with JSON-shaped output**. That may partly reflect *prompt mirroring* rather than the character's intrinsic incoherence. This test isolates **presentation format** without changing admitted memories or player questions. It does not test motives, intentional lying, scripts or end-to-end case solvability.

## Fixed experiment

- Exact released and SHA-verified SmolLM2-135M-Instruct original base and two SFT LoRA cartridges: V-G02-A and V-G09-C. Same immutable native weights and Village-001 SFT release/directory hashes as the original smoke. There is **no new training**, no NCR experimental NCA hook and no Sprocket-500M model substitution.
- Two exact same Bay 17 witness situations from the original experiment. For each lane/situation, two first-turn prompts: **JSON** (byte-for-byte same structured model-facing messages as initial smoke) and **natural** (case ID, format version, exact admitted fact IDs/text and identical first player question, line-rendered). System message and greedy decode configuration identical across conditions.
- No followups in this A/B. That eliminates variation from previous responses; it is not a demonstration of longitudinal memory. One seed (24301), at most 80 tokens each × 12 = **960 new tokens**, 4 CPU threads, 20-minute job ceiling, 0 API spend and 0 optimizer steps.
- The rendered input contains **no model ID, species, backstory, fault label, other NPC secrets, locked terminal credentials or global truth ledger**. Neither version has an informational advantage. This is an in-family, already-inspected *development* case, not a sealed final case-family holdout.
- All 12 original unmodified replies, special-token decoded text, output token IDs, exact source prompt strings, input token hashes, checkpoint/adapter IDs, CPU, package versions, timing and one-step receipt verification are archived.

## Predeclared descriptive measurements

Primary: fraction of the six responses **beginning with an object opening brace**, separately for JSON and natural presentations. Secondary: exact schema-key echo strings, generated-token count, character/whitespace-word count, latency, and representative verbatim samples. These are **pure presentation diagnostics**, not semantic classification or a trained persona outcome.

A difference, even a strong one, does not prove the JSON formatting caused incoherence; the tiny matched sample and decoding artifacts require replication on other cases and seeds. Interpret low-level echo rate only. **No significance test, model selection or winner designation** is authorized.

## Validation / live run

Offline stdlib-only checks:

    python3 -m py_compile scripts/detective_001b2_presentation.py scripts/detective_001b2_verify.py tests/test_detective_001b2_presentation.py
    python3 -m unittest discover -s tests -p 'test_detective_001*.py' -v
    python3 scripts/detective_001b2_presentation.py --mode check

One PR-only, exact owner/branch-gated GitHub Actions job runs the bound models after offline checks, then invokes **independent** source-match and prompt/echo/record readback. Preserve partial raw records as an Actions artifact even if inference fails. Promote to a **new and disjoint** dataset under datasets/detective-001-presentation-ab, only when complete, hash-bound and checked against the original unmoved PR source head. Never overwrite the first smoke. A rerun only validates existing identical archived evidence, and fails closed on altered code/config.

Do not treat this branch as permission for six-way village maturation, paid teacher generation, general-purpose host scripts, truth-label selection, or scripter fine-tuning. A later Sprocket-500M root Instruct test requires publisher-source-correct model/template pins and an independent source-bound plan.

**Exit criterion:** native inference and all source/evidence checks pass, or preserve a precise failure/blocker. The behavioral admission verdict remains **NOT EVALUATED** regardless of the measured echo frequencies.
