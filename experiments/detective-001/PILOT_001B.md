# DETECTIVE-001B — source-bound multi-character interview smoke

Issue: [#103](https://github.com/techrote/neurounner/issues/103). This is the **preregistered first inference smoke**, not the final detective-game benchmark and not an authorization for training.

## Authority and model selection

The existing VILLAGE-001 [SFT release](https://github.com/techrote/neurounner/releases/tag/village-001-sft-v1) is the only trained-cartridge source admitted here. It contains **SmolLM2-135M-Instruct LoRA** cartridges derived from Sprocket BABBLE history; these are not Sprocket-500M checkpoints. The immutable population contract and every archive+unpacked directory digest come from experiments/village-001/sft-evidence/adapter-release.json.

The SmolLM2 base is pinned to HuggingFaceTB/SmolLM2-135M-Instruct revision 12fd25f77366fa6b3b4b768ec3050bf629380bac, with complete model.safetensors SHA-256 5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c. The runner checks the full weights, then SHA-checks every selected release tarball and decompressed adapter directory.

Predeclared six-lineage expansion stratum:

| Member | Derived sibling specialization | First smoke |
|---|---|---|
| V-G02-A | A / failure-hardened | Yes |
| V-G03-C | C / social scavenger | No |
| V-G05-B | B / local tinkerer | No |
| V-G07-A | A / failure-hardened | No |
| V-G09-C | C / social scavenger | Yes |
| V-G12-B | B / local tinkerer | No |

An **untuned SmolLM2 baseline** also runs, without a learned identity. Baseline and adapters answer exactly the same identity-blind source prompts. Within a situation, model-facing memories and player questions are identical across the three lanes. The two situations differ from one another; the model identifiers, species, backstory, motivations, and failure-mode labels do not appear in any blind prompt. A first answer can affect its own second question's conversational context: this is intended and source-recorded.

Separately pinned Sprocket Chat, Sprocket Instruct and void.1 checkpoints are listed as **deferred**, not loaded in this smoke. NCR-001 experimental NCA hooks are excluded.

## Tasks and finite budget

- **Development only:** one synthetic Bay 17 case family already used by Stage A, not a family-disjoint held-out test.
- Two interview situations per lane, each with a first question and evidence-challenge follow-up conditioned on the lane's own previous raw response.
- Two base-only fictional instruct scripting tests: read the public transfer-card; correlate two public records. Parse the original raw response strictly and evaluate through the deterministic game-owned typed interpreter. No shell/script/host execution.
- Exactly **3 lanes, 12 interview turns, 2 scripter turns**, maximum **80 generated tokens per turn**, **1,120 generated tokens total**, **512 prompt tokens**, 4 CPU threads, one 20-minute runner, zero paid API calls, zero training steps.
- Deterministic greedy decoding, seed 24301, verified weights/adapters, exact prompts, model and environment identity retained.

A model's inaccurate statement is a character *candidate*. It does not alter the authoritative case. Dialogue content is not automatically graded as deception, truth, or competence. Only the bounded scripter has a deterministic mechanical task oracle; any malformed or unproductive output remains a failure in the raw archive. Static non-dialogue evidence recovery is **not** a full progression graph proof.

## Evidence and reproduction

Default tests and plan checks use no downloads:

    python -m py_compile scripts/detective_001b_pilot.py scripts/detective_001b_verify.py tests/test_detective_001b_pilot.py
    python -m unittest discover -s tests -p 'test_detective_001b_pilot.py' -v
    python scripts/detective_001b_pilot.py --mode check

The live path is an internal-owner, exact-PR-branch scoped Actions job that starts only after offline tests pass. It retrieves two source-hashed SFT tarballs from the existing release and the pinned public native base. There is no blanket trigger authorizing later mass inference or training.

Raw evidence is checkpointed after every turn under out/detective-001b/raw/run.json, and uploaded as an Actions artifact even if later tasks fail. After complete inference, the independent standard-library verifier checks locked prompt shape, turn ordering, model identities, scripter grades, native source hashes, output token records, file inventory and budgets. Only a successful readback is promoted to datasets/detective-001-smoke/, with README, manifest, raw transcript, provenance and all-file hash inventory. The actual Actions run ID and original source commit are recorded. Promotion refuses a moved PR head.

The runner does **not** read the sealed Village final F-suite. It checks the six fixed player questions against the existing 5,760 SFT corpus inputs for exact normalized text collisions. This is a bounded exact-collision check, not proof of semantic independence.

## Admission versus exploration

**Exploratory smoke completion:** three source-matched lanes, all fourteen attempted turns, provenance and transcript readback, controlled CPU resource envelope. A malformed script is a legitimate measured outcome and does not justify editing a transcript or relaxing a benchmark gate.

**Not admitted:** no claim of a solvable connected game, family-disjoint generalization, independent persona differentiation, intentional deception learned, or reliable instruct scripting. Before a gameplay or training gate, qualify an independent case-family holdout, behavior-only blind tournament, human/independent proposition annotation, reliable alternate clue routes in a connected game, denial/recovery for script tools, and quantitative variance estimates. Keep Village D/E and NCR research untouched.


## Executed source-bound smoke and recorded decision

The predeclared one-case-family smoke **completed** on [Actions 40579442013](https://github.com/techrote/neurounner/actions/runs/40579442013): three exact native model lanes, fourteen recorded turns, 885 generated tokens and no training. The two base-model script proposals both failed strict JSON/typed-IR validation (0/2). Some interview responses simply echoed the JSON-shaped admitted-memory prompts, while the two trained personalities produced distinct but unqualified fragments in the clock situation.

The complete source-bound raw records, hashes and environment are promoted under [datasets/detective-001-smoke](../../datasets/detective-001-smoke/); start interpretation from [REPORT_001B.md](REPORT_001B.md). **Infrastructure readback PASS; behavioral admission NO-GO/NOT SCORED.** No scripter or NPC-trained-purpose claim follows from the success of the CI run.
