# DETECTIVE-001A — unreliable witnesses and fictional evidence scripting

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103)

Status: **offline contract and synthetic fixture**, not model inference, model training or a playable game. This directory is separate from VILLAGE-001 and NCR-001 source/evidence authorities.

## Objective and game mechanic

Interview a cast of damaged goblin and robot intelligences. Their unreliable memories, lies and obsessive reasoning become useful investigative texture: a player may compare testimony, trade favors, identify motives, exploit behavioral quirks and confront inconsistencies.

**World truth and character claims are deliberately different objects.** A witness's statement may be invented, contradicted, withheld, strategically true or meaningless. No generated statement changes the canonical world timeline or counts as independently verified evidence. A narrative utterance and its evaluative annotation must be preserved separately.

Every *mandatory* lead must have at least two independent sources, including an accessible, non-dialogue evidence route. This permits deception to be interesting without converting a small-model lapse into an unrecoverable detective-game softlock.

## Existing model population and candidate pilot

VILLAGE-001 is **12 Sprocket-derived BABBLE lineages × three independent SmolLM2-135M LoRA siblings** (A failure-hardened; B local tinkerer; C social scavenger), not 36 independently fine-tuned Sprocket-500M weights. The 36-SFT stage #12 is closed; the separate preference and final qualification stages #13/#14 must retain their own original datasets, custody and release gates.

A future pre-registered interview pilot should start with six lineages spanning A/B/C twice (V-G02-A, V-G03-C, V-G05-B, V-G07-A, V-G09-C, V-G12-B), plus independent pinned **Sprocket-500M Chat**, **Sprocket-500M Instruct**, and **void.1** controls, with appropriate same-model prompt-only controls. The selection is a suggested preregistered stratum, not evidence that these six are currently best, or that new fine-tuning has occurred. Keep NCR-001's frozen-adapter experiments wholly separate.

Do not train merely toward maximum literal accuracy. Discriminate these factors in evaluation:

| Mechanism | In-fiction justification | Observable probe |
|---|---|---|
| Time-order confusion | damaged local clock / merged memories | contradictory temporal follow-up |
| Withholding | loyalty, debt or fear | bargain and pressure reversal |
| Self-serving fabrication | black-market status preservation | disconfirming physical evidence |
| Obsessive repetition | stale cache or looping firmware | novel-topic transfer |
| Imprecise speech | damaged language decoder | clue recovery despite phrasing |
| Overconfidence | stripped uncertainty circuit | confidence versus actual knowledge |
| Gossip contamination | crowd-sourced scrap-memory | independence of witness accounts |
| Rare sudden competence | hot-swapped intact subsystem | task-conditional recall |

These phenotypes need character-specific, persistent motivations and social relationships. Goblin/robot species and decorative names are *in-fiction metadata*, not authoritative model identifiers. One model lineage can portray different embodiments only when explicitly tested as a separate condition.

## Contract, authority and example

- `pilot-case.json` stores private canonical facts, actor-specific admitted knowledge, evidence assets, fictional terminals, and lead routes. Its contents are **game authority**, not a model prompt.
- `scripts/detective_001_contract.py:validate_case` checks strict field sets, referential integrity, nonduplicate source routes, and public non-dialogue coverage for every mandatory lead.
- `witness_view` projects **only the selected witness** and their admitted memories, excluding other actors, hidden facts, terminal credentials, evidence catalogues, solved outcomes and evaluation labels. In a full runtime, add game-owned attitude/session and observed player evidence as explicitly versioned admitted fields, not by sharing the whole case.
- For **blind phenotype qualification**, call `witness_view(..., blind=True)`: the model-facing view omits member ID, goblin/robot species, backstory, motives and fault labels. Only the orchestrator's non-model-facing receipt may bind those identities. Use the gameplay projection and blind projection as different, reported test conditions; never describe prompt-injected behavior as learned identity.
- The model's answer is free-form **untrusted witness dialogue**. A separate scoring/annotation pipeline may label propositions, contradictions, sincerity hypotheses and actual facts after generation. Never use a model's self-declared claim IDs as authoritative evidence.
- `parse_program` accepts only `detective-script-v1` containing up to eight typed steps. Allowed operations: `read_asset`, `correlate_assets`, `probe_terminal`. Unknown ops/fields, duplicate JSON keys and overlarge proposals refuse.
- `execute_program` receives game-owned `authorized_asset_ids` and a fictional case. Only the fictional terminal's exact synthetic credential can expand **local** visibility. The returned receipt is deterministic and SHA-256-bound; input world objects never mutate. This design provides no filesystem, OS shell, networking, real credential handling or general Python/JS interpreter to model output. A production adapter will need an authoritative session/state/version, denial and replay protocol of its own.

Example fictional program:

```json
{
  "schema": "detective-script-v1",
  "steps": [
    {"op": "read_asset", "asset_id": "dock-clock-ledger"},
    {"op": "probe_terminal", "terminal_id": "camera-console", "credential": "OXIDATION"},
    {"op": "read_asset", "asset_id": "camera-cache"}
  ]
}
```

The example uses a **synthetic game password**, not a real login or host security test.

## Evaluation, training admission and remaining limitations

Run the offline acceptance suite:

```sh
python -m unittest discover -s tests -p 'test_detective_001_contract.py' -v
```

This test establishes *structural* safety and reproducibility only. It does not demonstrate that any NPC can convincingly lie, remember or disclose facts, or that an instruct model produces valid scripts.

Future inference/training qualification requires:

- Exact model/checkpoint/revision/tokenizer/adapter SHA and generation settings, seeds, runner/device identity and raw, unfiltered answers archived following `docs/DATASET_FILING_SYSTEM.md`.
- Case-family-disjoint held-out scenarios, including all follow-up interrogation turns. Do **not** reuse Village's sealed F-suite as a training set.
- Independent scoring of persona distinction (raw and style-neutral), motives, lie/withholding policy, contradiction dynamics, uncontaminated knowledge, hallucination type, interrogation recovery and **mandatory clue recoverability**. Avoid rewarding random inaccuracy with no character motive.
- Fictional scripter metrics: strict JSON/IR validity, successful evidence processing, correct access denial, refusal/recovery after authoritative rejection, resource use and decision latency. Include weak prompt-only and deterministic baseline controls.
- Before launching remote generation or new training, approve a finite compute/API budget, exact admitted model pins, dataset licensing/provenance, success thresholds, halt conditions and host access. Preserve the existing Village stages. Remote professor/critic outputs are not truth authorities.
- This fixture's `public` flag proves an intended accessible recovery route *in the schema*. A production case graph must prove actual physical availability and progression reachability; the game must not withdraw all public evidence through later gameplay. A true game integration also needs turn/session IDs, actor belief updates, authoritative evidence unlock transactions and replayable interview records.

The useful end state is a solver where every player interview is interesting even when the NPC is wrong; verified evidence, access permissions and eventual case conclusions still derive only from the game world.
