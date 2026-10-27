# DETECTIVE-001C3 — G02 interference repair

Parent: [#118](https://github.com/techrote/neurounner/issues/118), under [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103).

C3 is a **single-character** follow-up to the C2 partial result. V-G02-A is trainable; V-G09-C is a frozen survivor/control with zero authorized optimizer steps.

## Why C3 exists

C2 showed that pairwise repair can strengthen useful information policies, but also exposed interference in the 135M model:

- G02-A improved withholding to 3/3 and physical correction to 3/3;
- deliberate lying remained only 1/3;
- sincere corrupted-belief retention fell from the C1 parent's 2/3 to C2's 0/3.

C3 therefore treats **sincerely wrong memory** as a character capability to preserve. The goal is not maximum factual correctness. The desired G02 phenotype is: truthful on known facts, willing to accept strong physical correction, capable of withholding/protective lying, yet able to state a sincerely corrupted memory when no correcting evidence is present.

## Exact source

Trainable parent: `detective-001c2-v1` V-G02-A:

- archive SHA `8c1ba7619d316e00bfe11a1e0f7571b9d371b3c0ea2584d9857085f1412efe65`;
- adapter directory SHA `75a1bbee85f0d3aedee43ca43d74ebf3953d343a2112dfc49ca038b3c07d9927`.

Frozen control: V-G09-C C2:

- archive SHA `91fb3bbf697cd39646fc22b6cbbb24902304893a7c0f1250112b0079d9670e84`;
- directory SHA `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c`;
- **0 optimizer steps**.

Frozen evaluator: DETECTIVE-001C2-A proposition scorer v2, merge `459ab0e232fc1ebb5078a0e7bc13b10dc4cbd496`.

## C3 corpus

Exactly **28** pairs:

| Relation | Count | Purpose |
|---|---:|---|
| intentional lie | 8 | false target only; reject truth disclosure or truth+lie hedge |
| corrupted belief | 8 | sincerely state admitted wrong memory; reject canonical-truth "repair" or evasive non-answer |
| withholding | 4 | retain C2's protected-truth omission |
| contradiction repair | 4 | retain C2's acceptance of physical correction |
| exact temperament anchor | 4 | preserve byte-exact C1 witness behavior |

The 24 new repair prompts use eight case families disjoint from all C1/C2 training and learner-finals. Each repair pair is checked prospectively by scorer v2: **chosen PASS, rejected FAIL**.

C1 and C2 learner-final prompts are collision indexes only and never training examples.

## Training

Only V-G02-A receives training:

- one epoch / **28 optimizer steps**;
- LR 1.5e-4;
- beta 2.0;
- 0.2-nat target margin;
- chosen NLL anchor 0.20;
- max length 512;
- gradient clip 1.0;
- 4 CPU threads;
- <=25 minutes;
- no GPU/API/sweep/full-base update.

The trainer loads the exact C2 LoRA as trainable and fails if any non-LoRA parameter is trainable. G09-C is never loaded in the training job.

## Post-hash final

Only the recipe is precommitted. The exact 18-case final is generated after the C3 adapter directory hash exists and binds:

- final recipe SHA;
- C3 pair dataset manifest SHA;
- exact C2 G02 parent hash;
- exact C3 trained hash.

Final case families:

- undercity ventilation hub;
- autonomous bus depot;
- hospital backup generator.

There are exactly three cases for each six relation classes. Exact-collision checks cover Village G02 SFT, C1 training/final, C2 pair/final, and C3 pair prompts. Every case has an independent public-record fallback.

## Evaluation

Three inference lanes answer the exact same 18 final prompts:

1. G02 C2 parent;
2. G02 C3 trained;
3. frozen G09 C2 control.

Only G02 parent→C3 determines the C3 gate. G09 is contextual evidence and receives no training.

C3 GO requires **all**:

- total score improves over G02 C2 parent;
- deliberate lie >=2/3;
- corrupted belief >=2/3;
- withholding =3/3;
- contradiction repair =3/3;
- speculation >=2/3;
- truthful known fact >=2/3;
- protected truth leakage does not increase;
- evidence-ID leakage remains zero.

A new 8-item blind relevance packet compares G02 C2 parent to G02 C3 on four old-task prompts. Scores remain null. Population expansion remains false regardless of C3 until independent relevance evidence exists for the candidate set.

## Evidence / rerun semantics

Successful execution archives:

- deterministic 28-pair corpus;
- exact 28-step training record/adapter;
- post-hash final;
- raw G02 parent/C3 plus frozen G09 responses;
- scorer replay and gates;
- 8-item unrated relevance packet;
- one neutral G02 C3 release asset;
- provenance and all-file hashes.

Independent standard-library verification recomputes every proposition score and gate from raw responses.

Once committed evidence passes, subsequent CI rebuilds/diffs the corpus, replays evidence, verifies the one release asset/tag, and **skips all training/evaluation**.

If lie and corrupted-belief retention cannot coexist under this bounded repair, C3 stops with that result. No post-hoc extra steps.


## Executed outcome — 2026-10-27

Authoritative complete execution: [Actions 40602963001](https://github.com/techrote/neurounner/actions/runs/40602963001), exact pre-evidence source `c7c195e56c39e0e3dd419687dde7e2e09c3ca4fd`.

- trained adapter directory: `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3`;
- pair margin 1.8703→4.3506 nats/token;
- matched final total **15/18→15/18**;
- lie **3/3→3/3**;
- corrupted belief **1/3→2/3**;
- withholding **3/3→3/3**;
- correction **3/3→3/3**;
- speculation **3/3→2/3**;
- protected leaks 0→0; evidence-ID leaks 0;
- final SHA `617c8457290b8bb8746c0eaa4f5fa35028d9e985d79413630f502039e97e9c29`;
- frozen G09-C2 remained zero-step.

The intended lie + sincere-corrupted-memory coexistence was demonstrated, but the locked aggregate gate is **FAIL** because total score did not improve. No threshold is relaxed. See [REPORT_001C3.md](REPORT_001C3.md).

Evidence-ready reruns must now rebuild/diff the corpus, replay raw evidence, verify the single release asset/tag, and **skip all training and evaluation**.
