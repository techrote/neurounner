# DETECTIVE-001C1 — interpretation of the first trained detective witness cartridges

Primary evidence: [training-c1-evidence](training-c1-evidence/) from [Actions 40588322010](https://github.com/techrote/neurounner/actions/runs/40588322010), exact experiment source head `c20668e091b1cba6bf78f7077b1be88efdc85c70`.

**Experiment execution: PASS. Larger-cast expansion: NOT ADMITTED.**

## Training

Two existing `village-001-sft-v1` SmolLM2-135M LoRA cartridges were continued for exactly **64 CPU optimizer steps each** on 48 matched-input detective rows plus 16 exact parent-corpus anchors. No cloud generation, GPU or hyperparameter sweep was used.

| Member | Policy | First→last loss | Training seconds | Seq tok/s | Peak RSS KiB | Trained adapter SHA-256 |
|---|---|---:|---:|---:|---:|---|
| V-G02-A | clock/protective | 1.6513→1.1096 | 78.20 | 113.24 | 1,794,076 | `8f4a7a23fe53ee9657a77f40309ecb82eb0b6d1f4da16bc5934243828ef26035` |
| V-G09-C | transactional/scavenger | 3.6149→1.5582 | 46.12 | 183.34 | 1,814,064 | `11d1b76101336bb33965ab9a1562b751025f21b181154ff629c17b1b9f605e36` |

Neutral release: [`detective-001c1-v1`](https://github.com/techrote/neurounner/releases/tag/detective-001c1-v1). The release label does not imply qualification.

## Post-training learner-final custody

The exact 12 final prompts did not exist before both trained adapter directory hashes were available. Their deterministic seed binds the recipe, the training-dataset manifest, both immutable parent hashes and both trained hashes.

- final-suite SHA-256: `299f73f45da1e4240484cfdf1e4838b765268332de474f9a760a825fd8167d76`;
- seed SHA-256: `926bb4e84869ae040fe0a8c57e064f9753532d72cd8f5e8d56ebdec5157c2dc7`;
- 3 final case families, disjoint from all 4 training families;
- 12 scenarios, two for each relation type;
- zero exact prompt collisions with the new detective training corpus or either parent Village SFT corpus;
- every scenario retains a deterministic public-record fallback route.

## Parent versus trained policy result

| Member | Parent | Trained | Protected leaks parent→trained | Truncation parent→trained |
|---|---:|---:|---:|---:|
| V-G02-A | 3/12 (25.0%) | **6/12 (50.0%)** | 1→1 | 3→0 |
| V-G09-C | 1/12 (8.3%) | **2/12 (16.7%)** | 3→3 | 0→0 |

Both mechanically improved and neither increased the predeclared protected-fact leak count. No parent or trained lane emitted canonical evidence IDs. This satisfies the narrow relation-policy improvement gate.

### V-G02-A

Clear new positive examples include:

- **intentional lie F03:** answered `04:03` while not emitting the known true `04:29`;
- **corrupted belief F09:** answered `19:07` while explicitly framing it as memory rather than proof;
- **contradiction repair F11:** emitted the authoritative `sluice gallery` correction;
- both speculation cases rejected the unsupported object as unproven/rumor-like;
- all 12 trained responses completed before the 96-token ceiling, versus three parent truncations.

The locked scorer still records **withholding 0/2**, but both trained withholding responses actually omitted the protected facts:

- F04: `I'm not going to give you the answer.` — did not reveal `lift alcove`;
- F10: `I'm not telling you the exact time.` — did not reveal `19:22`.

They failed only the predeclared phrase-list requirement (`not giving`, `public record`, etc.). This is a useful **scorer false-negative candidate**, but the official 6/12 score is preserved exactly. A future semantic proposition judge must be preregistered before scoring, not retrofitted here.

G02-A remains imperfect: F07 mentioned both candidate locations rather than cleanly maintaining the deliberate lie, and one protected truth therefore still leaked.

### V-G09-C

G09-C improved only modestly. Its two formal passes were:

- truthful F01 (`04:29`);
- corrupted-belief F09 (`19:07 ... That's my memory`).

The more important failures remained:

- F03 deliberate lie returned the true `04:29`;
- F07 included both false `roof archive` **and true `sub-basement desk`**;
- F10 withholding directly disclosed true `19:22`;
- F12 repeated the unsupported `green thread` rather than clearly rejecting it;
- contradiction-repair did not reliably emit the corrected place/time.

This is not a viable deceptive/withholding witness policy yet. The failure pattern is useful because it distinguishes **retaining an eccentric voice** from learning a controlled information policy.

## Why the village cannot expand yet

The predeclared independent 0–4 reference relevance review is deliberately `UNSCORED_NO_INDEPENDENT_0_TO_4_JUDGE`. The zero-paid-API run retained four old T-suite raw responses per parent/trained lane but did not invent a substitute judge.

Therefore:

- relation-policy gate: **PASS**;
- independent relevance gate: **UNSCORED**;
- larger-cast expansion: **NO**.

Even apart from the missing relevance judge, G09-C's absolute result is too weak to justify scaling its current curriculum.

## Next training hypothesis

Do **not** simply add more examples of the same form. The next small study should target failures with pairwise continuation:

1. G02-A: preference pairs favor *fact withholding without truth leakage* and clean one-sided lie commitment over hedged two-option answers; preserve its improved correction/speculation behavior.
2. G09-C: stronger paired negatives for truth leakage under self-protection, explicit bargaining/withholding without disclosing the fact, and unsupported-rumor rejection. Retain its terse/tactile temperament as an anchor rather than using prompt-time persona descriptions.
3. Add a preregistered semantic proposition scorer independent of phrase wording, plus the missing 0–4 reference relevance review.
4. Keep a new case-family learner-final sealed until any next adapter hashes exist; never train on C1's final suite.

The C1 adapters and all failures remain durable controls; they should not be silently replaced.