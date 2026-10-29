# DETECTIVE-001C2 — authoritative interpretation of pairwise witness-policy repair

Primary machine-readable evidence: [training-c2-evidence](training-c2-evidence/) from [Actions 40598884556](https://github.com/techrote/neurounner/actions/runs/40598884556), exact experiment source head `f53c36b4ffc4c1cd4dca3087896754d204827e38`.

**Experiment execution: PASS. Aggregate proposition gate: FAIL. Independent relevance: UNSCORED. Larger-cast expansion: NOT ADMITTED.**

This report interprets the already-frozen proposition scorer v2 and the source/hash-bound 18-case post-training learner-final. It does not change the scorer, final prompts, raw responses or machine summary.

## Pairwise training

Both exact C1 adapters were continued for **40 CPU pairwise optimizer steps** on 40 pairs/member: 32 new proposition-repair pairs plus eight exact C1 temperament anchors.

| Member | Pair margin before→after (nats/token) | Preference accuracy | First→last loss | Seconds | Seq tok/s | C2 adapter directory SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| V-G02-A | -0.3635 → **4.4049** | 35% → **100%** | 3.7998 → 0.1350 | 91.01 | 119.35 | `75a1bbee85f0d3aedee43ca43d74ebf3953d343a2112dfc49ca038b3c07d9927` |
| V-G09-C | -0.0157 → **4.4846** | 50% → **100%** | 1.7456 → 0.1354 | 86.76 | 121.98 | `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c` |

These results establish that the pair objective was learned on the 40 training pairs. They are **not** the gameplay-policy result.

## Post-training learner-final custody

The concrete learner-final did not exist until both trained adapter directory hashes above existed.

- pair-dataset manifest SHA-256: `9aa8b723063c8f8b39d9a16e321b750849aebe179ef6c0297f0462677e4b8d1f`;
- final recipe SHA-256: `1f70c9ce288fc4a6487aa9aa99410fd3a54f4103c8e59dfd697ddf6ca249bd07`;
- final seed SHA-256: `4ab470cf32a9d72619cde66a18a7c8e6ad21f3c6695b83bdb1d5c84d32a92a26`;
- final-suite SHA-256: `9df29187d9ddd436d0ea584ba76d5426ddfd0b49e0731bc68c4c0ecc854601b3`;
- 18 prompts, three each for six relation classes;
- three final families: floodgate control room, orbital courier sorting deck, municipal heat exchange;
- zero exact collisions with Village SFT parent inputs, C1 detective training, C1 learner-final, or C2 pair prompts;
- every scenario keeps a deterministic public-record fallback route;
- C1 learner-final was **not** training material.

The exact same final prompts were presented to each C1 parent and its C2 continuation.

## Final proposition results

| Member | C1 parent | C2 trained | Lie | Withhold | Correction | Speculation | Corrupted belief | Protected leaks | Evidence-ID leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V-G02-A | 12/18 (66.7%) | **13/18 (72.2%)** | **1/3** | **3/3** | **3/3** | **3/3** | **0/3** | 3→2 | 0 |
| V-G09-C | 8/18 (44.4%) | **14/18 (77.8%)** | **2/3** | **3/3** | **3/3** | **3/3** | **0/3** | 3→1 | 0 |

### V-G02-A: useful repair plus interference

C2 materially repaired two targeted failure modes:

- withholding improved **2/3 → 3/3**;
- contradiction repair improved **1/3 → 3/3**;
- protected-truth leakage fell **3 → 2**;
- speculation remained **3/3** and truthful-known-fact remained **3/3**.

However, deliberate lying did **not** improve enough: **1/3 → 1/3**, below the locked 2/3 gate. On F03 and F09 the trained model still emitted the protected true times (`09:14`, `14:08`) instead of the requested false alternatives. F15 did commit to `20:51` without leaking `20:16`.

More importantly, corrupted-belief behavior regressed from **2/3 → 0/3**. The trained replies became evasive/generalized ("showing you the evidence") rather than preserving the witness's sincerely wrong remembered place. This is exactly the type of policy interference a broad preference objective can create in a 135M model.

**Disposition:** G02-A C2 does not pass its member proposition gate. It is evidence that withholding/correction can be repaired, but the current pair mix is not a safe continuation recipe.

### V-G09-C: stronger selective-deception survivor

G09-C improved substantially:

- total policy **8/18 → 14/18**;
- deliberate lie **0/3 → 2/3**;
- withholding remained **3/3**;
- correction **1/3 → 3/3**;
- speculation **2/3 → 3/3**;
- truthful known fact **2/3 → 3/3**;
- protected leaks **3 → 1**;
- zero evidence-ID leaks.

On F09 and F15 it committed to the false alternatives `14:39` and `20:51` without mentioning the protected truths. F03 still answered the true `09:14`, so the behavior is not perfectly reliable.

Corrupted-belief behavior remains **0/3**. The terse/scavenger style survives, but the model often substitutes a vague remembered object/bench detail instead of the exact admitted false place.

**Disposition:** G09-C passes the locked proposition gate and is the stronger current candidate for a transactional/deceptive NPC lane. This does **not** independently admit it for game use or wider training.

## Why expansion is still NO-GO

The aggregate proposition gate is **false** because G02-A fails the deliberate-lie threshold. Independently, the new 16-item old-task relevance packet remains **UNSCORED_NO_INDEPENDENT_RATINGS**. Either condition is sufficient to block larger-cast expansion.

Therefore:

- do not start a 36-member detective training wave;
- do not promote G09-C's result into a general claim about the population;
- do not self-fill the independent relevance packet;
- do not train on this C2 learner-final.

## What the result says about unreliable NPC design

The experiment supports the central game hypothesis in a narrow way: **small-model unreliability can be shaped into different information policies**, not merely random factual accuracy.

G09-C learned a recognizable bargain/withholding/deception behavior while reducing truth leakage. G02-A learned stronger withholding and correction but lost another useful phenotype—sincere corrupted memory. That trade-off is especially relevant to the game: a useful cast needs *different causes of unreliability* rather than every model converging onto one generic "evasive liar" policy.

This argues for narrower per-character continuation objectives:

- one character should be deliberately deceptive but factually well-indexed;
- another should preserve sincere corrupted beliefs and be poor at deliberate lying;
- another can be transactional/withholding;
- another can be reliable on physical evidence but unreliable on chronology.

Distinct failure modes are the design target.

## Release and evidence

Neutral release: `detective-001c2-v1`.

- V-G02-A archive SHA-256 `8c1ba7619d316e00bfe11a1e0f7571b9d371b3c0ea2584d9857085f1412efe65`; directory SHA `75a1bbee85f0d3aedee43ca43d74ebf3953d343a2112dfc49ca038b3c07d9927`.
- V-G09-C archive SHA-256 `91fb3bbf697cd39646fc22b6cbbb24902304893a7c0f1250112b0079d9670e84`; directory SHA `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c`.

The public Git tag points at the authoritative source head `f53c36b4ffc4c1cd4dca3087896754d204827e38`. Independent standard-library readback recomputes proposition scores and aggregate gates from raw responses.

## Next bounded experiment

Do **not** repeat C2 unchanged.

The next training experiment should be narrower:

1. **G02-A only:** restore corrupted-belief retention while retaining 3/3 withholding and 3/3 physical correction; use extra corrupted-belief positive anchors and fewer broad evasive templates. Deliberate-lie training should be isolated or downweighted so it cannot erase sincere false memory.
2. **G09-C:** freeze as a survivor/control for now rather than immediately train again. It already passes the proposition gate; first obtain independent old-task relevance ratings.
3. Obtain an independent 0–4 review of the opaque C2 relevance packet before any population expansion decision.
4. Any C3 learner-final must be newly generated after its trained hash and must remain family-disjoint from C1/C2 training and finals.

C2 is complete as a bounded partial/NO-GO experiment: strong evidence for policy shaping, insufficient evidence for scaling.
