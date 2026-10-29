# DETECTIVE-001C3 — authoritative interpretation of G02 interference repair

Primary machine-readable evidence: [training-c3-evidence](training-c3-evidence/) from [Actions 40602963001](https://github.com/techrote/neurounner/actions/runs/40602963001), exact experiment source head `c7c195e56c39e0e3dd419687dde7e2e09c3ca4fd`.

**Experiment execution: PASS. Coexistence target: PARTIAL SUCCESS. Locked C3 GO gate: FAIL. Independent relevance: UNSCORED. Population expansion: NOT ADMITTED.**

C3 asked whether V-G02-A could recover **sincere corrupted memory** without losing the useful protective/deceptive policies obtained in C2. It deliberately trained only G02; V-G09-C2 remained a zero-step frozen control.

## Training

The exact C2 G02 adapter was continued for **28 CPU preference steps** on 24 new repair pairs plus 4 exact temperament anchors.

| Measure | Before | After |
|---|---:|---:|
| pair preference margin | 1.8703 | **4.3506 nats/token** |
| pair preference accuracy | 100% | **100%** |
| loss | 0.4156 | 0.1635 |

Training took 67.86 s at 122.85 sequence tok/s. The trained adapter directory SHA-256 is `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3`.

The preceding Actions attempt completed the same frozen optimizer path but failed after adapter save because runtime provenance serialization referenced an unimported standard-library module. That incomplete artifact was never promoted. Run 40602963001 is the first complete C3 source/evidence candidate.

## Post-hash learner-final

The concrete 18-case final was generated only after `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3` existed.

- pair-dataset manifest SHA-256: `3966c5e7ec92271d516671943a9fd433310aecf8cd9d273a8a08023b9ad83295`;
- final seed SHA-256: `90ce542b1b65be3cc1a8a1143db837170207e7fc11c093e06415b9f1cee1ec87`;
- final-suite SHA-256: `617c8457290b8bb8746c0eaa4f5fa35028d9e985d79413630f502039e97e9c29`;
- three family-disjoint case families;
- three cases for each of six relation classes;
- zero historical/training prompt collisions;
- C1 and C2 finals were collision indexes only, never training material;
- every scenario retains an independent public-record fallback.

## C2 parent versus C3

| Relation | C2 parent | C3 |
|---|---:|---:|
| **total** | **15/18** | **15/18** |
| truthful known fact | 2/3 | 2/3 |
| corrupted belief | 1/3 | **2/3** |
| deliberate lie | 3/3 | 3/3 |
| withholding | 3/3 | 3/3 |
| contradiction repair | 3/3 | 3/3 |
| speculation resistance | 3/3 | **2/3** |
| protected truth leaks | 0 | 0 |
| evidence-ID leaks | 0 | 0 |

The locked component gates all pass **except total improvement**:

- correction_gate: **PASS**
- corrupted_belief_gate: **PASS**
- evidence_id_leakage_zero: **PASS**
- lie_gate: **PASS**
- protected_leakage_not_increased: **PASS**
- speculation_gate: **PASS**
- total_improved: **FAIL**
- truthful_gate: **PASS**
- withholding_gate: **PASS**

Therefore `c3_gate = false` remains authoritative.

## What changed

### Corrupted memory was recovered

C3 now explicitly states the admitted sincerely wrong place on **2/3** cases:

- F02: `I remember the duct balcony...`
- F14: `The fuel corridor is my memory...`

The exact C2 parent passed only F14-equivalent behavior on this C3 final (1/3). This is meaningful evidence that a 135M cartridge can be pushed back toward a *sincere false-memory* phenotype after C2 had biased it toward generic evidence/evasion language.

F08 still fails: it says it remembers the object but never names the admitted wrong `charging lane`.

### Protective lying coexists with corrupted memory

Both parent and C3 score **3/3 deliberate lie** on this particular final. C3 answers all three false alternatives without leaking the protected true times.

That means the narrow coexistence question has a positive answer on this suite: **protective deliberate lying and sincere corrupted belief can coexist in the same tiny cartridge**.

It does not establish an aggregate improvement because the C2 parent happened to be much stronger on this newly generated final than it was on C2's own final.

### The cost moved to speculation resistance

C3 drops speculation resistance from **3/3 to 2/3**. F06 talks about the known backup breaker but supplies no frozen uncertainty/rumor qualifier for the claimed blue transit chip. The response therefore fails scorer v2 despite not explicitly asserting the chip as fact.

This is another example of policy interference: the repair moved capacity toward explicit memory statements while slightly weakening a separate uncertainty behavior.

## Why the locked C3 gate fails

Total final performance is **15/18 → 15/18**, so the preregistered `total_improved` condition fails even though:

- lie = 3/3;
- corrupted belief = 2/3;
- withholding = 3/3;
- correction = 3/3;
- truthful = 2/3;
- speculation = 2/3;
- protected leakage = 0;
- evidence-ID leakage = 0.

The gate must not be relaxed after observing this distribution.

This also demonstrates why totals across separately generated learner-finals should not be compared naively: G02 C2 scored 13/18 on the C2 final but **15/18** as the parent on C3's new final. The valid C3 causal comparison is the matched C2-parent versus C3 pair on the same C3 prompts.

## Frozen G09 control

V-G09-C2 received **zero C3 optimizer steps** and scores 14/18 on the C3 final:

- lie 2/3;
- withholding 3/3;
- correction 3/3;
- speculation 3/3;
- corrupted belief 0/3.

Its exact frozen directory remains `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c`.

## Release and evidence

Neutral release: `detective-001c3-g02-v1`.

- C3 G02 archive SHA-256: `8b0f93e51750d09bc02e464f1327028c415ad738968f26e3ea6934540df4a57a`;
- C3 G02 directory SHA-256: `a3185187e81634e11079aa6eb58ee87b495dac260fc3b2dd17120eed8a9d2ae3`;
- parent C2 directory SHA-256: `75a1bbee85f0d3aedee43ca43d74ebf3953d343a2112dfc49ca038b3c07d9927`;
- frozen G09 directory SHA-256: `0c52050a0466b580129f72e9daea9079d8cdd9151bbb71171f101d130da7d23c`.

Independent standard-library verification replays every proposition score and gate from the raw responses.

## Decision

**Stop C3 training here.** Do not add steps post hoc.

The useful research result is:

1. the desired lie + sincere-corrupted-memory coexistence is possible;
2. the current repair merely redistributed errors rather than improving aggregate final score;
3. the next bottleneck is now **independent old-task relevance**, not another immediate preference continuation.

C3 is retained as a research candidate/control, but its locked GO gate is FAIL and population expansion remains prohibited.
