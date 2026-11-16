# Goal/recovery curriculum — executed synthetic CPU transfer (16 November 2026)

**Draft/HOLD; research only. No native/game modifications.**

The new recovery_curriculum.py is an executable research-only teacher/student harness extending the source-pinned closed_loop_transfer_probe.py at commit c8263253. It uses synthetic 96×96 host-legal observations, local materials, protected cells, goal deltas, action history and STOP/RETRACE. The oriented geodesic teacher uses privileged full-world data **only to produce labels**; no privileged map or distance field enters the student prompts.

## TRAIN-only synthesis and source pins

- Prior v1 TRAIN archive SHA-256: 80b11c731e29306dd04c0b09539ab8fc7796592933276a13e7e44e83f17f6dbc. The earlier 4,313 TRAIN-only examples train the comparison baseline.
- NEW recovery TRAIN seeds 1000–1119: 120 procedural world groups, three family-labelled episodes per group, 192 steps maximum, 360 genuine teacher-directed synthetic episodes.
- 32,630 teacher-labelled states, including 5,299 host-applied off-policy perturbations that generate recovery observations, 91,898 candidate legal distractors and 351 RETRACE targets.
- 3,073 near-duplicate state observations (cross-family normalised) rejected; 29,557 remaining model-ready TRAIN messages in reproducible ZIP with indexed SHA-pinned provenance, explicit split labels, hard-negative channel and world IDs.
- TRAIN ZIP SHA-256: **e8693c43b495c4f0a94d056d871f3e9174a6eb8110660404b2ef0cc64bb3d29d** (locally verified, not yet published as immutable GitHub Actions artifact).
- Real CPU SGD/log-loss fitting: 24 epochs, optimizer counter 783,121, 365 features. No checkpoint binary in Git.

## New held-out-seed closed-loop synthetic evaluation

24 distinct seeds (61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179); 3 family labels, 3 policies, 192-decision budget = **216 actual host-applied inference trajectories**. These world IDs are disjoint from TRAIN, although they share the same procedural generator. Not official DEV, independent generator or native evidence.

| Policy | Goal arrivals (72 runs) | Successful worlds (24 seeds) | Mean unique cells | Mean RETRACE |
|---|---:|---:|---:|---:|
| New goal/recovery-trained CPU SGD | **60** | **20** | 63.458 | 12.375 |
| Existing heuristic-trained CPU SGD | **0** | **0** | **113.514** | 0.278 |
| Seeded legal random | **0** | **0** | 94.444 | 41.931 |

The three labelled family rollouts per world are correlated; **20/24 world-level** success is the more defensible denominator. Successful arrivals average 66.2 decisions. Four unsolved seed worlds: 89, 113, 139 and 167. Existing heuristic student covers more area but commonly stops before a goal. The new curriculum materially changes objective-directed learned behaviour; no claim of improved style, aesthetics, native WORM capability or Sprocket/LFM checkpoint transfer.

## Qualifications, reproducibility, delivery

- Five focused local tests PASS: oriented teacher legality, immutable protected cells, digest-pin refusal, world split overlap rejection, dedup/export integrity and repeated byte-stable exports.
- Full original local experiment repeated independently with identical report SHA-256 **fbd0a326ed0e62eae68066a2364ae577e310a6c3838236104b1455ad85401eb8** and identical TRAIN ZIP SHA-256 above. Eighteen original matched PNGs were produced; each has an SHA-256 inventory entry.
- Prior source import: PR #638 at c8263253. Exact Python research source and tests are committed under experiments/driv3r/corpus_factory/**; no production paths touched.
- Artifacts are local verified exports, **not GitHub Actions artifacts**; hosted qualification and durable artifact publication remain outstanding. No immutable optimizer checkpoint was persisted or promoted.

**Trainer handoff:** Treat the goal/recovery dataset as a separate, TRAIN-only candidate, not an artistic-style curriculum. Checkpoint owners should compare equal-seed goal progress, recovery and motifs independently, avoid evaluator leakage, and keep primary optimizer leases and native host authority unchanged. Never train on W0 quarantined reference or new held-out rollouts.
