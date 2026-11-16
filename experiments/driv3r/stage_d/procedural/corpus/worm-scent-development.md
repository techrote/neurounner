# D-2 — paired steering-scent ablation (DEVELOPMENT, 15 November 2026)

**Question.** For the owner's “slightly deranged, steerable drawing worm,” does the existing deterministic wayward controller visibly respond to soft scalar cues, and when does that response collapse into retracing? This is a character/controllability study, **not** path-optimality or structural-strength optimization.

**Design.** Six matched 80×80 scenes (open/slats/ring × seeds 23/79), each with maximum 192 cardinal moves, identical hard permissions, start, deterministic random sequence and pen cadence. Six arms: uniform **neutral**, original **attract ×1**, original with **gain ×4** and **×16**, inverse **repel**, and horizontally mirrored scent. Gain variants explicitly alter the controller weight; they are not equivalent-input map controls. The scorer is a strict fork of previous synthetic wayward v1: the ×1 arm matches all old events, trajectories and final material sets on all six scenes. The preceding 18-case corpus and its source bytes remain unchanged.

Each of **36 episodes** records the complete original mask/PNG, full u16 scent, seeded recipe, host-legality candidates, chosen cardinal moves, pen-ups/downs, material before/after event hashes, trajectory, final material PNG, separate metrics and provenance. No original native or model decisions are fabricated; native/model fields are null, and UNDO_MATERIAL_EDIT is explicitly unavailable.

| Arm (six cases each) | Unique painted cells, sum | Turns, sum | Chosen scent advantage over legal candidates, mean u16 | Fraction choosing highest-scent legal candidate | Mean material Jaccard vs neutral |
|---|---:|---:|---:|---:|---:|
| Neutral | 899 | 258 | 0 | Not informative (constant field) | 100% |
| Attract ×1 | 988 | 224 | +87 | 31.5% | 11.2% |
| Attract ×4 | 927 | 266 | +181 | 32.7% | 9.9% |
| Attract ×16 | 666 | 345 | +297 | 35.9% | 5.7% |
| Repel | 955 | 253 | +2 | 24.0% | 9.6% |
| Mirror-x | 930 | 249 | +69 | 30.1% | 14.2% |

All 36 episodes reach their 192-move budget and all 30 non-neutral episodes depart from the neutral path. **Large footprint divergence is not proof of strong local cue-following**: a tiny preference perturbation can redirect an otherwise chaotic deterministic path; immediate steering sensitivity is weak (×1's highest-scent choice is only 31.5%). At ×16, distinctive scribbling/retracing increases but unique painted coverage falls to 666 across six cases; this might be aesthetically interesting or might be undesirable. No human art-quality or learned-policy benefit has been measured.

**Exact-source local qualification:** 6/6 new Python-stdlib tests PASS. Tested Git blobs: source `008ff42192b2942badcae7762a0f5bea5b01db6c`; tests `3ae12f62ec32ad6bde74cb5c7d37ef26c4b3aab4`; unchanged baseline `b419282d2556e20cc7ca26e62f5dea3c2827dc0a`. The 36-case generated `index.json` SHA-256 is `5be16e72730f09901f8d01b36a33ac4ee07a0ed88a44ea8145d1390adc38fd2f`. Tests include exact baseline event parity, legality, material event reconstruction, paired geometry, admission caps, deterministic byte-for-byte regeneration and synthetic output tampering detection. **Full exact-head D-2 Python suite and frozen eight-case source-qualified confirmation remain outstanding.** This is new synthetic DEVELOPMENT evidence, not native/model/paid inference/Actions dispatch.

**Interpretation and next experiment:** measure *line character, responsiveness to a steer input, retained novelty and retracing* as separate axes. A soft influence may intentionally provoke wild detours. Trial a bounded cursor novelty/escape budget as a separate policy, with art review before promoting any aesthetic metric. Owner-authorised real pinned-Q8 worm experiments are independently owned by #611 and must not be conflated with this D-2 synthetic control.
