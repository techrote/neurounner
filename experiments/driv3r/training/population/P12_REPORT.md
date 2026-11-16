# P12 — Withheld-cue grammar and long-memory-gap falsification

Research-only, external P0 synthetic World; **not native CyberSand** and not full-mission self-play.

## Protocol

- Counterfactual legal rollouts have identical current observations, masks, paint/parent geometry, legal actions, history-token multiset, and last eight action tokens but opposite correct next painting actions.
- Three permutations of E/W/N survey branches; train grammar A/B, withhold grammar C (N surveyed first); three depths of southern retrace distractors: train 5/7, withhold 9.
- All actions are executed through P0 `World.legal()` and `World.step()`. Only minimum probe corridors are cleared; elsewhere generated obstacles and protected cells are retained.
- Four matched initialization/data regimes: 32-token full-memory GRU, last-8 history GRU, shuffled-history GRU, state-only control. AdamW CPU, 40 hidden units, 16 epochs, 48 train/16 validation/24 heldout seed-worlds per independent cohort.
- No auxiliary order-classification target: train on **next host-legal action** cross entropy only. Final goals are niche-dependent, balanced within identical-world pairs.

## Independent cohorts and observed results

| Cohort | Treatment | Seen grammar / 7 depth | Unseen grammar / 7 | Seen grammar / 9 | Unseen grammar / 9 |
|---|---|---:|---:|---:|---:|
| 649 | full_history | 100% | 100% | 100% | 100% |
| 649 | last8_history | 50% | 50% | 50% | 50% |
| 649 | shuffled_history | 50% | 50% | 50% | 50% |
| 649 | state_only | 50% | 50% | 50% | 50% |
| 650 | full_history | 100% | 50% | 100% | 50% |
| 650 | last8_history | 50% | 50% | 50% | 50% |
| 650 | shuffled_history | 50% | 50% | 50% | 50% |
| 650 | state_only | 50% | 50% | 50% | 50% |

**Scientific interpretation:** both cohorts learn in-distribution temporal-order memory with full history, but **the withheld N-first grammar generalizes in cohort 649 and fails in cohort 650**. The gap-9 distractor itself is not the limiting factor for a familiar grammar. Information-blind controls remain at exactly 50% by construction. No universal compositional generalization is established.

Both cohorts have 4,608 training examples (2,304 order-pairs), 1,536 validation examples, and four heldout test slices with 288 or 576 pairs each, depending on grammar count. Each trains four scratch GRU/scorer models with 2,304 AdamW updates per cohort (4,608 combined).

## Source, controls and evidence

- All 8 P12 unit tests pass, including world-state identity, untouched protection outside the cleared pocket, novel-grammar split, last-8 information impossibility, optimizer weight changes, parameter parity, and action legality.
- Training and heldout worlds are seed-disjoint; model initiation and minibatch seeds are matched between arms within cohort.
- Every holdout prediction is legal by masking, but **legal action** does not imply correct goal, safe independent gameplay or native runtime success.
- Complete checkpoints, learning curves, per-niche heldout accuracies, original SVG/JSON paired traces, manifest and stdout were retained.

## Caveats

- The `World` host is the prior P0 **synthetic surrogate**, not a verified byte-for-byte native Workbench gameplay interface. Its materials are obstacles/protected masks, not a full material solver.
- The task is an artificially constructed *single next-decision probe*. It does not demonstrate autonomous multi-step mission completion, persistent memory beyond 32 tokens, learned materials semantics, artistic quality, self-play, or pretrained SLM abilities.
- A 50% accuracy can mean fixed side guessing, not a functioning navigation policy. Pair-both-correct is also reported to prevent paired trial pseudo-replication.
- Novel grammar shares its action vocabulary and task rule with the training grammars; performance cannot establish open-ended instruction generalization.
- Torch binary checkpoint hashes are environment dependent. Original exact bytes preserved and hashed; no cross-environment bit-reproducibility claim.

## Recommended P13

Attack the 650 failure: randomize branch ordering/permutations, systematically change cue locations, and train a relational target based on *first occurrence of E vs W*; test genuinely unseen distractors and complete legal follow-on missions. Keep all negative initializations, not merely the 649 success.