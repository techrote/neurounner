# RECOVERY-BALANCE-001 — recovery-positive replay, matched sampling ablation

2026-11-16. **DEV-only; mixed / no established improvement.** No native run, paid inference, model-hub download, structural heldout opening or production code change.

## Research question

Does actively sampling legal, recovery-positive states (especially actions that return from a forced six-step dead-end) help a learned worm recover and paint new material following earlier bad moves, versus the same model and training examples with uniform sampling?

## Design

- Two independent seeds **592 and 593**; two **39-parameter linear-softmax** controllers each, with *identical initialization, host-visible features, policy architecture, labels and update budgets*. No pretrained d1/LFM/Sprocket weights.
- Exactly **180 Adam updates per controller** (60 at each 36→72→128 training horizon); fixed minibatch 64. Only minibatch sampling differs: `uniform` versus `recovery_balanced` (16× for host-legal BACKTRACK/RELOCATE_START labels, 4× for immediate post-return states). Both learn from the *same* TRAIN curriculum and forced-dead-end correction states.
- Six externally imposed cardinal east moves are each checked against the exact legal host menu. Crucially, these six mistakes are **not** included as desired imitation labels. The training examples capture corrective BACKTRACK and RELOCATE_START actions and subsequent paint decisions. Source world/material/protected-cell logic is reused unchanged.
- No hidden future state is given to the student or active at evaluation time. Teacher supplies labels only on TRAIN worlds. DEVELOPMENT worlds have distinct seed partitions, and the two structural HELDOUT families remain sealed.
- DEVELOPMENT evaluation: four ordinary + four deliberately trapped family/seed worlds, 128 and 160 decisions, intact and forced variants, paired seed/model comparisons, all original probability/action/geometry traces recorded. Forced trap episodes impose six host-legal initial east moves plus declared deviations at 12,35,71,109 when alternatives are available.

## Results

| Seed | Metric, forced trap only | Uniform | Recovery-balanced |
|---|---|---:|---:|
| 592 | 160-decision mean painted cells | 145.00 | 154.00 |
| 592 | 160-decision early STOP | 1/4 | 0/4 |
| 592 | 160-decision post-trap outside-corridor ink | 139.00 | 148.00 |
| 592 | Return with ≥8 outside-corridor new cells by decision 46 | 4/4 | 4/4 |
| 593 | 160-decision mean painted cells | 155.00 | 154.25 |
| 593 | 160-decision early STOP | 0/4 | 0/4 |
| 593 | 160-decision post-trap outside-corridor ink | 149.00 | 148.25 |
| 593 | Return with ≥8 outside-corridor new cells by decision 46 | 4/4 | 4/4 |

**Main caveat:** All forced trap worlds succeed on the explicit early-return threshold even without oversampling. The curriculum does not yet discriminate a *qualitative* recovery capability. The apparent seed-592 advantage is not reproduced in seed 593, and across all eight DEVELOPMENT worlds with non-corridor forced deviations, seed 592's balanced arm stops earlier more often at 160 decisions (4/8 versus uniform 2/8). The experiment is **mixed / negative for a robust general improvement** rather than a promotion candidate.

## Original evidence / boundaries

- CPU runs with source-faithful 96×96 synthetic host worlds and no true native receipts.
- 64 checkpoint-replayed episodes per seed; 128 total; 9,080 + 9,213 = **18,293** source-replayed host decisions.
- Four restorable `.npz` trained checkpoints, two per seed. Exact byte hashes, source hashes, original probability distributions, action selections, host menus, histories, painted materials, SHA-256 artifact inventories, elapsed CPU and peak RSS are retained in external ZIP artifacts, never committed to Git.
- Run times **20.869 s / 21.761 s**; peak RSS **128,904 KiB / 129,416 KiB** for seeds 592/593. Decision latency measured inside the complete synthetic policy feature calculation and reported per episode; not a native GPU/host measurement.
- Eight new local unit tests cover hosted recovery actions, same initialized weights, heldout guardrails, actual changed minibatch allocation, original replay, and deliberate probability/material tampering.
- The synthetic `RELOCATE_START` action is not proof of any new native capability. The native 64-logical-move ceiling is unchanged; the >64-decision episodes cannot be called native acceptance.

## Interpretation and next escalation

Our strongest diagnostic is **saturated return success**. Next create more discriminating training/evaluation worlds where several possible returns compete and only a limited recovery budget remains, while keeping materials/protected boundaries source-faithful and independent DEV labels. Do **not** inflate recovery weights further until forced-route resumption and motif continuity can be measured separately from merely avoiding STOP.
