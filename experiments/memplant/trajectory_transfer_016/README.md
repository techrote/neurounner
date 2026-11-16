# DG016 — unseen topology, timed obstruction and neural donor transfer

**Research-only. Draft/HOLD. Never merge/Ready without owner acceptance.** Stacks on DG015 PR #248 (`66fc77fdabba020f09cfa1231f9910a974fc852b`). No CyberSand edits.

DG015 used mirrored, rotated corridor layouts; this experiment asks whether transferred neural material/goal/paint rules survive **randomized 9×9 maze topology** and delayed closures. It uses random spanning-tree mazes with independently sampled cycles, wet clusters, protected painting cells, goal branches and a gate on a traversable route. The host-legal six-action menu is `LEFT`, `RIGHT`, `UP`, `DOWN`, `PAINT`, `RETRACE`; retrace traverses and pays for an actual edge and cannot pass a closed wall. All mazes are verified connected when the gate closes. Observable channels contain only map cells, actor/goal/history markers, painted cells, gate location, gate-closed state and time until closure — **no precomputed distances**.

**Train**: neural wet, goal and paint specialist donors, six narrow plus six held-out gated donors with private input permutations/sign flips. Train a conditioned 192–128 MLP receiver and an equal-architecture source-blind MLP with Dijkstra-replanning imitation on 300 distinct randomly generated maze seeds and seven three-law combinations. The `+++` law triple is excluded. **Sealed evaluation**: 100 IID mazes with 2–4 move closures and 100 OOD mazes with sparser cycle topology and 5–8 move closures, repeated over two independently seeded training runs. Compare correct donor, wrong wet donor, wrong goal donor, blind receiver and lawful oracle in up to 96 decisions. The donor architecture used at test time is gated and absent from receiver training.

Preserve all donor and receiver checkpoints, full source/manifest hashes, reports and original visual trajectories as immutable GitHub Actions artifacts. Primary endpoints are **completion**, **failure-penalized traversal cost across all cases**, **joint-success true-law cost regret with denominators**, wet-law counterfactual action changes, recovery actions and illegal-action count. Goal/donor swaps test inherited semantics; oracle under opposite wet physics is evaluated independently on the same world. A closure can occur after the episode completes: do not miscount that as a failure.

**Limitations:** synthetic worlds and authored teacher laws, donor *output* probes rather than true weight/KV transplantation; changing maze cycle density and gate timing together prevents causal attribution between them. Positive performance is supervised behavioural inheritance, not discovery of unobserved physics. Negative generalization is valuable and must not be suppressed.

```bash
python -m experiments.memplant.trajectory_transfer_016.experiment --selftest
python -m experiments.memplant.trajectory_transfer_016.experiment --seed 26011628 --out /tmp/dg016
```
