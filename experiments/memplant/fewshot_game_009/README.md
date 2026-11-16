# DG009 — Few-shot inheritance of previously unseen game-law combinations

**Research target:** Can a small neural receiver infer a compact behavioural
rule from a donor's legal choices in several **different** material/obstacle
worlds, then apply it to a new world whose correct choice is not present in
those demonstrations?

This is a continuation of DG008 [#241](https://github.com/techrote/neurounner/pull/241)
and directly addresses its negative result for an unidentifiable A∧B-only
physics change. We now supply observations that *identify* the hidden law,
rather than incorrectly expecting unseen physics to be inferred from no
evidence.

## Construction and information boundary

- The six host-legal actions are LEFT, RIGHT, UP, DOWN, PAINT, RETRACE.
- Each synthetic observation contains 12 local material, obstacle, movement,
  novelty, branching, and paintability features per proposed action plus
  a host-owned legality mask. The authored target uses explicit score rules.
  **This is not a native CyberSand world or full-grid simulator.**
- The synthetic interaction is active *only when both parent laws A and B
  are present*. Its three signed components govern wet seeking, branch versus
  retrace, and painting material. Neither parent alone identifies the signs.
- Seven of eight possible allele triples appear during training and
  validation; **(+1,+1,+1) is absent from both**. The sealed evaluation
  presents new independent query worlds with this withheld triple.
- Each donor consists of 2/4/8 separate source worlds, action-feature
  observations, legal masks, and **only the selected action**. No true
  three-bit allele, target score, or queried world's answer is exposed to
  the receiver.
- A DeepSets-style neural context encoder compresses eight donor decisions
  into a learned 16-dimensional latent. The same receiver selects a legal
  action in an independently generated query world. A separately trained
  query-only baseline has a comparable small MLP policy.
- Train donor dropout (K=0,2,4,8). Compare held-out correct donors,
  no donors, and matched **opposite-allele donor choices on identical
  donor worlds**. The latter is the proper causal negative control.
- The additional shuffled-donor statistic mixes *other source worlds*. On
  the sealed set, ALL sources share the withheld triple, so this
  permutation does **not** test allele identity and must not be interpreted
  as a causal negative control.

The heldout source interactions are **authored synthetic teacher laws**, not
native KV transfer, genuine Qwen donor messages, cross-model transplantation,
real CyberSand game play, real model-generated donor policies, physics
discovery, or unseen atomic physics functions. Success would show
**supervised few-shot composition of known atomic game preferences under
an unseen combination**, not spontaneous discovery.

## Reproduce

```bash
python3 -m experiments.memplant.fewshot_game_009.experiment --selftest
python3 -m experiments.memplant.fewshot_game_009.experiment \
  --out runtime/memplant/dg009/seed13 --seed 26011613 \
  --steps 650 --train-cases 3200 --heldout-cases 750
```

The dedicated CI trains independent seeds 26011613 and 26011614 using
bounded CPU runners. Original model checkpoints `receiver.pt` and
`blind.pt`, source-separated query identities, detailed held-out metrics,
actual model-selected action decisions, training curves, and SHA-256 records
are attached to each exact-source Actions run. No pretrained DG008
checkpoint is silently claimed as being inherited by this model.

## Interpreting a positive or negative result

Require higher exact-action accuracy than the independently trained blind
policy **and** the no-donor setting on withheld (+1,+1,+1) worlds. The
matched opposite-donor intervention should significantly reduce accuracy,
showing that the decision depends on inherited behavioural information.
Legality is checked for every predicted action. Performance on the held-out
triple may remain well below that on seen triple combinations; preserve it as
a genuine compositional-generalisation limitation.

**Next scientific step:** train *actual specialised neural donor policies*,
inherit source-pinned latent states rather than authored demonstration
choices, and test rule families whose atoms—not only combinations—were
unseen. Keep this DG009 synthetic demonstration result distinct.

## Isolation

Only Neurounner research files and this dedicated CI workflow may change.
All CyberSand game code/content is **read-only**: never touch `native/**`,
`godot/**`, scenes, Workbench, materials, maps, renderer, physics, assets,
or production/default providers. No paid API/GPU. PR stays **Draft/HOLD**;
never merge or mark Ready without owner approval.
