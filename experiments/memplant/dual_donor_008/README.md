# DG008 — specialised neural game-law donors and held-out composition

**Status: Draft/HOLD, research-only.** Independent of native game edits. Stacked on DG007 PR #239, which remains unmerged.

## Scientific question

Can a receiver use **internal representations from two separately trained, frozen game-law specialists** to choose legal actions under combinations of laws not present in its own training? Is learned pairwise composition preferable to simply adding the donors' atomic effects?

This is a more demanding composition test than DG005–DG007's three-bit state table, but it deliberately uses **small synthetic neural donors**, not the previous Qwen2.5 attention-V source or native KV transplantation. Do not cross-claim results across those source types.

## Protocol

- Specialised parent A learns material/paint effects for law families 0–2; parent B learns motion/goal/retrace effects for families 3–5. Each has two polarities per law and a 16-dimensional learned internal embedding, plus an action-specific effect head. Fit separately, **freeze both**.
- Receiver inherits **only those embeddings and predicted per-action atomic effects**, not raw law-card names. Train a permutation-symmetrised pairwise residual module to correct additive single-law scores.
- Train examples: 6,500 distinct 9×9 worlds with **zero, one or two active laws**; validation: 1,200 further worlds. The two parent specialists separately see 4,200 worlds each. Train 280 parent updates and 450 receiver updates per seed.
- Sealed tests: 1,600 worlds each with 1, 2, **3 or 4 active laws**. Three- and four-law combinations are withheld **as a cardinality**, not as unknown law identities. All six law families, both polarities, and underlying pair laws are seen in atomic/pair training.
- Host controls legality for LEFT/RIGHT/UP/DOWN/PAINT/RETRACE irrespective of donor outputs. Source and paired synthetic histories are deterministic.
- Compare paired residual receiver, frozen-donor atomic additive baseline, source-free exact base, shuffled donor, and scripted oracle. An additional **outcome-conditioned 8,000-world stress bank** isolates cases where authored pair interactions change the oracle action relative to ideal symbolic addition (report selection conditioning; never present as natural-frequency accuracy).
- Probe identical-world polarity flips for counterfactual causal influence; retain 32-turn sequential synthetic trajectories and original PNG bytes.
- Train two independent CPU seeds 26011611 / 26011612; version checkpoints and immutable Actions artifacts; publish failures and negative outcomes.

## Results from locally executed CPU training

These are source-matched offline results **before** GitHub exact-head CI qualification; prefer the new Actions-run reports when completed. Raw full reports and checkpoints are saved separately. Accuracy is **exact synthetic expert-action agreement**, not CyberSand gameplay success.

| Condition | Seed 26011611 | Seed 26011612 |
|---|---:|---:|
| Grafted, 3-law withheld combinations | 94.5% | 95.2% |
| Additive, 3-law withheld combinations | 95.3% | 94.6% |
| Grafted, 4-law withheld combinations | 92.2% | 94.0% |
| Additive, 4-law withheld combinations | 95.1% | 95.3% |
| No-donor, 4 laws | 37.2% | 37.8% |
| Shuffled donor, 4 laws | 28.6% | 27.5% |

**Important negative:** overall the pairwise graft **does not beat the simple additive donor** at four laws. Do not promote it based on the high headline accuracy alone.

However, on interaction-specific cases selected because true pair laws **change the oracle action**, the learned graft beats the additive model:

| Interaction-conditioned test | Seed 11 graft/additive | Seed 12 graft/additive |
|---|---:|---:|
| 3 laws, 136/8,000 cases | 54.4% / 10.3% | 57.4% / 4.4% |
| 4 laws, 261 or 263/8,000 cases | 51.7% / 7.7% | 60.1% / 7.2% |

This enriched evaluation is **not an unbiased population metric**. These rare but demanding scenes reveal a benefit in modelling pairwise interactions even though the proposed receiver extrapolates poorly on ordinary four-law combinations.

Across 600 identical-world polarity flips per seed, the graft correctly determined whether action choice should change in 97.3% / 94.8% of pairs; additive achieved 97.2% / 95.5%. This does **not** prove independent new law discovery.

## Reproduce

```bash
python3 -m pip install 'torch==2.5.1' 'numpy>=1.26,<3' 'Pillow>=10,<13'
python3 -m experiments.memplant.dual_donor_008.experiment --selftest
python3 -m experiments.memplant.dual_donor_008.experiment --seed 26011611 --out runtime/memplant/dg008/seed11
python3 -m experiments.memplant.dual_donor_008.experiment --seed 26011612 --out runtime/memplant/dg008/seed12
```

Large model files, JSON reports and original images go to retained Actions artifacts, **not Git history**. Reports record checkpoint SHA-256, seeds, train curves, controls, source limitations and complete synthetic action traces.

## Interpretation and next falsifiers

The frozen neural donor messages carry task-relevant causal information for synthetic legal action selection. But donor cards were supervised, families and polarities are all present in training, world physics is deliberately authored, and the receiver has a **hand-designed pairwise inductive bias**. Learned embeddings can act merely as identifiers; **random-signature and independently unseen donor-family controls are still needed**. Future research should test non-additive novel families, learned rules from actual game observations, real Qwen-state donor bridging, unlabelled source inference, and donor-content-controlled information bottlenecks.

**Hard boundary:** Never modify CyberSand game native/Godot/Workbench code, assets, materials, maps, content, rendering, simulation or default providers. No paid API/GPU. Do not merge or mark Ready without owner approval.
