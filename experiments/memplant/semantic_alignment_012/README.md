# DG012 — donor-private feature semantics with paired observational calibration

**Status:** Draft/HOLD, research only. **No changes to CyberSand game code/content.** Stacked on DG011 #244.

## Question

DG011's successful interventional fingerprint explicitly selects the known meaning of each source observation axis. Can a receiver recover useful game-law behaviour from neural specialists whose **own ten input axes are independently permuted and sign-reflected**, without seeing their private feature coordinate maps?

DG012 trains 18 independent neural specialists per seed (three architectures × three law families × two signs) on privately transformed inputs. A new DeepSets receiver is trained to combine three unordered neural donors and select one of six host-legal game actions in synthetic material, obstacle, branch, painting and retrace worlds. The transfer channel is either:

- **Paired black-box observations:** 8, 16 or 24 ordinary canonical game-world probes with matching real neural donor outputs; fit a ridge-regularized, action-specific functional fingerprint. No explicit feature-setting intervention or donor coordinate map is supplied to the receiver.
- **Raw output control:** the same number of donor logits, without their matching probe-world observations.
- **No-source control:** identical architecture/parameter count with zero source.

The calibration input worlds are known to the experimenter/receiver. Only the **donor's private feature semantics** are scrambled. This is *not* ontology induction without any known observation language. A deliberately mispaired-world control retains donor output samples while scrambling their correspondence to observed world features.

## Training and sealed tests

- 18 separate specialists per seed, 220 optimizer steps each; 900 receiver updates; three receiving models per seed.
- Train on 5,200 worlds and seven of eight sign triples; validate on 500 independent worlds.
- Test on 900 entirely new **withheld (+,+,+) law** worlds and separately 900 new **mixed-law** worlds, with disjoint world hashes.
- Receiver training sees only narrow/wide architecture specialists, three training probe banks and 8/16/24 observation budgets.
- Sealed architecture is a separately initialized *gated* MLP. Sealed probe bank is also unseen. Every specialist has an independently sampled private permutation/reflection of input features.
- Evaluations include donor swap (only wet law sign changes, same query world), correct counterfactual oracle, zero source, raw output, wrong probe-world pairing, exact donor slot permutation invariance, and an intentionally **unidentifiable** +++-only third-order interaction.
- Source code, donor and receiver checkpoints, 64 sampled legal action predictions, JSON reports and SHA-256 manifests are in immutable CI artifacts, not Git history.

## Independent local CPU findings (before CI)

The same source/CLI parameters were run with Python + **torch 2.10.0+cpu**. Values below are exact-action matches against a *synthetic authored expert*, not gameplay success or discovered physics. CI uses a separately pinned torch 2.5.1 CPU environment; its results are authoritative for CI qualification.

| Gated architecture / unseen probe bank, 900 independent test worlds | Seed 26011620 | Seed 26011621 |
|---|---:|---:|
| Correct paired, 8-probe, all 8 laws | **79.89%** | **79.00%** |
| Correct paired, 16-probe, all 8 laws | **86.22%** | **91.22%** |
| Correct paired, 24-probe, all 8 laws | **88.67%** | **92.11%** |
| Raw 24-probe, all 8 laws | 47.44% | 46.33% |
| Source-blind, all 8 laws | 42.22% | 42.56% |
| Deliberately mispaired 24-probe worlds, all 8 laws | 24.22% | 18.56% |
| Paired 24-probe on withheld +++ only | 59.00% | 77.78% |
| Correct after same-world wet-law donor swap, mixed laws | 86.78% | 92.56% |

**Important caveats:** Performance on the single withheld +++ combination varies substantially by seed. Law signatures are trained using an authored synthetic utility teacher; the receiver inherits *neural functional effects*, not independently discovered game physics. Raw source has no matching calibration-world features; comparison measures practical calibration protocols, not an identical information budget. Paired transfer still requires the experimenter to supply canonical probe observations. Third-order physics added only at test time remains unidentifiable, as expected.

## Running

```bash
python3 -m pip install 'torch==2.5.1' 'numpy>=1.26,<3'
python3 -m experiments.memplant.semantic_alignment_012.experiment --selftest
python3 -m experiments.memplant.semantic_alignment_012.experiment \
  --seed 26011620 --donor-steps 220 --receiver-steps 900 \
  --train-cases 5200 --sealed-cases 900 --out /tmp/dg012-20
```

## Interpretation and next step

This adds a stricter, causally probed **cross-donor observation alignment** condition beyond DG011: a receiver can use paired world/output examples to recover behaviour from independently trained neural functions even when donor-local feature identities are unavailable. It does **not** establish real cross-transformer KV grafting, unsupervised discovery of physical laws, native CyberSand performance, or long-horizon control.

Next: measure trajectory-level decision quality, learn calibration requests adaptively rather than receiving fixed random probes, and withhold whole material-interaction operators (not just law-sign combinations). CyberSand is read-only; any game-side interface proposal requires separate owner approval.
