# MEMPLANT DREAMGRAFT-001 — G1 first donor imprint (smoke-scale)

**Authority:** user-directed continuation of G0; bounded G1 only. G2–G5 and merges remain on hold. **No evidence of actual semantic transplant before sealed controls complete.**

## Experiment

- **Donor:** frozen Qwen2.5-0.5B-Instruct at immutable revision `7ae557604adf67be50417f59c2c2f167def9a775`.
- **Recipient:** frozen SmolLM2-1.7B-Instruct at `31b70e2e869a7173562077fd711b654946d38674`, qualified with both a native fitted-prefix oracle and two counterfactual world full-text tests in [G0C](https://github.com/techrote/neurounner/actions/runs/43678241820).
- **Three feature channels:** contextual pooled Qwen values (`v`), fixed-projected Qwen contextual hidden states (`hidden`), and uncontextualised Qwen token embeddings (`embedding`). Each supplies exactly **2048 floats** to the **same** 64-wide, eight-slot soft-prefix adapter. Fixed random baseline projections are label-free and seeded.
- **Small corpus:** 24 world families × 2 counterfactual rules. 16 families (32 records) train, four (eight) validation, four (eight) sealed; both words appear once in every source, with the disputed/tested claim reversing. Training, validation, and sealed use different templates, groups and materials. This is exploratory and **not** statistically powered confirmation.
- **Training:** one masked teacher-forced answer objective per arm, 48 updates maximum, identical frozen 1.7B, optimizer and seeded training order. Alternate `hardens or softens` vs `softens or hardens` prompt ordering.
- **Controls:** genuine donor, swapped twin (scored against **new donor truth** as well as original), zero prefix and equal-norm random prefix, for both answer orders. Per-label conditional *total* log likelihood, not mean NLL when word lengths differ.
- **Creative:** all twelve raw postcards from the value-derived arm (two held-out counterfactual sources × two unseen creative tasks × genuine, swapped and zero). Unedited, uncurated and **not scientific accuracy evidence**.
- **Artifacts:** full synthetic corpus/provenance, state-feature hashes, raw per-case response scores, gradient trajectories, source-swaps, RSS/CPU, small adapter checkpoints, and creative transcripts. No large donor/recipient weights stored in Git.

## Distinctions and interpretation

A learned donor-derived prefix that follows its source is **a numerical semantic communication channel**. It is not direct K/V grafting, associative KV memory, editable causal allele, shared model-neutral substrate or generational inheritance. The `v` channel winning over the uncontextualised embedding channel would indicate **useful contextualisation under this budget**. Winning over `hidden` is a task-specific bottleneck/inductive-bias result and cannot prove that V contains information absent from the complete hidden representation.

Train/test splits are *group-disjoint*, but all synthetic examples remain small and structurally related; do not generalise from eight held-out examples. Full text and extractive text postcards are known practical controls from G0 and are planned for subsequent scientific comparison; G1 here specifically tests **whether any real source-specific numerical imprint survives**.

## Running

[The G1 Actions workflow](../../../.github/workflows/memplant-g1.yml) prepares verified donor features once, then trains three completely separate CPU jobs in parallel. The native workloads are time bounded: 13 minutes for source extraction and 22 minutes per method, with a 12-minute **training-only** cap inside each method. Maximum scheduled runtime across all jobs is below the initially proposed two aggregate runner-hour fitting limit.

Local example in a fresh, compatible PyTorch/Transformers CPU environment:

```bash
python -m unittest discover -s tests -p test_memplant_g1.py -v
python -m experiments.memplant.g1.features --out runtime/memplant/g1/features
python -m experiments.memplant.g1.imprint --method v --features runtime/memplant/g1/features --out runtime/memplant/g1/fit/v
```

**No automatic promotion to G2–G5, no autonomous schedules, and no merge**. Failures and non-source-specific results are retained as useful negative outcomes.
