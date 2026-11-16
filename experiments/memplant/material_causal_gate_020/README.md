# DREAMGRAFT DG020 — material-law-only causal gate

**Status:** research-only, Draft/HOLD. Extends DG019 on PR #253; no new PR or CyberSand edits. The exact pass/fail thresholds were preregistered [in the PR comment](https://github.com/techrote/neurounner/pull/253#issuecomment-6929893255) before evaluation.

## Hypothesis

Holding world, fork state, observation, legal mask, receiver parameters and all non-material information fixed, changing **only the inherited trained wet-material cost donor** causes the receiver to choose the independently oracle-correct opposite action. This is narrow neural **functional** inheritance, not native state/KV transplantation.

## Fixed experimental contract

Two routes with 12 cells each, a shared 1–4-step MOVE_FORWARD prefix, and a LEFT_ROUTE / RIGHT_ROUTE decision. Dry costs 1.00; wet costs either 0.35 (law A) or 2.30 (law B); the same non-material tile fees and texture are present under both. Independent donor networks learn local tile costs from supervision. A tiny trained receiver aggregates per-tile donor forecasts. It never sees a law label, family ID, full-route oracle cost, graph distance, or oracle action at inference.

Training families: IID material occupancy and short wet runs. **Unseen test families:** edge bands and alternating patterns. For each seed and test family retain 160 worlds for which exact path-cost oracles disagree and both optimality margins are >=0.60. Physical observation, prefix, goal and legal mask are identical between each pair. Only the neural donor changes.

## Predeclared PASS (every seed and held-out family)

Three independent training seeds, 160 pairs per unseen family; 100% input identity and oracle divergence; >=90% **both directions correct on the same pair**; >=95% donor-swap action change; >=94% accuracy per law; <=5% same-law sham flips; <=60% shuffled and zero-donor both-correct; 0 source-blind both-correct, 0 illegal; donor withheld-tile MSE <=0.02. Failure of ANY condition is FAIL. Record confidence and logit changes, per-decision receipts and all negative cases.

## Controls and provenance

- Same-law sham: new independently initialized neural donor for each law, receiver unchanged.
- Shuffled donor: independent random law-source assignment unrelated to each counterfactual condition.
- Zero donor: zero local-cost vector in both conditions.
- Law-blind: separately trained receiver without donor signal, with paired opposing supervision.
- Oracle: exact route-cost sum, recomputed from raw receipts.
- Receipts: entire observations, individual donor tile costs, oracle costs/gaps, both actions and logits/probabilities, all control actions, legal mask and common prefix, SHA-256 hashes.

From repository root:

    python3 experiments/memplant/material_causal_gate_020/experiment.py --selftest
    python3 experiments/memplant/material_causal_gate_020/experiment.py --seed 26011636 --out /tmp/dg020-36

GitHub Actions runs all three pinned seeds and publishes checkpoint and receipt bundles with embedded SHA-256 manifests.

## Interpretation limitation

This diagnostic intentionally uses equal-length paths and only wet cost changes. Its answer often reduces to a **wet-cell-count comparison conditioned on donor predictions**. A positive result establishes causally effective donor-conditioned choices in this restricted synthetic setting, not open-world physics, multistep navigation, recovery, weight/KV transfer, or general Counterfactual Allele success.
