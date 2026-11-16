# C9 — bounded multi-turn host feedback (16 November 2026)

**Research-only / Draft-HOLD.** External, synthetic CyberSand-like worm game; exclusive changes under experiments/driv3r/training/guidance_recovery/. No native, Godot, Workbench, game materials, accepted contracts, provider defaults, resident worms, paid spend, real material Undo, merge or Ready change.

## Mechanism

C8 uses a BACKTRACK-only residual requiring simultaneous changed-goal and host-cue flags. The synthetic host clears goal-change after a single action, making sustained correction difficult. C9 introduces an **optional, research-only 26th observation scalar**: host-confirmed short-lived recovery context. The original 25 features are unchanged. Context begins only when a host goal-change and recovery cue coincide; lasts at most nine actions; terminates on goal change, cue expiry, Stop or goal arrival. **This is not an accepted shared/native contract change.**

C9 is a frozen-C8 plus trained 26→20→1 residual, gated by both active host cue and bounded context, affecting only BACKTRACK. Other nine candidate logits are exactly unchanged even when active. Without context or cue, **all logits are exactly C8**. Host rechecks legal menus and protected-material permissions. Synthetic BACKTRACK pops cursor history but does not remove painted ink.

Training: source-disjoint hairpin/staircase trails; held-out hook/zigzag topologies; reversal and false-warning productive continuation; no/weak/strong (0/0.35/1) guidance; two personalities, legal host-replayed intermediate states at multiple depths.

## Actual CPU training

180 training source groups × (five reversal + four continuation intermediate states) × 3 arms = **4,860 labeled candidate menus**. **608 actual AdamW optimizer updates**, 16 epochs. Regularized binary loss **0.185968 → 0.034970**. Checkpoint SHA-256: **03ac2de36e480b26fd16b86637899ddeaf7221431d13762fbc0b0181f75b2366**.

Frozen prerequisite checkpoint SHA-256:
- base: 8c8625a1bd8daed03e00188687dbba9732634c7bf7b8e7be5a4fa21cb310ac09
- C4: f1d538200c0d5e183d46463a9058f73db162a5dcccf13daeb521073da9bfc0fa
- C7: 9ddadf372465abca9634eca8bf450cb03e5903434da8065de588511b44b5da9a
- C8: b1d485773fd33d760d7a41f1bb90b3aa23e208f1241c7dfcabb04cef1aa697bd

## Held-out host-applied stochastic results

**4,608 original 24-turn synthetic episodes**. Each C8/C9 pair starts with the same source World, seed, material, legal menu, personality and random stream. Trajectories no longer represent same-state interventions once actions diverge. Only 24 independent seed identities per topology; three guidance arms and both personalities reuse matched identities.

| Cue-present outcome | C8 goals reached | C9 goals reached |
| --- | ---: | ---: |
| Training families, goal reversed onto history | 120/288 | **283/288** |
| Withheld hook/zigzag, goal reversed | 98/288 | **207/288** |
| **All goal reversals** | **218/576** | **490/576** |
| Training families, false cue/productive goal | 277/288 | 279/288 |
| Withheld families, false cue/productive goal | 271/288 | **265/288** |
| **All false-cue continuations** | 548/576 | 544/576 |

On withheld reversals, average longest consecutive BACKTRACK run **3.00 → 6.07** decisions; successful recovery trajectories average **10.57 → 8.58** actions to goal. An exploratory bootstrap clustered by 48 withheld topology×seed identities (retaining paired personalities and scalar arms) found **+0.3785** relative arrival fraction, 95% percentile interval **[+0.299,+0.455]**. That is synthetic sampling uncertainty, not native generalization.

Cue-off runs: **1,152 C8/C9 pairs with identical action sequences and material trajectories**; no-cue reversed goals both **307/576**. Zero host-rejected moves, constant protected materials and no ink deletion.

**Negative:** withheld productive false warnings worsen by 6/288 goals; active-cue productive turn-personality separation across all families drops **0.0299 → 0.0027**. A BACKTRACK-only logit change can still alter entire artistic paths. Host context depends on truthful host cue issuance; non-authoritative misleading advice is not fully rejected. This is not a trained d1 model or evidence of native material rollback.

## Qualification, artifacts and limits

Source and nine new cheap tests are published exclusively under the training lane. **88/88 locally available tests pass** (79 prior C8 modules plus nine new). New C9 Python source/test/gallery files were matched to exact Git blob identifiers. **The missing C5 optional-clearance module and 11 tests were absent from the archived local checkout: full C5-inclusive exact-HEAD suite remains unqualified.**

Original full run saved checkpoint, 4,608 complete raw trajectories, source receipts, numerical metrics and **16 source-reconstructed gallery panels**. A second independent CPU train (with fewer evaluation episodes) yielded byte-identical checkpoint and train-source receipts. Do not claim the full 4,608-case evaluator was independently repeated. Binaries and raw receipts remain off Git history, in a locally hashed owner bundle; immutable GitHub Actions artifact custody still requires a separately owned workflow and remains open.

Reproduce from guidance_recovery/ with the four prerequisite safetensors installed locally:

    python -m unittest discover -s . -p 'test_*.py' -q
    python persistent_feedback.py --base /path/base.safetensors --c4 /path/c4.safetensors --c7 /path/c7.safetensors --c8 /path/c8.safetensors --out /tmp/c9 --n 180 --epochs 16 --rollout-n 24
    python persistent_feedback_gallery.py /tmp/c9/raw_trajectories.json /tmp/c9/gallery.html

Next focus: adversarial host cues, evidence-calibrated early revocation, and active-correction path-style retention. Preserve C8 as a negative/tradeoff comparator; C9 is not approved for production/native transfer.
