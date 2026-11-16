## WORM-001 LFM W6 (new, separate source-fenced experimental training)

Owner authorization: #612; Draft/HOLD #575. Fresh from exact full LiquidAI/LFM2.5-VL-450M pinned parent; no W5 checkpoint continuation or full-Art-v2 corpus.

**Intervention:** 120 CPU AdamW optimizer steps; 20 original supported steering records and 20 explicitly synthetic last-four-RIGHT counterfactual legal steering choices per family (three families). Synthetic history is deliberately *not* represented as a physically executed W0 observation. Training-only derivation, no sealed FINAL, no DEVELOPMENT targets used for optimization.

**Evaluation:** preserve untuned vs learned 18 first-pass DEV decisions, raw EOS/token IDs, six 48-step source-faithful model-in-loop W0 PNGs and same-seed random/wall/oracle comparisons. Source/recipe/model/data/CPU/settings/inventory fully versioned. Raw model errors remain visible and W0 host alone enforces legal application.

Prior W5 finished 120 genuine updates but collapsed to 288/288 RIGHT in online rollout. This is a hypothesis-driven anti-loop experiment, not a claim of quality improvement.
