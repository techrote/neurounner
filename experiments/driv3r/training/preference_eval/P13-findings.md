# P13 — Goal-conditional STOP preference in real W16 synthetic-W0 rollouts

Research-only, 2026-11-16. Source: immutable P11 Actions archive `9a9f61038bb1a4f68395816a8296fb8957a8a8f166610921c2a9db6502308d9c`; W16 producer SHA-256 `a7b756a3b0c2d639e5e48dad36d518dfd0e6b752deadca8b04ac11326717eafb`; W16 auxiliary-head checkpoint SHA-256 `f5dc8482a145e39a0bfb10ba94617efad34dc957e3ff485afab3b13726fe90a9`. Frozen source HEAD `e9c7d9ca2a018c2da591b3fcb3429991c907fdfa`.

This **does not** evaluate real CyberSand native physics, real painting/material interactions, the Sprocket root SLM, or human artistic preferences. It trains and executes a new 21-feature, 24→12 MLP goal-completion *proxy* beside the frozen W16 142-input auxiliary PyTorch head in synthetic W0. Unmodified model movement logits use the same deterministic movement draws for all stop variants. All goals, labels and reward comparisons are algorithmically declared; no human/owner judgements were invented.

Four distinct, ex-ante goals were tested: visit at least 48 novel cells; reach 96 novel cells plus 10 branches; perform 6 retraces with 10 new cells after recovery begins; or enter the fixed landmark rectangle x=30..35, y=45..51. Goals were not picked from the outputs of held-out worms. The host continues to own movement legality and obstacle avoidance; this is not a test of protected paint cells.

Training: 18 discovery worlds (6 seeds × 3 families), 9 disjoint validation worlds, 27 disjoint held-out worlds. 2,304 training and 1,152 validation rows; 600 Adam updates; best validation epoch 440. On 3,456 held-out prefix/task rows the learned outcome-proxy head attained 97.54% accuracy, 0.01730 Brier, 0.99278 AP, but still made **54 false-positive and 31 false-negative classifications**. Row-level scores are correlated across time within a world; the grouped world split is essential.

Genuine new model-driven, host-legal closed-loop inference: **540 rollouts** (27 held-out worlds × 4 goals × 5 policies), **57,624 frozen-W16 forward passes**, **432 paired algorithmic comparisons**, and **zero illegal actions**. The baseline reproduces 108 original W16 trajectories and PNGs byte-exact. All new PNGs and JSON receipts are source-hashed in the evidence ZIP.

| STOP controller | Goal achieved / 108 | Stops with unfulfilled goal | Mean steps |
|---|---:|---:|---:|
| Original late-STOP W16 | 58 | 50 | 125.11 |
| Original STOP-from-turn-zero | 54 | 54 | 115.85 |
| Explicit goal-completion oracle | 59 | **0** | **97.69** |
| Trained completion head alone | 20 | **49** | 96.17 |
| Trained head plus explicit goal shield | 59 | **0** | 98.74 |

The unshielded reward/proxy model looked excellent in average precision yet failed badly when entrusted with stopping. The precise **goal-completed safety condition** prevented all such incomplete stops and preserved all 59 attainable successes. It did not outperform the exact symbolic oracle, which also finished roughly one step earlier on average. This is a counterexample to promoting a high-AP reward head into an autonomous behavioural controller without a host-verified accomplishment gate.

P13 does **not** demonstrate human interestingness, creativity, art quality, or reliable recovery beyond the narrow defined synthetic quotas. For future CyberSand usage, export only a proposal for a host-owned goal-completed bit and scoped STOP authorization; do not insert this model into game production settings. New objective families should require material, protected-cell and authoring-state contracts plus real human comparative ratings before wider preference optimisation.

Reproduce locally using:

```sh
export P12_P11_ARTIFACT=/path/to/immutable-P11-hosted-43918309938.zip
python -m unittest -v test_p12_recovery_preference test_p13_goal_conditional_stop
python p13_goal_conditional_stop.py --p11-artifact "$P12_P11_ARTIFACT" --out /path/to/rollouts
```

Keep the `.pt` checkpoint, original producer archives, images and emitted traces in immutable Actions artifacts or local evidence archives; not Git history. PR remains Draft/HOLD. No merges or physical native acceptance claimed.
