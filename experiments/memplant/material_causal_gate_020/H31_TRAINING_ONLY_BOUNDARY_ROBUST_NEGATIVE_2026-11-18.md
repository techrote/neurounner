# DREAMGRAFT DG020 H31 — genuine-donor near-boundary training-only negative (2026-11-18)

**Classification: TRAINING-ONLY INSUFFICIENT; not a new prospective PASS/FAIL.** The latest stringent prospective result remains **H27 FAIL 28/30**, while H26's historical restricted registered PASS is retained together with its post-hoc near-tie falsification. Draft/HOLD PR #253 remains open; no merge/Ready.

## Bounded intervention

Original three H29 209-parameter trained receivers, original four genuine trained H21R donor source/replicate pairs, original two laws, original 2×12 route topology. Used only the exact H27 training worlds and observation hashes: 2,560 near-tie matched worlds per receiver seed (7,680 total), from the two existing training placement families, with 2,560 corresponding broader-margin training examples also used for optimization per seed. No H27/H26 acceptance-world inference in this trial, and no new prospective holdout was opened.

**Frozen training prescription:** from original H29 checkpoint, 1,400 AdamW steps, learning rate 0.00012, minibatch 128, four genuine source/replicate pairs simultaneously. Same absolute tile-cost, worst-replicate route-margin, paired-law delta, equal-wet consistency and teacher-preservation objectives as H29; strengthen within-law tile disagreement term from 80 to **800**, and change sign hinge from 0.05 at 0.00025 to **20 at positive margin 0.0015**. All four donor variants must satisfy the supervised near-boundary action margin. Train RNGs receiver seed + 99001/99002. Oracle values are training labels and retrospective scoring only; receiver inputs contain no law labels, family IDs, oracle distances or correct actions. Geometry and receiver weights are identical within each law-swap counterfactual pair.

## Independent frozen-neural training-only results

| Measure (7,680 near-tie training pairs) | H29 baseline | H31 |
|---|---:|---:|
| Individual genuine-donor source/replicate action changes | 120 | **124** |
| All four donor variants both-law correct | 7,560 | **7,556** |
| Cheap-edge divergent donor changes | 13 | 16 |
| Expensive-edge divergent donor changes | 42 | 48 |
| Wet-rich invariant donor changes | 49 | 41 |
| Wet-poor invariant donor changes | 16 | 19 |
| Equal-wet invariant donor changes | 0 | 0 |

H31 primary individual-donor actions both-law correct: **7,613/7,680** (divergent **3,024/3,072**, invariant **4,589/4,608**). Swapped-law genuine donor control: **0/3,072** both-correct divergent. Zero-feature donor control: **0/3,072** both-correct divergent. Illegal actions: **0**. These latter controls establish training-world law dependence, not unseen-world selectivity.

**Negative result:** strengthening losses improved wet-rich invariant source stability by eight cases but worsened divergent categories and overall source stability by four. The improvement is insufficient; **do not proceed to a new prospective cohort**.

## Independent mechanism audit

Frozen ReLU-network Jacobian evaluation at wet tile + 0.15 fee, all four genuine individual donor predictions and both laws, found identical active ReLU masks within each law for every receiver seed in both H29 and H31. The H31 local tile-output derivative with respect to donor feature remains approximately **0.790–0.999** (H29: **0.797–0.999**). The receiver remains *locally affine* over the donor source/replicate span and does not exhibit a noise-rejection plateau. These fixed-input local Jacobians are not a global proof across every observation.

**Important reporting correction:** the executed trial's embedded per-bucket diagnostic inadvertently covers only the first training family (256/category). Use the separately written independent evaluator, which accounts for both families (1,536/category); its total 120→124 cross-replicate count agrees with the executed trial.

## Provenance / custody

- Frozen original H29 SHA-256, seeds 26011639/40/41: `bd7480c6d34a75ce1cc1c8f8869936aae501a86c6914c73f398981d5c06b7455`, `1eabd720b94431520d683a10c397c3698cd4b35b0fbebe4bf7ed8ac353c783ca`, `4424e95c72f7a248649fb86ab5c223f33e489aa20a23c277189552baf5f3aa75`.
- Original H31 checkpoint SHA-256, respectively: `2acfeea8b1f08ae99f131e9499abca2765a96ec7d312f825e0d450fe3d80708f`, `c39807671b2ec174d375aea83e2766299ce0d628443101b3d168e386ff3afdee`, `1c1e43563eab037926c3bfbf6054788f77291ad7468a00c6fce3aaa2b107f310`.
- Independently regenerated 7,680 raw receipt JSONL SHA-256: `2a240d6a571fa382033707b7016cc22d10e5e114a50bcc2d3d462c50b7c45ba6`.
- Owner handoff complete self-contained ZIP SHA-256: `cab7f2825265e924cd39314df112bec364514091c47284f9cf44124465e186ae`; 77/77 files verified from manifest in a clean extraction. Exact genuine donor/receiver checkpoints, trial sources, raw decisions/logits, complete capsule, old H27 training hashes and independent replay scripts are in that ZIP. This Git commit preserves findings and hashes, but the full new source/checkpoint/receipt ZIP is **not yet in GitHub custody or exact-HEAD GitHub Actions**.

**Smallest next mechanism change:** train-only activation-knee placement / initialization of *the same 209-parameter receiver* to permit a low-gain donor-feature response inside noisy source/replicate bands while preserving correct cross-law cost-dependent action sensitivity, absolute signed route margins and zero/shuffled controls. Do not mask individual-source variability by replacing it with a donor ensemble or hardcoded two-law labels. Demand simultaneous improvement in both divergent and invariant categories before opening preregistered new unseen families.

Governance: writable repo Neurounner only, this existing PR/branch only; CyberSand and DRIV3R read-only; no paid GPU/API spend; never merge/Ready.
