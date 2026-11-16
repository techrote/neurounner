# C18 — host-attested trail evidence; trained adapters rejected (16 November 2026)

**Repository lane only.** Draft/HOLD PR #639, no merge or Ready. This is synthetic CyberSand-like model-side research, not game or Workbench code.

## Question and boundary

C14–C17 learned a near-constant advice bias and struggled with conflicting hidden obstacle geometry. C18 tests a three-scalar, **host-verifiable** observation: whether the current goal occurs in the executed cursor-history stack, normalized depth of its last occurrence, and whether the next legal BACKTRACK would reach it. The host already owns these recorded facts; no hidden material topology, cue validity flag, global route answer, future reward or alternative-outcome oracle enters the model inputs. A goal on history is evidence, **not proof** that reversing is optimal.

A frozen C12 receives the original 29 features. The experimental residual receives the 29+3 extended menu, changes only the BACKTRACK logit, and only when advice is live, the host marks BACKTRACK legal, and the goal is attested on history. All off-history and all no-cue logits are bit-identical. Host permissions and protected cells remain authoritative.

## Training: real optimizer with explicit negative ablation

- Hairpin/staircase train worlds: **384 matched legal action menus**, three hint strengths, opposite personalities, two signed cues, both goal directions.
- **1,536** host-executed counterfactual suffix trajectories comparing forced BACKTRACK with the frozen policy's best legal forward move. Only realised trajectory returns supervise the head.
- Two frozen-C12 **32→16→1 AdamW adapters, 180 steps each** (360 total). A permits either signed instruction; B permits only explicit BACKTRACK.
- Losses: A **0.722043→0.739250**, B **0.704621→0.736715**. Both worsened; no optimizer-success claim. Exact source receipts, full-precision outcome advantages and rebuilt training inputs reproduce both 3.2 KB safetensors **byte-identically** in a second optimizer pass.
- A SHA256 `544b486f283069be233899959a4d6228ab72dfb4e8069cdea5a51228c373d8be`; B SHA256 `c07b1e2295a725040c459ea70dbe31049cb2afbc96d4f984237f92dc2a7f0ec5`. Binaries **outside Git history**.

## Independent source-seed holdout: 48-decision host-applied episodes

**3,456** C12/A/B independent trajectories, plus **2,304** untrained A0/B0 controls on exactly the same 1,152 source groups. Hook/zigzag topology families excluded from optimizer. Goal completions below are out of 192 per condition:

| Condition | C12 | A trained | A0 untrained |
|---|---:|---:|---:|
| Goal behind, valid BACKTRACK | 162 | 167 | **174** |
| Goal behind, misleading CONTINUE | 157 | 167 | **179** |
| Goal ahead, misleading BACKTRACK | 178 | 178 | 178 |
| Goal ahead, valid CONTINUE | 157 | 157 | 157 |
| Goal behind, no advice | 117 | 117 | 117 |
| Goal ahead, no advice | 175 | 175 | 175 |

On *withheld only* valid BACKTRACK: **67/96→71/96→78/96**, and misleading CONTINUE with goal behind: **69/96→77/96→86/96**. B trained achieves 165/192 valid BACKTRACK and its untrained B0 control achieves 170/192.

**Critical interpretation:** The *untrained* hard-attestation architecture outperforms **both actually trained** variants. The optimizer counterproductively dampens the useful initial +1.0 BACKTRACK bias; do not describe the improvement as learned policy quality. Retain C12 as the preferred trained checkpoint. A0 is an architectural research control, not a promoted trained model.

Exactly matching C12/C18-A/C18-B no-cue pairs: **384/384**; ahead-goal (off-trail) pairs **576/576**. No synthetic host rejections, illegal top-1 selections, protected-material edits or ink loss in the independent panel.

## Style and deliberate adversarial shortcut

Active BACKTRACK personality turn-gap: C12 **0.025525**, trained A **0.005329**, untrained A0 **0.023665**. Active CONTINUE gap: C12 **0.139284**, trained A **0.141131**, untrained A0 **0.103263**. No-cue gap unchanged at **0.197840**. This does **not** establish aesthetic personality retention.

An entirely new, host-legal loop/shortcut fixture places the goal six history pops behind, yet only two legal forward steps away. Each mode has 96 episodes/model. Under BACKTRACK advice, successful goals C12 **28**, trained A **35**, untrained A0 **42**. Under CONTINUE advice: **78**, **77**, **86**, respectively. Trained A's average steps under CONTINUE worsen **3.688→4.365**. This exposes why merely knowing that the goal is on history does not solve optimal routing.

## Safety, custody and disposition

**130/130** locally available tests PASS, including core-experiment logit equivalence, legal-menu guards and protected-material checks. This is **not** the full exact-HEAD repository suite; missing C5/C10/C15 archived modules. Additional 1,728 development trajectories, fresh-seed holdouts, source-reconstructed galleries, replica checkpoints and SHA256 receipts are in the external C18 bundle. Immutable GitHub Actions checkpoint custody still needs a separately authorised workflow owner. The small committed `attested_stack_core.py` is the reviewable model-side interface; complete executable optimizer/evaluation sources remain in the off-Git bundle.

**REJECT C18-A/B weights; keep C12.** Next: on-policy full-episode objective anchored to successful A0, with hard misleading-cue/style nonregression and more loop/shortcut obstacles. No game/native/Godot/material/map/Workbench/default-provider changes, paid APIs/GPUs, or merge.
