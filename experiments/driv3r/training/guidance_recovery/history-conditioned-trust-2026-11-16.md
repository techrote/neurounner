# C17: history-conditioned host-feedback training — negative result (16 November 2026)

**Scope:** Synthetic DRIV3R worm policy research only. **Disposition:** keep C12; C17-A/B/C/D rejected for promotion. PR #639 stays Draft/HOLD; do not merge or mark Ready.

## Actual optimizer runs

Four new frozen-C12 BACKTRACK-only 34→16→1 residual adapters. Each receives the original 29 C12 observation features plus five statistics calculated from completed, *independently sourced* synthetic host-advised episodes: number of prior trials, fraction reaching goals, realized goal-distance progress, most recent success, and success fraction for the same advice sign. No current cue validity, hidden obstacle cells, unused counterfactual outcomes, or adviser-quality latent enters model inputs.

The adviser reliability generator has hidden bands 0.15/0.50/0.85, with no/weak/strong scalar guidance arms, both personalities, protected materials, legal action menus, and host-applied synthetic paths. The frozen policy retains every non-BACKTRACK logit exactly even with live advice, and **all** logits exactly with advice absent, cancelled or expired. No material Undo.

- **C17-A/B/C:** 1,440 paired observations each; 322 real CPU AdamW updates each. A loss 0.477408→0.467174, B 0.477680→0.471898, C 0.477608→0.471558.
- **C17-D alias challenge:** 960 verified observation-identical but source-hash-different near-wall pairs; 8,640 training observations; 1,080 AdamW updates; loss 1.209981→0.886829. Held out depth-4 walls and rotations 2/3. Host advice correctness varies according to an unobserved adviser-quality variable; past host-executed trial outcomes are supplied, never current-hidden geometry.

## Independent evaluation

**C17-A, withheld hook/zigzag, 48 decision limit, 720 episodes per model:**

| Outcome | Frozen C12 | C17-A |
|---|---:|---:|
| Valid host cue: complete goal | 276/352 | 301/352 |
| Misleading host cue: complete goal | 309/368 | 280/368 |
| Overall | 585/720 | 581/720 |

The 40-decision development panel also rejected B (valid 291/369→306/369, misleading 297/351→291/351) and C (valid 304/369; misleading 289/351). Fixed-observation changes between high/low historical reliability were only about **0.0063 BACKTRACK logit** for A, compared with approximately +0.76 aggregate learned correction: the model principally learned a *constant advice-active bias*, not useful historical trust. Active-cue aggregate personality turn-bias separation declined 0.18518→0.14708.

**C17-D, deeper held-out aliased walls; 288 episodes per model, first decision under test followed by a frozen-C12 host-executed 28-decision suffix:**

| Outcome | C12 | C17-D |
|---|---:|---:|
| Valid advice: first-action correct | 77/146 | 64/146 |
| Misleading advice: first-action correct | 75/142 | 93/142 |
| Valid advice: goal reached | 136/146 | 128/146 |
| Misleading advice: goal reached | 110/142 | 106/142 |
| Overall goals | 246/288 | 234/288 |

The apparent gain in rejecting false hints does not survive trajectory-outcome evaluation. This challenge shares a synthetic corridor template with training, not true CyberSand-native OOD evidence. The remainder of the D trajectory runs *frozen C12*, not the candidate.

## Provenance, safety and custody

- **C17-A** safetensors SHA256 `369b4734eab0abfb4ca6a6acd67246d80aa7a93b87904431b7e41e11db9ecaf2`.
- **C17-B** `24e4057d84227606526bd6c7c169355d3d91555a0210f327d22398b424c52f57`.
- **C17-C** `2bebd0d3ac78f550a68616e530009c66eefa6a869c5980fe0a9ad4cdecdf5ae7`.
- **C17-D** `9a6d3351045b09bc30ea038f3b746da652660132b97620cf1c9754a8797d4744`.
- Full source SHA256: `history_conditioned_trust.py = ccd972883016784f44d00737f015f93eae7349787cb24dd5f5b2d2c0805c6687`, `aliased_history.py = 18784c1bab9ab2a7b14355f1c97a54f3c464caf7aed121b4710822e98388aa64`. The archive also contains 130 passing local-source regression tests and real per-decision synthetic trajectories.
- The owner-facing off-Git, SHA-manifest-verified **86-file** research bundle is `driv3r_guidance_recovery_C17_history_trust_negative_2026-11-16.zip`, SHA256 `76558a45d95f87974888636f9668676b799096ee71ca40307dbb7b6350100cf1`. It contains source, CPU checkpoint replicas, all four variant weights, tests, prerequisites, raw rollout receipts, 20-panel SVG/HTML gallery, and reproduction instructions. **The binary is not in Git**. This commit preserves the research record, not a Git-integrated trainer; the owner archive contains exact tested Python source.
- Two complete *optimizer* repeats yielded byte-identical A and D safetensors. The entire evaluation was **not** independently repeated. The reported trajectories had zero synthetic host rejections, zero raw illegal top-1 actions, protected materials unchanged, and monotonic ink.
- Exact-head complete repository qualification remains outstanding because C5/C10/C15 tests were not included in the available offline archive. Immutable GitHub Actions checkpoint custody requires an authorised workflow owner outside this lane.

**Conclusion:** Do not promote C17. Prior success rate is a weak, confounded proxy for whether *this* cue is useful. A better experiment needs immediate, cue-attributed and host-verifiable error signals with hard misleading-cue non-regression constraints.

No game/native/Godot/Workbench/material/map/asset/default-provider modifications, real d1 training, paid computation, merge or Ready promotion.
