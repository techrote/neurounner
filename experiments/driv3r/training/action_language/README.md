# GAL-001 — Game Action Language & Embodiment Lab

**Research-only / Draft-HOLD.** No CyberSand game/runtime modifications; no native Godot integration, W0 source-faithful claim, SLM residency, paid inference or native PAINT receipts.

## Execution

From this directory on Python 3.11+ (standard library only):

```bash
python -m unittest discover -s . -p 'test_lab.py' -v
python lab.py --train 60 --eval 12 --output /tmp/gal001
```

This is a **real optimiser experiment**, but the learned model is a tiny 18-feature linear softmax policy, NOT an SLM, LoRA or native game policy. It uses 60 independent on-policy seed-labelled training worlds per arm and 12 disjoint held-out worlds per arm. Per-decision REINFORCE-style stochastic gradient updates with EMA baseline and weak entropy regularisation use actual host-selected legal actions. The random-choice, untrained-zero-weights policy is evaluated on precisely the same held-out worlds and action vocabulary. STOP is available in every offered menu and is never silently masked.

The six nested action-language arms increment one new host capability at each stage:

1. `cell`: one-cell INK, STOP.
2. `sweep`: up to four *contiguous, individually checked* INK cells per model decision.
3. `lift`: pen-up crawl across empty or own ink only; protected/foreign refuse.
4. `retrace`: one adjacent recorded-history step, without material deletion or teleport.
5. `anchor`: `SAVE_ANCHOR` marker, maximum four; there is **no native-compatible anchor teleport or rewind-to-anchor**. A negative/control ablation until a real protocol is reviewed.
6. `jog`: bounded two-cell orthogonal turn (`JOG_A_B`) validated at both intermediate and final cells; would need a separately approved native one-cell receipt sequence before admission.

Actions are finite IDs chosen from a *host-authored* per-observation catalog. The model never returns material coordinates. The offer SHA-256 binds lease, seed, family, world material, cursor, heading, turn and catalog; mismatched offers refuse before mutation. Scenarios are 48x48 synthetic representations, materials `0` empty, `3` foreign, `5` protected and `58` own ink, with `96` maximum model choices, `128` ink cells and `384` logical steps. **None of these is an approved native CyberSand budget.** Native #611 remains 36 decisions/36 edits and M3 64 logical movements until owner-reviewed host changes. The external host applies atomic synthetic strokes; interrupted native multi-cell writes, Undo, STOP cancellation and authenticated per-cell receipts are NOT implemented or qualified.

## First local qualification (16 November 2026)

Executed standard-library tests **11/11 PASS**, and six independent real optimiser campaigns at `--train 60 --eval 12`, using train seeds 10000–10059 per arm and held-out 90000–90011 per arm. A train/test World may be from the same generative family, but the *seeds* do not overlap. This is a deterministic **local container** run; it is **not GitHub Actions execution**, and checkpoints are **not published as immutable Actions artifacts**. Binary/weight checkpoints must be archived in Actions before being treated as durable deliverables. Exact local checkpoint and per-trajectory SHA-256 digests are in the generated metrics.json.

| Action arm | Learned ink / decision | Untrained ink / decision | Learned ink (mean) | Learned decisions (mean) |
|---|---:|---:|---:|---:|
| cell | 0.9495 | 0.4082 | 46.00 | 46.75 |
| sweep | 2.8618 | 2.1063 | 85.83 | 29.75 |
| lift | 1.6629 | 1.0651 | 117.58 | 82.92 |
| retrace | 1.7032 | 0.9979 | 128.00 | 79.50 |
| anchor | 1.6572 | 0.7616 | 128.00 | 81.58 |
| jog | 1.4380 | 1.0879 | 128.00 | 90.42 |

Values reflect **the distinct learned policies with distinct legal action sets**, not identical policy weights, and do not establish a causal effect beyond the small deterministic held-out sample. A one-cell decision can never paint more than one cell, while a four-cell action can paint four; that mechanical gain is the primary expected advantage. Several arms show excessive early STOP under the untrained uniform model; untrained controls are deliberately not STOP-masked, so their poor raw totals are not evidence that learning discovered sophisticated art. LIFT/retrace/anchor often trade painter throughput for recovery/exploration; `jog` need not outperform simpler sweeps. Objective uses only local ink/novelty/turn/persistence proxies; actual artistic quality requires human evaluation of original PNGs, ideally blinded and matched by cells/turns.

## Evidence / provenance

`lab.py` emits a per-arm plain-text JSON **checkpoint (weights),** twelve held-out per-episode evidence summaries, 5 original 4x enlarged material PNGs and exact model-selection receipt traces per arm, plus one metrics.json. For reproducibility compare full SHA-256 of each file; generated gallery records contain model-chosen action IDs, offered SHA-256, prior and final World material SHA-256, lease, original selection string and explicit `native_receipts=0`. Renderings come from the synthetic stored material array; they are **not native Godot snapshots**. Four models sharing one wrangler, LFM/Smol actual inference process speed/memory, camera/vision and real neural-game acceptance remain outside this experiment.

## Proposed interface only (not implemented in game)

Offer request: `{lease_id, world_epoch, source_sha, model_sha256, cursor, heading, material_focus_sha256, protected_focus_sha256, legal_action_ids, offer_token, decision_budget, ink_budget, logical_move_budget}`; model response `{offered_action_id, offer_token, original_response_bytes_sha256}`. Host must recompute the allowed swept path, execute *one native single-cell operation at a time* with authenticated cursor/material acknowledgments, independently count changed material cells, record partial-transaction state, and expose a real STOP/cleanup/Undo-history contract. No anchor teleport, screenshot/vision or 128-turn exemption is requested. Coordinate with #627 Commander bridge and #629 lease scheme; preserve #611 frozen evidence.

**Next exact task:** run this experiment as an isolated source-pinned hosted CPU job that uploads checkpoints and original gallery as immutable Actions artifacts, independently audit the recorded hashes, then repeat an appropriately sized matched-seed evaluation with fixed policy cross-arm counterfactuals and stronger held-out custody. Only then consider a separate owner-authorised native adapter proposal.
