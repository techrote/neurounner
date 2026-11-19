# DG020 H35 — frozen authentic heldout donor-source invariant failure

**2026-11-18 · TRAINING-WORLD donor-source screen: FAIL. Prospective unseen-WORLD-family scientific gate: NOT PASSED.**

Existing Neurounner research PR #253 remains Draft/HOLD. No merge/Ready, no new PR/branch, no CyberSand or DRIV3R modifications, no paid compute.

## Registration and identity

[Locked donor-source registration BEFORE new H35 donor training and evaluation](https://github.com/techrote/neurounner/pull/253#issuecomment-6995219470).

- H35 posthoc head-only training source SHA-256: `54dcd02c003851d03a77392c93c350bfb7863014930022c97b8a4ce39d2f593b`.
- Locked H35 heldout genuine-neural donor evaluator source SHA-256: `c9b49ad976d2d610d85aec293dcb29dee7128dd86ed6cb4d4c7f53d6f567fb45`.
- All three receiver seeds `26011639`, `26011640`, `26011641`. Existing 209-parameter architecture. Hidden layers frozen; only output-head calibration on **original H27 exposed training worlds**.
- Newly held-out donor source seeds `29002143`, `29002149`, two independently trained replicas per law; eight genuine 4→24→24→1 tile-trained networks. 1,000 CPU AdamW steps each, fixed wet laws 0.35 and 2.30, dry law 1.0.
- All receiver weights, geometry, goals, legal actions, fees and texture identical within each counterfactual world pair. Only inherited donor-supplied material law feature swaps. No law IDs or oracle/family features at inference.
- Full negative archive with original donor/receiver checkpoints and raw receipts, evaluated scripts, logs, independent audit and manifests: `DG020-H35-heldout-genuine-donor-invariant-FAIL-20261118.zip` SHA-256 `b859a6ff899ea93fe47f457a0e9cf20fcd55dec7e461c8c88366299dd426d164`. This archive was supplied in the run handoff; it is **not** in GitHub custody.

## Fixed-head training-only repair, NOT prospective

H35 calibrated against all four original H21 genuine donor profiles and all four **now-exposed** H34 donor profiles on 7,680 original H27 training examples, with head L1 movement bounded at 0.002 per seed. The training-only calibration reached 7,680/7,680 both-law correct, zero within-law donor-source action changes and zero invariant law flips across these eight **fitted** profiles. That does not make H34 new-source generalization evidence for H35, because H34 donors are used in calibration.

Frozen H35 serialized receiver checkpoint hashes:

| Receiver seed | H35 checkpoint SHA-256 | Head worst signed fitted margin |
|---|---|---:|
| 26011639 | `2b2b809da8e3ac2e593c048cb727ea703e097abc2419fee9e70e3f8572651480` | +0.00050608 |
| 26011640 | `ccfa6d3bb9cbf4a4208c2d116653d98e6dbd0fc031885a1417ad3bed61b2b86e` | +0.00050864 |
| 26011641 | `cc466aa37cc4a3ace0a06b9893a654f1d699521247928ddbcbf3e0f5024b618d` | +0.00000744 |

The exact double-precision calibrated eight-element output-head vectors (before float32 cast) for restorations onto frozen H33b hidden layers:

```json
{
  "26011639": [0.9934562496771839,1.9903395175933838,-2.016361951828003,2.0694854259490967,-1.997772904154215,0.03671620338806413,-0.0,-20.29313087463379],
  "26011640": [1.0379393533295307,1.9665737133481551,-2.058803424559538,1.8568438280910842,-1.9936375617980957,-0.0,-25.116857528686523,32.19784164428711],
  "26011641": [0.9998714137200014,1.9809075171095392,-1.966172838376537,1.9656258821487427,-1.9866725206375122,-0.0,-0.0,-0.0]
}
```

## Heldout donor-source verdict: FAIL

| Preregistered aggregate metric | Result | Required |
|---|---:|---:|
| All-donor both-law correct | **7,678 / 7,680** | 7,680 |
| Divergent required law flips, both correct | 3,072 / 3,072 | 3,072 |
| Invariant pairs both-law correct | **4,606 / 4,608** | 4,608 |
| Inappropriate invariant law flips | **2** | 0 |
| Same-law genuine donor-source action changes | **2** | 0 |
| Zero-donor divergent both-correct | 0 / 3,072 | 0 |
| Wrong-law divergent both-correct | 0 / 3,072 | 0 |
| Shuffled-donor divergent both-correct | 714 / 3,072 | <=1,850 |
| Illegal actions | 0 | 0 |
| Original H27 observation hashes reproduced | 15,360 | 15,360 |
| Individual cells passing | **28 / 30** | 30 |

Both failures occur only at receiver seed `26011641` and on **new genuine donor source 29002149-rep0**, the wet-rich invariant bucket. All other donor profiles gave the oracle action. The failing observations were (a) `random_scatter` index 702, SHA `fd8516f108ae76c756345b7f45c2df7b73b48bb32cfd94b69401678a57f886c3`, and (b) `cyclic_run` index 1904, SHA `70edc9429fe1dc49efcc3f0bf44972d3c981307aa88812e2fc559d465af09dd8`. In both worlds the ground truth is action pair `[1,1]`; the offending genuine donor profile produces `[1,0]` (an **inappropriate invariant law flip**). Expensive-law true route margins were approximately 0.000505 and 0.000602; neural score margins collapsed to order 1e-6.

An independent script, which does **not import the H35 evaluator**, loaded original genuine neural donor and frozen receiver checkpoint weights, reconstructed donor quadrature features, and replayed all 7,680 world observations across 30 cells. It reproduced the **two exact failures**, verified every original world hash and all donor/receiver checkpoint hashes, and observed maximum absolute logit drift **0.0**.

## Disposition and bounded next change

**H35 HELDOUT DONOR-SOURCE SCREEN FAIL.** This is NOT a new prospective unseen-world/family verdict; the last stringent prospective cross-family result remains H27 **FAIL 28/30 cells**. The H34 30/30 donor-source PASS belongs to the original frozen H33b receiver and weaker preregistered thresholds on exposed training worlds; H35's stronger zero-error independent donor control reveals remaining instability.

Do **not** claim material-law inheritance or expand architecture/laws/topology. Next training-only mechanism should address the signed expensive-law route margin for seed 26011641 under genuine per-replicate donor uncertainty, without hardcoding law identity or tuning to the now-exposed two failures as supposed heldout success. Before another unseen-family claim, freeze complete receiver/donor states, source, seeds and strict both-direction, invariant, sham/zero/shuffle and family-holdout thresholds. Preserve all negatives.

**Custody limitation:** the H35 complete original checkpoints/raw receipts are in the handoff archive but not durable GitHub custody or exact-HEAD CI. This report preserves the original receiver reconstruction vectors and hashes, not the complete checkpoint bytes or raw receipts.