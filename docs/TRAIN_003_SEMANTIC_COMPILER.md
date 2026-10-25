# TRAIN-003 — Semantic mechanical compiler qualification

Issue: [#4](https://github.com/techrote/neurounner/issues/4)  
Qualification run: [40145401600](https://github.com/techrote/neurounner/actions/runs/40145401600)  
Qualification source: `46ce9068005f48fe791b589028373317c63e6cf8`  
Model: `HuggingFaceTB/SmolLM2-135M-Instruct`  
Base revision: `12fd25f77366fa6b3b4b768ec3050bf629380bac`

## Result

TRAIN-003 replaces TRAIN-001's 30-example format-oriented corpus with a durable semantic compiler corpus and a family-disjoint held-out qualification.

The qualified TRAIN-003 LoRA materially outperformed the authoritative TRAIN-001 stage-2 adapter on the same 128 held-out examples and identical evaluator contract:

| Metric | TRAIN-001 stage 2 | TRAIN-003 | Delta |
|---|---:|---:|---:|
| JSON rate | 0.796875 | 1.000000 | +0.203125 |
| Exact six-key schema | 0.796875 | 1.000000 | +0.203125 |
| Core mechanical schema | 0.796875 | 1.000000 | +0.203125 |
| Decision accuracy | 0.000000 | 1.000000 | +1.000000 |
| Mechanical score | 0.291780 | 0.904742 | +0.612962 |
| Overall score | 0.283934 | 0.933319 | +0.649386 |

The evaluator deliberately keeps the mechanical score independent of the new `decision` field and exact six-key schema. TRAIN-001 therefore receives credit for mechanically correct legacy-style output instead of losing solely because TRAIN-003 adds `VALID / REPAIR / INTERESTING_BROKEN / REJECT`.

## Corpus

The durable corpus is under `datasets/train-003/`.

- 896 total reviewed examples.
- 768 training examples.
- 128 held-out examples.
- 24 training mechanism families.
- 4 held-out mechanism families with zero family overlap: `wall_trace`, `asymmetric_step`, `flash_fascination`, and `darkness_hold`.
- Four balanced decisions: `VALID`, `REPAIR`, `INTERESTING_BROKEN`, and `REJECT`.
- 192 training examples per decision and 32 held-out examples per decision.
- Deterministic expansion from 28 curated mechanism-family specifications.
- Deterministic checks for unique IDs/inputs, exact output keys, balanced labels, train/held-out isolation, state/update/action separation, local-only non-reject outputs, and reject canonicalization.
- Durable manifest, source specification, system prompt, SHA-256 inventory, and source-artifact provenance.

Qualification identities:

- manifest SHA-256: `d487a886654f4c8e815a5f6d79b67f067b520684e9db6cb52c46a7dcdfb353d5`
- source-family SHA-256: `899a2070884c8de8d06204a712e3349a6db25734af9498b2c9c10a7a0a733897`
- system-prompt SHA-256: `db169ae6426ba975b576c44e4aed310fccaa5543d6eb50810a8c2322eac594a2`

## Training measurement

TRAIN-003 uses the same pinned 135M base identity and LoRA shape as the TRAIN-001 CPU proof: rank 8, alpha 16, dropout 0.05, and the available projection modules among `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

The qualification performed one deterministic 768-step corpus pass on a GitHub-hosted `ubuntu-24.04` runner:

| Measurement | Value |
|---|---:|
| Training examples | 768 |
| Steps | 768 |
| Sequence tokens | 197,064 |
| Train seconds | 1,338.260 |
| Sequence tokens/s | 147.254 |
| First loss | 2.871297 |
| Last loss | 0.011884 |
| Mean loss | 0.173881 |
| Peak RSS | 2,170,948 KiB |
| Adapter bytes | 9,829,630 |

The candidate adapter remains an Actions model artifact rather than being committed to Git. Its evidence inventory records `adapter_model.safetensors` SHA-256 `2a29f80fd71ce001582b72071d3a979eb224a8415fe2570c5ccb422b034698c1`.

## Held-out semantic results

Mechanical scores by decision:

| Decision | TRAIN-001 | TRAIN-003 | Delta |
|---|---:|---:|---:|
| VALID | 0.408231 | 0.871489 | +0.463258 |
| REPAIR | 0.419717 | 0.894624 | +0.474907 |
| INTERESTING_BROKEN | 0.339172 | 0.852854 | +0.513683 |
| REJECT | 0.000000 | 1.000000 | +1.000000 |

TRAIN-003 achieved 100% decision accuracy and 100% exact-schema/JSON rates in every decision group. The lower-than-perfect non-reject mechanical scores are retained rather than rounded away: the model can classify and structure the unseen mechanisms correctly while still substituting or weakening details such as a specific causal failure description.

## Authoritative baseline

The baseline is the actual TRAIN-001 stage-2 adapter recovered from Actions run `40110636171`, not a recreation.

- TRAIN-001 source head: `4a9d183fb5aa85df94e90a77393641edb29083f1`.
- Stage-2 global step: 40.
- Original artifact: `train001-stage2-a1`.
- Original artifact digest: `sha256:ad964bcd554522e1f5fd8335f37adbd41732c38accc634d23dc28426c81b0537`.

The baseline and candidate evidence confirm an identical base model, base revision, evaluator version, dataset manifest, and held-out shard hashes.

## Durable evidence

Raw held-out responses and the direct comparison are committed under `experiments/train-003/evidence/`:

- `train001-heldout.json` — all 128 authoritative TRAIN-001 responses and per-example scores.
- `train003-heldout.json` — all 128 TRAIN-003 responses and per-example scores.
- `comparison.json` — aggregate and per-decision direct comparison.
- `provenance.json` — qualification run/source plus Actions artifact IDs and digests.

The original qualification artifacts were:

- corpus: `train003-corpus-a1`, artifact `12581175003`, digest `sha256:f2a0c261ee8a3bd12432afae5f84ac9b40e1741636e4bfd8bdf0f9243b0bb4e5`
- TRAIN-001 evidence: artifact `12581031568`, digest `sha256:77ce44e874fe2be5d1503e65b6cae2e7573bba7ec4fb4d59a85df3d9ce7f64a9`
- TRAIN-003 candidate: artifact `12581298433`, digest `sha256:69e1cf3df1b42bebccf68e8aefbe2b54feb6b5f56fef8a47b48a4d86209caff4`
- comparison: artifact `12580734353`, digest `sha256:3e9305632c3e2b6ec95e1e5b2ca02a91b6ba200685e0add1dfbce784cbc30721`

## Interpretation and limits

TRAIN-003 clears the issue's required transition from format learning to explicit semantic supervision: it teaches the four decisions, preserves deliberately odd local mechanics, separates bounded state/update/action, rejects nonlocal/future-information requests, and measures causal/mechanical content on mechanism families absent from training.

This is still a controlled synthetic qualification, not proof of unrestricted semantic generalization. Training and held-out examples share the deterministic prompt grammar, and the mechanical scorer combines lexical concept coverage with structural/locality checks rather than an independent human or learned judge. The raw outputs expose remaining imperfections, particularly in precise failure-mode wording. Future corpus work should increase linguistic and causal diversity without changing this TRAIN-003 evidence after the fact.
