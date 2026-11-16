# VOID lane: reviewed byte contract and CPU handoff

This lane supports the exact public checkpoint
[`dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`](https://huggingface.co/dotlabs/void.1/tree/9c2be5087772867d45961abd11f6b35110617563)
through native Transformers `Qwen3ForCausalLM`. The
[profile](profile.json) freezes model/config/tokenizer/source hashes, architecture,
license, framing and LoRA targets. It does not grant native drawing authority.
Model proposals still pass the shared strict JSON/compiler/native admission
boundaries. Host seeds, material permissions, focus and application remain host
decisions.

**Qualification status:** the byte contract, all 12 admitted TRAIN budgets and
meta-device model/LoRA structure are checked locally. These are not optimizer or
inference evidence. A source-matched GitHub-hosted optimizer, serialized checkpoint
and fresh-runner resume result must qualify the lane before a worker starts.
Issue [#568](https://github.com/techrote/cybersand/issues/568) owns those actual
hosted results and schedule readiness. The owner authorized this preflight in
the [later programme direction](https://github.com/techrote/cybersand/issues/568#issuecomment-6865551592).

## Source and license custody

The pinned model card declares `apache-2.0`; the repository at this revision has
no separate LICENSE file. The original card, tokenizer implementation, model and
generation configs, vocabulary, special-token config, and upstream training
config are preserved byte-for-byte in [upstream](upstream/PROVENANCE.json).
[Apache License 2.0](upstream/LICENSE-2.0.txt) is included from Apache's official
license text, separately identified in provenance rather than attributed to a
nonexistent upstream file. Original authorship is `dotlabs` / `appvoid`, as
recorded in the original [model card](upstream/README.md).

The model weights are not vendored. Their pinned LFS SHA-256 is
`ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`
(360,622,744 bytes). The runner must download and hash the actual bytes; the
local metadata review alone does not verify a downloaded weight file.

`config.json` names native Qwen3 and has no `auto_map`. Only
`tokenizer_config.json` declares a remote `AutoTokenizer` implementation. The
downloader retains that reviewed Python file as inert evidence. The runtime
never calls `AutoTokenizer`, executes downloaded Python, or enables
`trust_remote_code`. The exact upstream implementation is executed only in the
optional offline conformance test after its committed SHA-256 is verified.

## Byte and framing contract

[adapter.py](adapter.py) provides `CyberSandVoidByteTokenizer`, an audited pure
Python equivalent of the upstream mapping for valid UTF-8 text:

| Symbols | Meaning |
| --- | --- |
| IDs 0–255 | One token for each UTF-8 byte, with no BPE or normalization |
| ID 256 | PAD |
| ID 257 | BOS, defined but not automatically inserted |
| ID 258 | EOS, explicitly appended and supervised after the complete target |
| Literal `<pad>`, `<bos>`, `<eos>` | Ordinary UTF-8 bytes, never injected special IDs |

Upstream replaces malformed Unicode and UTF-8 with replacement characters and
maps some invalid token representations to NUL. This lane deliberately refuses
such input/output, retains generated token IDs in the shared rejection receipt,
and never repairs bytes into accepted JSON. Whitespace cleanup and truncation
are refused. The tokenizer module imports no Torch, Transformers or PEFT.

VOID is a pretrained base model with **no upstream chat template**. The new
versioned supervised task framing is explicitly named `cybersand-void-task-v1`:

```text
UTF8("cybersand-void-task-v1\n")
+ UTF8(canonical_json({"instruction": system, "request": prompt}))
+ UTF8("\n")
+ UTF8(complete_canonical_admitted_program_json)
+ [258]
```

Canonical JSON uses sorted keys, compact separators and `ensure_ascii=False`.
The first three segments form the inference prefix and receive loss labels
`-100`. Every completion byte and the final integer EOS receives a supervised
label. No BOS is added. This is a new training format, not a claim that the base
checkpoint already understands instructions or painter JSON. Both training and
inference use the same `encode_prompt` implementation.

The frozen [TRAIN budget record](train-byte-budget.json) covers the original
source-qualified foundation corpus at
`633920373a3d406a2ede61ce4137c324f76be23b`, manifest
`2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`.
All 12 TRAIN rows pass full native-admission custody checks and fit:

| Family | Rows | Full sequence tokens, including EOS |
| --- | ---: | ---: |
| Greebles | 4 | 830–837 |
| Pipework | 4 | 605–615 |
| Architecture | 4 | 761–821 |

The longest supervised target is 486 tokens including EOS. No target or prompt
was truncated; no FINAL target or image was read. The model context is 2,048
byte tokens. Future corpus extensions must pass the same complete-sequence
budget check. Inference must also bound prefix plus generation budget together.

## Native Qwen3 and LoRA contract

The pinned configuration has 24 layers, hidden width 640, intermediate width
1,440, 10 attention heads, 2 key/value heads, head dimension 64 and tied
input/output embeddings. The runner loads local verified safetensors into the
explicit `Qwen3ForCausalLM` class on CPU float32 with eager attention.

LoRA covers `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and
`down_proj` in all 24 layers: **168 modules, 336 trainable matrices, 1,984,512
parameters** at rank 8, alpha 16 and dropout 0.05. Base weights, embedding,
output head and all biases stay frozen. Validation checks actual module names,
shapes, adapter membership and the complete trainable parameter set, rather
than accepting a target-name list alone. Meta tensors are allowed only by an
explicit private structural-test helper. Both `load_base` and the production
`validate_trainable` hook require materialized CPU float32 parameters.

The shared trainer owns CPU numerical setup, data custody, optimizer, sample
order, deadlines, cancellation, checkpoint inventories and resume equivalence.
The adapter exposes `load_base`, `encode_row`, `encode_prompt`,
`validate_trainable` and `set_training_mode`. `save_pretrained` writes its exact
tokenizer/vocabulary/special-token contract and immediately reloads the actual
files, checking UTF-8 and special-token equivalence. No remote `auto_map` or
invented chat template is emitted into checkpoints.

## Reproduce the focused preflight

Offline contract tests need only Python and the committed reviewed metadata:

```sh
python -m unittest discover -s experiments/agent_painter/lanes/void -p 'test_*.py' -v
```

In the shared pinned CPU environment, optional meta-only architecture and
upstream tokenizer conformance tests perform no model forward/backward or
optimizer step:

```sh
VOID_META_TESTS=1 VOID_TRAIN_CORPUS=/path/to/qualified/native/corpus \
  python -m unittest discover -s experiments/agent_painter/lanes/void -p 'test_*.py' -v
```

Executed locally on 2026-11-13 in the foundation's pinned CPU environment:
**14 tests passed**, including custody and full-byte budgeting of all 12 TRAIN
rows. The meta-only checks instantiate architecture/LoRA shapes without model
weights, inference or optimization. Hosted real training and fresh-runner
resume receipts remain separate required evidence; do not relabel these tests
as training.

## Exact fresh120 promotion-gate custody — 14 November 2026

The lane-only `completion_gate.py` is an extra fail-closed **promotion** check,
not another native admission or independent proof that the synthetic test fixture
performed training. It requires all 120 original canonical `progress.jsonl`
optimizer events, complete native-parent/model/source/numerics identities,
checkpoint inventory and adapter readback. Following independent P3 review, it
also pins the **actual original positively native-admitted TRAIN split** SHA-256
`5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1`,
the original compiler SHA-256
`4d8559805df1ff5f3f23277a7bca98e47fa8feb53337aa79e8546d0bb89132f6`,
and the exact set of twelve TRAIN row IDs from the source633 immutable artifact.
The full historical corpus manifest is
`2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`.
Neither DEVELOPMENT nor sealed FINAL target rows are used in this guard.

The model-free positive fixture uses **synthetic** adapter/optimizer bytes and
events but real frozen source/split ID constants; adversarial fixtures reject
valid-looking substituted TRAIN/compiler digests or missing/extra/duplicate
TRAIN row IDs before touching candidate application. New native-admitted
curricula must receive a **separately versioned campaign gate**, not silently
alter these pinned v1 identities or resume an incompatible checkpoint.
Original fresh120 training and model-native evidence retain their own qualified
source/run identity; this guard does not retroactively upgrade semantic quality.


## Exact seed568 sampler and real original fresh120 re-audit — 14 November 2026

The independent reviewer subsequently found that a self-consistent **permutation**
of the twelve genuine native-admitted TRAIN IDs could pass the earlier promotion
gate. The [lane-only correction](completion_gate.py) now freezes their exact
**family-interleaved, per-family SHA256(seed568:row-id)** order from the original
source633 curriculum, not just their set membership. Canonical compact-JSON
order SHA-256:
`9209cc4ea49ed24c9bf3d2e2ba7adf421072534f695317664a0b1f8f245c68ee`.
The synthetic positive fixture now uses that order; an adversarial fixture
swaps two valid IDs and rehashes/rebuilds all 120 optimizer events, canonical
progress bytes, receipt and checkpoint inventory, and must be rejected.
[Exact-source hosted model-free regression #43567424687](https://github.com/techrote/cybersand/actions/runs/43567424687)
completed SUCCESS, **16/16 tests PASS**. This fixes a future-run downstream
custody gate; the synthetic fixture is not genuine training evidence.

More importantly, a separate ordinary GitHub-hosted
[original-evidence audit #43567614273](https://github.com/techrote/cybersand/actions/runs/43567614273),
executed at source `9c6b86dc03ddedd018337f169e6da1fc082a265f`,
downloaded **the actual unchanged original** fresh120 optimizer/checkpoint
artifact `14087659446` from producer
[#43530053171](https://github.com/techrote/cybersand/actions/runs/43530053171)
(source `7d90bf5294d16a8644694a4c8a6a36a5227fa3c0`), plus
the separately admitted actual original source633 native-corpus artifact
`14076703326` from
[#43506373969](https://github.com/techrote/cybersand/actions/runs/43506373969).
The audit checked producer status, exact artifact names and published digests,
then **revalidated all twelve actual TRAIN native programs, material readbacks,
receipts and compiler geometry**, independently rebuilt the deterministic
seed568 order from those rows, and applied the corrected completion gate to
the **original real** training summary, 120 optimizer events, inventory and
serialized checkpoint members. **PASS, 120/120 receipts, eight checkpoint
inventory files, six distinct TRAIN structures.** Original progress SHA-256
`2bf580ca2cfddb0bb6d38ca2aeda9a38858420633b5170bef1d4fb47a9ca65a4`;
original checkpoint inventory SHA-256
`41ca76d96d107a210fa9a8515fc2d01b22bb61e842dd938987b69556afd55e0c`.
Small provenance/promotion report retained in
[artifact #14103291995](https://github.com/techrote/cybersand/actions/runs/43567614273/artifacts/14103291995).

This re-audit verifies historical **actual checkpoint custody**, not new
optimizer execution, new model inference, native application or model quality.
No sealed FINAL target/image was inspected. Its original candidate still has
only **1/6 distinct DEVELOPMENT compiler-accepted prompts**, **2/2 native seed
views from one accepted prompt**, and **zero request-conforming multi-branch
Pipework designs**. No useful `vN` is frozen. Further training remains blocked
on shepherd/reviewer-positive admission of structurally novel TRAIN groups and
a separately versioned expanded curriculum.
