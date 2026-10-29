# DETECTIVE-002A — diverse compact NPC body source qualification

Parent: [DETECTIVE-001 #103](https://github.com/techrote/neurounner/issues/103). Tracking issue: [#127](https://github.com/techrote/neurounner/issues/127).

This slice broadens the future NPC body population **without training any new detective character**. It is explicitly independent of the reviewer-blocked [DETECTIVE-001R1 #121](https://github.com/techrote/neurounner/issues/121): no blind relevance mapping is read or reconstructed.

## Why this exists

The current detective experiments proved useful behavior shaping on a SmolLM2-135M Village body, but also exposed policy interference at tiny scale. A future cast should not consist only of more adapters on one base model. The next population should include genuinely different compact architectures and pretraining lineages.

Stage A therefore asks only:

> What exactly is each candidate, can its source be frozen reproducibly, what license/runtime constraints attach to it, and is it eligible for a later no-training smoke?

It does **not** ask which body is the best NPC.

## Cohort

| Candidate | Class | Nominal size | Intended use |
|---|---|---:|---|
| SmolLM2-135M-Instruct | existing control | 135M | current Village/detective baseline |
| Sprocket-500M Chat | existing control | ~501M | chaotic direct-dialogue control |
| Sprocket-500M Instruct | existing control | ~501M | distinct root instruction/tool-wire control |
| LFM2.5-VL-450M text core | runtime reference | 450M | known very-fast compact runtime reference |
| SmolLM2-360M-Instruct | direct candidate | 360M | same-family scale control |
| Qwen2.5-0.5B-Instruct | direct candidate | ~500M | conventional cross-family instruction body |
| Qwen3-0.6B | near-band exception | **751,632,384 current API params** | tool/thinking-capable compact robot/control body |
| LFM2.5-350M | direct candidate | 350M | fast cross-architecture instruction body; license-gated |
| Zenyx-MCEN-Delta-Mamba | mutation only | ~565M | unfinished/custom architecture diversity source |

Qwen3-0.6B is intentionally **not** represented as <=700M merely because its name says 0.6B. The frozen source resolver requires the current Hugging Face safetensors total of 751,632,384 and keeps it in `near_band_exception`.

Zenyx is intentionally **not** a direct conversational candidate. It is a work-in-progress base checkpoint with custom Mamba/Gated-DeltaNet/MCEN architecture and is constrained to a future mutation/base-completion protocol.

## Existing immutable identities

The source resolver cross-checks existing authority instead of replacing it:

- SmolLM2-135M-Instruct revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, weight SHA-256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`;
- Sprocket-500M Chat and root Instruct share repository revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668` but use different weight paths;
- root Sprocket Instruct weight SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`;
- LFM2.5-VL-450M Q4_K_M runtime reference SHA-256 `1093f1331319199bbcacdbd7ecc9aa6e5678db6b55073948d0f508be90c8ab68`.

Chat/Instruct Sprocket checkpoint identity is a hard invariant. A resolver result that collapses them onto one path refuses.

## What Stage A resolves

Using Hugging Face metadata APIs plus only small `config.json` and `tokenizer_config.json` files:

- exact commit SHA;
- public/gated/disabled state;
- declared license;
- library/pipeline/model type/architectures;
- safetensors parameter total when available;
- declared weight-file path;
- LFS SHA-256 + size metadata;
- chat-template presence and tool-template indication;
- source last-modified timestamp;
- custom-runtime / instruction-model flags.

**Model weight bytes are never downloaded.**

The evidence records `weight_bytes_downloaded: 0`, `training_steps: 0`, `paid_api_calls: 0`, `relevance_mapping_accessed: false` and `population_expansion_admitted: false`.

## One-time freeze semantics

New candidates initially declare `source_ref: main` only as a resolver instruction.

On the first owner PR run:

1. query source metadata;
2. resolve `main` to a full 40-hex commit SHA;
3. collect LFS metadata and small-file hashes;
4. write `experiments/detective-002/source-evidence/`;
5. independently replay the archive;
6. stale-source-check the PR branch;
7. commit the frozen evidence to that exact branch.

After that commit, CI **does not re-resolve live `main`**. It only verifies the committed evidence and hash inventory. Stage B must consume `resolved_revision` from that archive.

This makes later upstream changes visible as a separate requalification decision rather than silently changing the candidate.

## License policy

Source qualification and production/game-distribution clearance are separate.

- Apache-2.0 candidates carry no special Stage-A production gate.
- LFM2.5 candidates declare `lfm1.0` and remain `production_license_gate: true` until a separate license review.
- Sprocket's source license is recorded as declared by the repository; if absent, the unresolved state is preserved and its production license gate stays true.
- A missing/changed license is never guessed or silently substituted.

## Stage-B eligibility

Stage A may produce:

- `stage_b_candidate` — direct/near-band conversational body whose source gates pass;
- `stage_b_reference` — existing runtime reference;
- `mutation_protocol_only` — Zenyx-style non-instruction research body;
- `blocked` — source, parameter, license or file-identity gate did not pass.

A Stage-A source PASS means **eligible for a later no-training runtime smoke only**. It does not imply good dialogue, game admission, commercial clearance or permission to train a larger cast.

## Stage B, later

A separate PR may download only the frozen promoted direct/near-band bodies and run a common four-CPU smoke measuring:

- startup time;
- RSS;
- prompt/decode throughput;
- deterministic short dialogue;
- contradiction/uncertainty handling;
- repetition/truncation;
- output-template behavior.

Mutation-only bodies get a separate base-completion protocol and are never compared as if they were instruction models.

No detective fine-tuning or population rollout is authorized while #121 remains independently unrated.


## Executed Stage A outcome

The source-freeze run resolved and archived all nine candidates. **9/9 source gates passed**.

Stage-B classes from the frozen evidence:

- `stage_b_candidate`: SmolLM2-135M, SmolLM2-360M, Sprocket Chat, Sprocket Instruct, Qwen2.5-0.5B-Instruct, Qwen3-0.6B near-band, LFM2.5-350M;
- `stage_b_reference`: LFM2.5-VL-450M Q4 text-core runtime reference;
- `mutation_protocol_only`: Zenyx-MCEN-Delta-Mamba.

The evidence corrected two provisional metadata assumptions before merge:

1. Sprocket source metadata declares **Apache-2.0**; Chat and Instruct remain distinct weights at the same pinned revision.
2. LFM2.5-350M and Sprocket expose standalone `chat_template.jinja` files, now hashed and retained by the resolver.

Qwen3-0.6B remains an explicit near-band exception at **751,632,384** API-reported parameters. LFM350 remains production/game-license gated under `lfm1.0`.

No weight bytes, inference, training or relevance mapping were accessed. The exact resolved revisions and LFS hashes in `source-evidence/` are now the only permitted Stage-B source authority.
