# Dataset filing system

This document defines the durable archival contract for generated corpora, experiment conversations, model outputs, repair passes and derived training datasets in `neurounner`.

The governing rule is simple: **Actions artifacts are transient staging, not the archive.** Any experiment worth keeping must be promoted into the durable filing layout before its artifacts expire.

## Goals

The filing system exists to preserve:

- raw model generations, including failures, malformed output and contradictions;
- the exact prompts/messages and generation parameters that produced them;
- model/runtime identity and source revisions;
- runner/hardware observations when available;
- immutable orchestrator-owned IDs and lineage;
- raw external-provider responses before repair or normalization;
- derived/compiled datasets without overwriting their sources;
- enough provenance to reproduce, audit or reinterpret an experiment later.

It also exists to make those records easy for repository search, RAG and future agents to retrieve.

## Canonical location

Durable datasets live under:

```text
datasets/<dataset-id>/
```

Dataset IDs are lowercase, stable and never reused for a materially different corpus. Examples:

```text
datasets/babble-001/
datasets/kernel-compiler-001/
datasets/visual-observation-001/
```

## Required top-level files

Every archived dataset must contain:

```text
README.md
manifest.json
provenance/hashes.sha256
provenance/source-artifacts.json   # or equivalent source-runs record
```

`README.md` explains the dataset in human terms. `manifest.json` is machine-readable. `hashes.sha256` covers every durable file except itself. Source provenance records Actions runs, artifacts, commits or other origins.

## Recommended directory layout

Use only the sections that apply, but do not collapse raw and derived material together.

```text
datasets/<dataset-id>/
  README.md
  manifest.json

  guides/                  # high-level prompts, guide packets, seed specifications
  conversations/           # exact prompt/message histories + raw model responses
  raw/                      # source artifacts/logs in their least-transformed form
  provider/ or <provider>/  # remote API requests + raw provider responses/repairs
  compiled/                # normalized or curated packets derived from raw data
  training/                # final SFT/preference/other training shards, if produced
  prompts/                 # human-readable prompt/template notes
  provenance/
    source-artifacts.json
    hashes.sha256
```

Large collections may use `.jsonl` shards rather than one file per record. Keep text shards reasonably small (target <= 1 MiB where practical) so GitHub code search and RAG retrieval remain useful.

## Conversation record contract

Each generative interaction must preserve, directly or inside a JSONL record:

```json
{
  "conversation_id": "immutable orchestrator-owned ID",
  "model": {
    "id": "model/repository identifier",
    "revision": "exact revision where available",
    "artifact": "exact weight/quant file where relevant",
    "sha256": "weight hash where relevant"
  },
  "request": {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "max_tokens": 0,
    "temperature": 0,
    "top_p": 0,
    "seed": 0
  },
  "response": {
    "role": "assistant",
    "content": "raw response, uncleaned",
    "finish_reason": "..."
  },
  "lineage": {},
  "runner": {},
  "provenance": {}
}
```

Additional provider fields are welcome. Do not omit the original message text merely because a prompt can theoretically be reconstructed from a template.

## Raw-before-derived rule

Raw outputs are authority. Derived outputs are additional records.

Do not overwrite or delete a raw response because it was:

- malformed;
- repetitive;
- offensive or low quality;
- rejected by a validator;
- repaired by another model;
- superseded by a better compilation;
- internally contradictory.

Those failures can become negative examples, quantization evidence, mutation material or future research data.

A repair pass must store both the original and repaired response, plus a reason or validator result when available.

## External-provider rule

For cloud/API stages, preserve:

- provider and exact model identifier;
- endpoint class;
- exact request messages and generation controls;
- raw provider response body;
- extracted assistant content;
- usage/token accounting;
- HTTP/retry metadata when useful;
- subsequent repair requests/responses;
- a flag when the backend is remote/unpinned.

Never archive API keys, authorization headers, cookies or other secrets.

## Lineage authority

IDs and lineage are orchestrator-owned.

Models may propose lineage, but the orchestrator must verify and write the canonical source IDs. A compiled training packet should point back to the guide/conversation/babble IDs from which it was derived.

When a model returns incorrect lineage, preserve the raw model response and record the deterministic lineage repair separately.

## Provenance and hashing

Every archive promotion must record:

- source workflow/run IDs;
- source commit SHAs;
- artifact IDs and artifact digests when applicable;
- model revisions and hashes where available;
- archive creation date/schema version;
- SHA-256 inventory of durable archive files.

Hashes are integrity evidence, not semantic equivalence claims.

## Immutability and corrections

Treat durable raw records as append-only.

If an archival bug is found:

1. keep the original raw source;
2. add corrected normalized/derived records;
3. bump the dataset manifest/schema or record a revision note;
4. regenerate the hash inventory;
5. explain the correction in `README.md`.

Do not silently rewrite history in a way that removes the original evidence.

## Actions-artifact promotion rule

At experiment completion:

1. verify expected counts and unique IDs;
2. recover all successful partial artifacts, including outputs from failed workflows;
3. reconstruct/store exact conversation messages if the original worker record omitted them;
4. preserve raw provider responses and repair attempts;
5. write durable manifests/provenance/hashes;
6. commit the archive under `datasets/`;
7. only then treat expiring Actions artifacts as disposable.

A workflow being red does **not** mean its outputs should be ignored. Partial artifacts must be inspected and promoted if they contain unique successful work.

## Secrets and privacy

Never commit:

- API keys or bearer tokens;
- GitHub tokens;
- passwords/cookies;
- private credentials;
- private user data or material that is not intended for the public repository.

A public corpus should contain only material safe to publish.

## Large-data policy

This repository is optimized for searchable text corpora and metadata, not model weights.

For data too large or binary-heavy for normal Git:

1. keep a durable manifest, hashes, lineage and retrieval instructions in-repo;
2. store bulk assets using the mechanism selected in `docs/MODEL_STORAGE.md` or another explicitly documented external store;
3. never rely solely on short-lived Actions artifacts.

## Required signposting

`RAG.md` is the retrieval entry point for agents and must link this specification and the dataset index. New archival authorities should also be added there.

## BABBLE-001 precedent

`datasets/babble-001/` is the first archive brought into compliance with this system. It preserves 12 thematic guides, 48 exact Sprocket conversations, raw worker/server records, all retained MiniMax compiler/repair responses and 12 final personality/training packets.

It also records one legacy gap: the original Qwen3.5 guide-generation job retained parsed guides but not its raw pre-parse response text. Future experiments must not repeat that omission.
