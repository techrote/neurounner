# VILLAGE-001 experiment control

VILLAGE-001 scales the BABBLE/PERSONA line into a 36-member Sprocket-derived SLM population.

Authority:

- campaign design: [`docs/VILLAGE_001_CYBERGOBLIN_FOUNDRY.md`](../../docs/VILLAGE_001_CYBERGOBLIN_FOUNDRY.md)
- epic: #9
- execution issues: #10, #11, #12, #13, #14
- machine-readable configuration: [`config.json`](config.json)
- deterministic member compiler: [`scripts/village_001_manifest.py`](../../scripts/village_001_manifest.py)

The initial bootstrap intentionally does **not** launch the full cloud-generation/training campaign. The preflight first proves that all 36 immutable member identities resolve to the durable BABBLE-001/BABBLE-002 ancestry.

Expected member IDs are `V-G01-A` through `V-G12-C`.

The final durable data archive will live at `datasets/village-001/`. Generated adapters remain outside normal Git history and are represented by hashes/retrieval metadata in the repository.

## VILLAGE-001A contract gate

Issue #10 adds two explicit row contracts:

- [`member-corpus-row.schema.json`](member-corpus-row.schema.json) — member identity, ancestral lineage, positive/negative target and provider provenance;
- [`sprocket-feedback.schema.json`](sprocket-feedback.schema.json) — post-SFT raw Sprocket feedback with immutable parent/member identity.

`scripts/village_001_validate.py` is the deterministic semantic validator. It checks constraints JSON Schema alone cannot express, including member-ID ↔ lineage/sibling consistency, duplicate normalized inputs, source-guide mismatch and the rule that parsed feedback is required when a feedback record is marked valid.

`scripts/village_001_provider_preflight.py` performs sequential protocol checks against the configured NVIDIA compiler, critic and adjudicator. Authorization is never written to disk; all archived request/response/error records are scrubbed against the live secret before serialization. The provider gate fails closed if the secret is absent, a configured model is unavailable, transport fails, or the provider response shape is incompatible.

Provider preflight evidence is written under `experiments/village-001/provider-preflight/run-<github-run-id>/` by the VILLAGE-001A workflow.

`scripts/village_001_gate.py` is the reusable fail-closed launch gate for #11 onward. It requires a durable successful provider preflight and verifies that its endpoint, roles and exact configured model IDs still match the current campaign configuration. Any later change to a provider model therefore invalidates the old gate until a new preflight is archived.


## VILLAGE-001B member-corpus foundry

Issue #11 expands each immutable member from its 100-row BABBLE-002 ancestral shard to a 160-row SFT corpus.

- A/B/C use different fixed surface quotas; this is curriculum diversity, not a hyperparameter sweep.
- MiniMax M3.1 Flash compiles the 60 new rows/member from the exact BABBLE-001 Sprocket conversations, durable packet, and sibling specialization.
- Nemotron 3.5 Bolt independently scores behavioral fidelity, genericness, and style dependence.
- Borderline cases alone are sent to Nemotron 3.5 Super for sparse adjudication.
- Failed or adjudicator-rejected rows are repaired by the compiler and re-reviewed; unresolved rows fail closed.
- All provider calls are retained before parsing and repair.
- The Actions matrix has 36 member jobs but cloud concurrency is one. A manual rerun may provide a prior workflow run ID to reuse complete member artifacts and resume partial ones.
- Publication is all-or-nothing: the assembler requires 36 canonical 160-row shards before replacing the durable `datasets/village-001/` stage.

The corresponding implementation is `scripts/village_001b_generate_member.py`, `scripts/village_001b_assemble.py`, and `.github/workflows/village-001b-member-corpora.yml`.
