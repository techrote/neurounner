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
