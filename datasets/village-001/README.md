# VILLAGE-001 durable archive

This directory is the durable archive root for the VILLAGE-001 36-member SLM campaign.

**Current status:** contract only. No generated village corpus is authoritative until #11 promotes validated provider/Sprocket records here.

The archive follows `docs/DATASET_FILING_SYSTEM.md`: raw records outrank derived summaries, orchestrator-owned IDs are canonical, and Actions artifacts are staging rather than the archive.

Planned layout:

```text
datasets/village-001/
  README.md
  manifest.json                 # created when the first durable corpus is promoted
  ancestors/
  member-corpora/
  sprocket-feedback/
  provider/
    minimax/
    nemotron-bolt/
    nemotron-super/
  training/
  preference/
  evaluation/
  provenance/
    source-artifacts.json
    hashes.sha256
```

No empty placeholder directories are committed. Each section appears when it contains durable evidence.

The exact 36 member identities are generated from `experiments/village-001/config.json` by `scripts/village_001_manifest.py`.
