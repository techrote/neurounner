# VILLAGE-001 — durable campaign archive

Current promoted stage: **VILLAGE-001D preference-matured population**.

The root `manifest.json` remains the immutable VILLAGE-001B member-corpus
manifest. Later campaign layers are additive and have their own stage manifests.

## Durable layers

- `member-corpora/` — 36 independently curated 160-row SFT corpora.
- `provider/` and `quality/` — raw VILLAGE-001B compiler/critic/adjudicator evidence.
- `sprocket-feedback/` and `sprocket-feedback-evidence/` — raw post-SFT Sprocket feedback.
- `preference/` — 36 member preference corpora.
- `provider/preference/` and `preference-quality/` — pair compilation/criticism evidence.
- `preference-manifest.json` — VILLAGE-001D population lineage and counts.
- `preference-hashes.sha256` — additive dataset inventory through VILLAGE-001D.

Model bytes remain outside Git. The in-repo preference evidence and adapter-release
manifest identify the exact `village-001d-recovery-40613393744-20261118` assets and hashes. Raw before/after
judge requests, responses, failed attempts and recovery receipts are preserved in
`experiments/village-001/preference-evidence/before-after-evidence/` and covered by
that evidence directory's `hashes.sha256`.

GitHub Actions artifacts are staging only; this tree is durable authority.
