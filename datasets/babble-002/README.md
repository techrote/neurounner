# BABBLE-002 — expanded personality training corpus

BABBLE-002 expands all 12 durable BABBLE-001 personality packets into a lineage-preserving supervised corpus.

## Corpus

- 12 personas
- 100 examples per persona
- 1,200 positive supervised examples
- 1,200 paired generic/sanitized negative targets
- source lineage on every row: guide ID, packet ID and the four BABBLE-001 babble IDs
- explicit trait and mechanical-seed references
- raw NVIDIA request/response/repair history retained under provider/

## Scenario balance per persona

- physical reaction: 15
- embodied decision: 15
- local memory: 15
- conflicting motives: 15
- failure and recovery: 15
- mundane low stakes: 15
- conversation/social: 10

## Training row

Each JSONL row contains example_id, source_guide_id, source_packet_id, source_babble_ids,
surface, input, target, generic_bad_target, why_bad, trait_refs, seed_refs and compiler provenance.

The positive target is intended for SFT/personality training. The paired generic negative is useful for
preference/ranking work and explicitly demonstrates what personality sanitization looks like.

## Authority

BABBLE-001 remains the authority for the original Sprocket babble and compiled persona packet.
BABBLE-002 is a derived training corpus.

Provider responses are preserved before parsing/repair. No API credentials are stored.

See quality/validation.json, manifest.json and provenance/hashes.sha256 for machine-readable evidence.
