# BABBLE-001 durable archive

Permanent repository archive for the first babble/personality dataset.

## Contents

- guides/ — parsed Qwen3.5 guide packets and exact guide-generation requests.
- conversations/ — all 48 exact Sprocket system/user/assistant histories with generation parameters and runner metadata.
- raw/workers/ — original Sprocket worker JSON artifacts unchanged.
- raw/server-logs/ — original local llama-server logs.
- nvidia/requests/ — exact reconstructed initial MiniMax compile requests.
- nvidia/raw/ — every raw MiniMax compile/repair provider JSON and assistant-content file preserved from the compiler artifact.
- compiled/packets/ — 12 final derived personality/training packets.
- compiled/manifest.json and compiled/calls.json — compiler summary and call records.
- provenance/ — source run/artifact identities, exact compiler source, and SHA-256 inventory.

## Authority

Raw conversations and provider responses outrank compiled packets when investigating what a model actually produced. Derived packets never erase their sources.

## Legacy gaps

BABBLE-001 predates the filing specification in two small ways:

1. the guide-generation workflow retained parsed Qwen3.5 guides but not the raw pre-parse response text;
2. NVIDIA repair request bodies were not stored verbatim, although raw repair responses and the exact compiler source defining repair prompts are retained.

Future experiments must preserve these directly.

No secrets are stored here.
