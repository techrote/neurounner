# STORY-001 durable archive

**The Shift That Kept Turning** is a 2706-word fictional first-person Sprocket memory story produced by a staged GitHub Actions experiment.

The incident is creative fiction. Repository Sprocket outputs were used as voice/source material; they are not evidence that the incident historically occurred.

## Pipeline

1. M0 — exact runner/runtime/model pin preflight with no cloud credential dependency.
2. M1 — three independent pinned Sprocket-500M memory-fragment generations on 'ubuntu-24.04'.
3. M2 — Qwen3.5-4B structural storyboard.
4. M3 — QwenStoryteller Q4_K_M eight-section long-form draft.
5. M4 — independent Qwen3.5-4B critique against the exact draft hash.
6. M5 — Qwen3.5-4B section-wise revision with aggregate length rebalancing only if required.
7. M6 — independent local qualification, durable promotion and SHA-256 inventory.

## Authority

- 'raw/sprocket/' preserves exact local requests, raw responses, server log and runner/model identity.
- 'local-models/requests/', 'local-models/raw/' and 'local-models/calls/' preserve the provider evidence and retries.
- 'compiled/storyboard.md', 'compiled/review.md' and 'compiled/story.md' are the promoted human-readable outputs.
- 'manifest.json' and 'provenance/' record qualification, run identity and hashes.

Raw evidence outranks derived prose when auditing what a model produced. All generation models are local GGUFs downloaded on GitHub-hosted runners and verified by SHA-256 before inference.

Actions artifacts were staging only and are not the archive.
