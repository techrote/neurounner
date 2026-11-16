# DRIV3R research migration and ownership

Owner direction, 16 November 2026: **Neurounner** owns model research, architecture probes, Frankenlayers/Mutagen, quantisation, corpus design, training, evaluation and orchestrated research experiments. **CyberSand** owns its game engine, Workbench, native simulation, observation/action boundary, real gameplay testing and integration acceptance.

This is a single **Draft / HOLD** source-preservation effort, not a production merge, game change, or scientific requalification. The one-shot migrator imports research-only directories from 17 original CyberSand PRs, verifies the Git SHA-1 of **every original source blob**, writes a machine-readable per-file manifest, and removes its own workflow when done. The original CyberSand PRs, Git history and Actions evidence remain accessible; model checkpoints and external evidence artifacts are **not** claimed to be transferred.

Before closing a source PR: require successful migration, an accurate manifest, preservation of original evidence references, and accounting for any related integration changes that deliberately stayed in CyberSand. Do not abandon or merge game-facing PRs as part of this operation. Subsequent research work should use a small number of long-lived programme branches/PRs in Neurounner rather than one PR per experiment.

Fourteen previously active CyberSand-directed hourly model-research automations were paused during migration. Resumption requires explicitly rewriting their prompts to target Neurounner; do not resume the old instructions against CyberSand.
