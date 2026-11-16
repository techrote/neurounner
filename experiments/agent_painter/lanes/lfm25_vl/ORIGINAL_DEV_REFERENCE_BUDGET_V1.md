# LFM original DEVELOPMENT teacher target budget v1 — reference-only

**15 November 2026 · original corpus only · no TRAIN, no model inference, not DEV60.**

This independently measures how many *official pinned Lfm2VlProcessor* tokens
are required to represent **complete canonical teacher responses** from the
original positively native-admitted six DEVELOPMENT structural groups, with
both source-native seed rows per group. It answers a narrower question than the
synthetic [DEV60 prefix probe](DEV60_PREFIX_PROBE_V1.md): *can the original
visible references fit an illustrative 512-new-token output ceiling, including
the real EOS7, with no truncation or target rewriting?*

The immutable original native source is
`633920373a3d406a2ede61ce4137c324f76be23b`; the admitted corpus manifest
SHA256 is `2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc`.
Only the checksum-verified `development.jsonl` is opened. The manifest's
TRAIN/DEVELOPMENT/FINAL *group metadata* is compared for split isolation, but
TRAIN and FINAL **file contents, targets, images and attachments are not opened**.
Both seeds of one source design remain **one independent group**, not two samples.

The probe uses source-lock checks for the untouched original adapter, profile
and shared trainer framing; downloads only eight original pinned processor/
license small files, uses the full **official** `Lfm2VlProcessor`, and invokes
the unchanged `adapter.encode_row` including its full end-at-EOS supervision.
The loader now resolves the hash-checked local adapter by exact file path,
ignoring any ambient pre-cached `sys.modules["adapter"]`; a model-free adversarial
test exercises this custody boundary. Neither upstream adapter nor framing was
changed.
No base model weights are downloaded. This is a reference representability test,
not a new full-model preflight, training, an evaluation generation, visual/native
painting or a model-quality pass. The frozen vision/projector, 92 LoRA projections,
numerical profile, license and baseline training recipe remain unchanged.

Output receipts preserve one hash of each unmodified prompt and canonical answer,
original group/seed identity, sequence and masked-prefix lengths, complete target
token counts, real EOS7, fit of the **unapproved provisional 512 ceiling**,
and paired-seed target-byte equality. They never publish answer text as model
output, and never treat 12 seed views as independent tasks. Full success proves
original *reference* encodability only. The future shared 60 authentic requests,
six additional DEVELOPMENT structures, semantic-equivalent five formulations and
approved per-model generation ceiling remain the shepherd/reviewer freeze gate.
The six original developmental references cannot be optimized on.

Run tests without the model environment:

```sh
python3 -m unittest discover -s experiments/agent_painter/lanes/lfm25_vl -p 'test_original_dev_reference_budget_v1.py' -v
```

The dedicated changed-source/explicit-dispatch GitHub-hosted CPU workflow
`.github/workflows/agent-painter-lfm25_vl-dev-reference.yml` downloads the
original positive native artifact in a fresh VM, verifies the original corpus
manifest, independently installs the exact pinned CPU processor dependencies
including `torchvision==0.29.2+cpu` and `Pillow==13.0.0`, retains partial
or failure evidence, then publishes reference receipts if and only if all
12 cases are fully encoded. No job is scheduled to train on a cadence.

No shared schema/compiler/registry changes, peer changes, FINALS, merged or
Ready PRs, default promotion, paid compute or extra training jobs are authorized.
