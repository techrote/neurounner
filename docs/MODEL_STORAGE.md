# Model storage policy

neurounner should remain a small source, results and manifest repository. Multi-gigabyte GGUF weights are not ordinary source artifacts and should not be committed to normal Git history.

## Git repository storage

GitHub currently recommends a maximum compressed repository size of 10 GB for the .git directory. A normal Git object is hard-limited to 100 MB, and a single push is limited to 2 GB.

Therefore GGUFs should never be committed as ordinary Git blobs.

## Git LFS

Git LFS stores a pointer in Git and the binary object separately.

Current GitHub limits:

| Plan | Maximum LFS object | Included LFS storage | Included monthly bandwidth |
|---|---:|---:|---:|
| Free / Pro | 2 GB | 10 GiB | 10 GiB |
| Team | 4 GB | 250 GiB | 250 GiB |
| Enterprise Cloud | 5 GB | 250 GiB | 250 GiB |

The included storage and bandwidth allowance belongs to the GitHub account. Creating more repositories does not create a fresh independent account-level free LFS allowance.

LFS is awkward for the current Q4 corpus: several 3-4B files are around 2.1-2.6 GB, and LFM2.5-8B-A1B is about 4.8 GB.

## Release assets

GitHub Releases permit up to 1000 assets per release. Each individual asset must be under 2 GiB. GitHub currently documents no aggregate release-size or release-asset bandwidth limit.

This can be useful for immutable tooling bundles or split model archives, but splitting every GGUF into sub-2-GiB pieces adds avoidable assembly and provenance complexity.

## Actions cache

The default Actions cache limit is 10 GB per repository and idle caches are normally evicted after seven days. Eligible repositories can configure larger limits, but cache remains an execution optimization rather than an archival model registry.

The neurounner core cache should stay small: pinned llama.cpp, Sprocket and LFM1.2. Large qualification candidates should normally download directly to ephemeral runner storage, verify SHA-256, run, then disappear with the VM.

Do not create extra repositories merely to multiply Actions cache quotas.

## Preferred hierarchy

1. Hugging Face upstream plus pinned SHA-256: default for public model weights.
2. neurounner Actions cache: only hot models used in many jobs.
3. Ephemeral per-runner download: large or infrequently used candidates.
4. ilikebigweightsandicannotlie: manifests, indexes, checksums, split-asset experiments, or genuinely self-produced weights that need GitHub-side staging.
5. Additional storage repositories: only when there is an organizational reason, not to evade a quota.

## When a dedicated weight store becomes useful

A dedicated storage mechanism becomes worthwhile when upstream revisions are unstable or likely to disappear, neurounner creates its own fine-tunes/checkpoints, repeated downloads become a meaningful fraction of runner wall time, a multi-hour experiment needs several large models on the same runner, or we want long-lived reproducible snapshots independent of upstream hosting.

At that point, prefer a real large-object or model registry. Hugging Face/Xet is naturally suited to multi-gigabyte model files; GitHub can remain the experiment-control plane.
