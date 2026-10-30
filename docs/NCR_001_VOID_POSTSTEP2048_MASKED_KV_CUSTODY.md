# NCR-001E — permanent original custody for post-update B=2 masked incremental-KV

**EVIDENCE CUSTODY ONLY / NO MODEL / NO TRAIN / NOT FINAL #72 ADMISSION.**

Original physical source is draft #220 head `b3e9e6a05e007297de19f3955b59e2a9eb118996`, Actions run **41136709967**. Source job `124340840759` and physical job `124340899423` both succeeded.

## Permanent original streams

- source ZIP: 1,132 B, SHA-256 `972687a60ec5acd797a0cfea204b82f780fc425db2345a98059e001dc74bbab5`, Git blob `30a472aeeaa126110bd7afffffad644fb7b0498f`;
- host ZIP: 10,029 B, SHA-256 `638c5d564eb92a6f3d2a7df9fd36582e2761d736d4421fd67802f5796dee5eaa`, blob `51882755d6b2e2c48d5f73c7d72f14dcd8499170`;
- source complete decoded job log: 56,049 B, SHA-256 `9741fa2c7812b34e1f6e189480fb9a17f80737478ac637a18e3424ffb904cee6`, blob `32503c33c48d3212a561b5d4575833555df3ef99`;
- host complete decoded job log: 83,610 B, SHA-256 `03174257d3373aa21286e8a532e6d6428d42a048e3deecc193254327b8c799da`, blob `2b9124c37dd4090f6883e56f5249801be0c95994`;
- manifest Git blob `439584bd144c5b640cbc49807200ea3845b384cb`.

The host ZIP contains eight original receipt/log/source-identity members and **no adapter/checkpoint files**. The source ZIP contains both original test logs: inherited producer suite 10/10 and masked-KV suite 9/9.

## Physical result preserved, not reinterpreted

One public 31-unit / 21-target EOS=258 NCA-only step, followed by a new Python interpreter on the same runner.

Both layouts are width 2,048 with long row 2,048 valid units, short row 1,536 valid units, 512 synthetic pads and 64 real incremental continuation units per row.

| Metric | Left padding | Right masked-gap |
|---|---:|---:|
| B=2 full vs cached max abs | 1.6927719116210938e-05 | 1.6927719116210938e-05 |
| B=2 cached vs B=1 cached max abs | 9.059906005859375e-06 | 9.5367431640625e-06 |
| B=2 full vs B=1 full max abs | 8.627772331237793e-06 | 8.448958396911621e-06 |
| Row-swap cached max abs | 0.0 | 0.0 |
| Row-swap full max abs | 0.0 | 0.0 |

All are below unchanged `5e-4` protocol tolerance. Explicit masks and mask-derived position IDs were used. The right case is explicitly a **synthetic masked-gap sentinel**, not publisher/autoregressive grammar. Effective Torch threads: 4; consumer elapsed 232.60827 s; peak RSS 1,641,212 KiB; frozen-backbone fingerprint unchanged.

The one-shot archival run permanently committed the complete originals, then its write-enabled workflow was deleted. The retained acquisition script is readback/provenance code only.

## Read-only acceptance

The primary auditor binds:
- literal Git mode/type/blob for manifest and all four original streams;
- whole-stream bytes/SHA-256;
- ZIP CRC/member inventory;
- exact source/head/job/artifact identities;
- one-step public diagnostic and frozen source;
- both mask layouts, 64 genuine B=2 incremental steps, batch-vs-single cached references and row swap;
- no temporary adapter;
- all scientific/training/compute gates false.

Adversarial tests mutate/delete/symlink streams and forge cache/mask/scientific fields. Final CI must also re-download the four original Actions streams and require byte equality with Git custody.

**Locked:** `train_authorized=false`; `void1_conformance_pass=false`; `full_72_conformance_pass=false`; `actual_compute_budget_qualified=false`.
