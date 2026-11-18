# DG020 H22 compact genuine-neural replay custody — 18 November 2026

**Scientific status is unchanged:** H22 prospective restricted gate PASS 30/30 under preregistered tolerances. This capsule performs old-evidence verification, **not new acceptance**, not full law transplantation, and not external methods review.

## Immutable self-contained handoff

An independently extracted **4,240,821-byte** ZIP was assembled entirely from original, byte-identical H22/H21R files: DG020-H22-self-contained-genuine-neural-replay-capsule-20261118.zip.

- Compact ZIP SHA-256: **47546540b96ae0800697e8a729c169e1191f111cec8c09f704c99a8843568667**
- Original 30,325,326-byte H22 ZIP SHA-256: **5a60ca1119a22916bda18ed46c0c6acecf77b4d891f5b1b841854a81533d1ca0**
- Original preregistered H22 source SHA-256: **a032a2b9057c150beb918e83e17ff8ad7e73a764fefcb4cf11cfd3049e1ef35d**
- Contents: 52 ZIP members, including 51 independently SHA-256-manifested files. All 30 original H22 raw receipt JSONLs, original report and 31-item seal, original locked source, independent checkpoint replay script, eight genuine trained H21R donor checkpoints, three H19 frozen receiver checkpoints, three source-blind receiver checkpoints, and model definitions. The capsule also indexes original negative-evidence archive hashes.

The full nested archive and original H17/H18 FAIL evidence remain preserved separately; this capsule does not rewrite or displace historical evidence.

## CPU verification

The repository-side **h22_compact_reproduce.py** first validates the exact compact ZIP SHA, rejects unexpected/unsafe entries, verifies all member SHA-256 checksums, and runs the original genuine-trained-donor + frozen-receiver neural replayer in a temporary directory. Neither the sealed source nor observations are modified or regenerated.

Independent local CPU PyTorch 2.10.0 / NumPy 2.3.5 qualification completed:

| Replayed historical result | Exact result |
| --- | ---: |
| Original H22 manifest | 31/31 |
| Matched observation/law pairs | 4,800 |
| Acceptance cells | 30/30 PASS |
| Divergent both-correct necessary flips | 1,920/1,920 |
| Invariant both-correct | 2,877/2,880 |
| Original inappropriate invariant flips | 3 |
| Same-law foreign and dry-only placebo changes | 0 |
| Pre-inference shuffled-donor divergent both-correct | 496/1,920 |
| Neural logit replay disagreements | 0 |

A one-byte-mutated ZIP was correctly rejected before extraction. This is a **local PASS** for the original restricted evidence replay, not a newly preregistered scientific PASS.

To reproduce once the ZIP is available locally, run the script on the ZIP path from the repository root: python experiments/memplant/material_causal_gate_020/h22_compact_reproduce.py DG020-H22-self-contained-genuine-neural-replay-capsule-20261118.zip

## Still open

**The new compact ZIP is an owner-handoff artifact, not yet uploaded to GitHub.** The code and identities are in PR #253 but **the actual source/checkpoint/receipt blob still lacks durable GitHub custody**, and no GitHub Actions runner has performed the complete H22 genuine-neural replay using this archive. The archive-contract workflow alone is not scientific replay. External methods signoff also remains outstanding.

PR #253 remains Draft/HOLD and unmerged. No added branches/PRs/issues, network GPU spend, law/topology changes, or CyberSand/DRIV3R writes.
