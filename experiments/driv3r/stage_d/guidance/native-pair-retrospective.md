# D-1B — read-only original native worm-pair retrospective (15 Nov 2026)

**Reverification status (15 Nov 2026):** The recorded figures below were
produced by the earlier retrospective verifier. New D-1B source at commit
`7123ad18a0c04a51a28e680bf45811fbf335a57a` additionally requires
every archived raw Q8 System One answer to re-parse consistently with the
selected native action, complete numeric hints for **all legal cardinal
moves**, and matching presentation digests. **The two original ZIPs have not
yet been rerun under these stronger checks.** The 13/36 maximum-scent count
and prior JSON hash are therefore *historical findings pending re-verification*,
not newly qualified output. This does not negate the separate #611 native
source evidence; it limits what this lane can presently claim.

**Evidence class:** independently checked **archived real native material readbacks** from a separate owner's source-qualified #611 pilot. This is **not** a D-1B native session, D-1B formal A/B, actual feedback/Undo, visual d1 inference, artistic acceptance, or new training. This lane only reads copies of existing artifact ZIPs. No permissions or materials are modified.

## Immutable source and custody

- Original #611 native source: `313b836b3c9ce2c60e4659e3763df140f84347a6`; original [Actions run 43678354515](https://github.com/techrote/cybersand/actions/runs/43678354515); read-only [source reconciliation run 43691749089](https://github.com/techrote/cybersand/actions/runs/43691749089).
- Immutable M1 owner-accepted D-COMMON source: `8fa90de6be209ed0b7bd125039e6ec8b332fb1a6`; accepted schema SHA-256 `ebe2fa211148602c614cba0f0a40f285cba797784b70efa035832d833877f80d`.
- [Free native archive artifact 14147579972](https://github.com/techrote/cybersand/actions/runs/43678354515/artifacts/14147579972), original ZIP SHA-256 `33083b5642c9ae6af04d7530147818453dd4513b9cf4e6f9576e4ef79e0d9aac`.
- [Soft native archive artifact 14147233975](https://github.com/techrote/cybersand/actions/runs/43678354515/artifacts/14147233975), original ZIP SHA-256 `82d64e708e5706d4532900c60c52eaee7de04105d64888aaf436270136e7eca1`.
- Deterministic output `native-pair-retrospective.json`, full SHA-256 `027d3396633b59ed682c13a6c16ffb811f03fb04ccc0cf64ee1aea4919073901`. The output contains the source/ZIP/raw guidance identities, World hashes and counters, not the large historical ZIPs.

Run the analyzer in a Python 3 stdlib environment with independently downloaded unchanged original artifacts:

```bash
python3 -m experiments.driv3r.stage_d.guidance.native_pair_shadow free.zip soft.zip > report.json
python3 -m unittest experiments.driv3r.stage_d.guidance.test_native_pair_shadow -v
sha256sum report.json free.zip soft.zip
```

The reference report hash assumes the module's exact pretty-printed output and final newline, as written by stdout redirection. The ZIPs need no re-execution of native Workbench or model processes.

## Read-only measurements

| Original arm | Q8 choices | Actual native one-cell material-58 edits | Brush-up, material-preserving recoveries | Selected highest-u16 legal destination |
| --- | ---: | ---: | ---: | ---: |
| Free, neutral scalar | 36 | 33 | 3 | 33 / 33 (all ties at 0; **uninformative**) |
| Soft positive scalar | 36 | 36 | 0 | **13 / 36** |

The initial full 512×512 World material SHA and initial host-owned legal-candidate catalog hash agree exactly across both arms. First choice divergence **zero-based index 4 (the fifth choice)**; first consequential material-state divergence **also index 4**. The script validates archived Q8 chosen IDs against each turn's offered legal catalog, little-endian 16-bit field hashes and samples, full 512×512 stored int32 World readbacks, one-cell 0→58 receipt deltas, and zero material changes on cursor-only recovery. It validates the archived trial's own source identity; it does not rerun Q8 or independently capture a new native session.

Soft-arm sum of selected destination guidance = **915,912 u16 units**, sum of signed one-cell change along selected destinations = **+8,676**. This is one observed trajectory, not an independent or calibrated learned preference score. In particular, **13/36 highest-legal-scent choices means the observed d1 worm was not simply executing greedy maximum-H steering**. It does not by itself establish useful causal sensitivity to the guide. The neutral-arm 33/33 is entirely an all-zero tie and must never be interpreted as superior guide adherence.

## Qualification and limits

- Against the locally executed bytes that match committed Git blobs, **8/8 source/unit tests passed** (Python 3.13 stdlib), including exact World-delta and brush-up invariants, forged extra edit, source and u16 provenance substitution, and guide-as-permission rejection.
- The original native execution and reconciliation are separately evidenced by #611; this work is a **subsequent read-only verifier** and does not confer D-1B native M2/M3 or integrated five-record conformance.
- No framebuffer/real visual input to the Q8 system_one model, no KEEP/UNDO/STOP model critic, no actual Undo, no D-1B model call, and no formal A/B run. The original Q8 launch was text-only (`--no-mmproj`). GIF/PNG images in the historical artifact are native-material-derived, not evidence of vision-conditioned d1 decisions.
- **One paired seed** and changed post-fork Worlds imply neither an episode-level causal effect nor generalisation. Comparing max-scent choices is posthoc descriptive and the free-arm tie rate is vacuous. Controlled trials need separate owner authorisation.
- The full exact-head D-1B Python suite and M1 five-record integration checks are **not qualified by these eight focused tests**. All requested newer work remains in D-1B ownership.

**Status: Draft/HOLD, no merge, Ready, shared-contract adaptation, GPU/API/train dispatch, native material writes, real material rollback or production change.**
