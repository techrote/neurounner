# NCR-001E Sprocket Instruct source-wire diagnostic — original evidence

This is the durable **read-only archival record** for the original frozen-model
[source-oracle and bounded diagnostic Actions 40573188117](https://github.com/techrote/neurounner/actions/runs/40573188117).
Parent [NCR-001 #67](https://github.com/techrote/neurounner/issues/67);
source/runtime acceptance [#72](https://github.com/techrote/neurounner/issues/72);
[stacked draft PR #102](https://github.com/techrote/neurounner/pull/102).
All four model lanes remain separate. **No optimizer steps, training admission,
sealed held-out evaluation, actual external tool execution or final #72 PASS.**

## What this archive preserves

- [original-text-members.tar.xz.b64](original-text-members.tar.xz.b64) — unchanged
  original `receipt.json` (**75,464 bytes, SHA-256
  `94b1ff4f3ee4cb19d3e1e8cd57f8e9e7eee9534503b87a86d1aa6c5951de3b45`**),
  `plan.json`, `legacy-reproduction.json`, `prefetch.json`,
  `packages.txt`, `unit-tests.log` and `checkout-sha.txt`, in a deterministic
  xz-compressed USTAR envelope. The original **raw JSON and log bytes** are
  retained unchanged inside the archive; its transport compression is separate
  from and does not normalize their contents.
- [provenance.json](provenance.json) — original Actions run, artifact ID, original
  ZIP SHA, exact source/weight SHA, fixed source fixture/protocol, runner details,
  untrained model controls, negative observations and archival limitations.
- [manifest.json](manifest.json) and [provenance/source-artifacts.json](provenance/source-artifacts.json)
  — retrieval/lineage inventory; [provenance/hashes.sha256](provenance/hashes.sha256)
  — hashes of durable files.
- [reproducible auditor](../../scripts/ncr_001e_sprocket_wire_archive.py) and
  [tamper/false-admission sentinels](../../tests/test_ncr_001e_sprocket_wire_archive.py)
  verify exact original source bytes, immutable weights and runner, all seven
  cases and **21 raw outputs**, causal/cache tolerances, literal negatives and
  explicit false training/semantic-success claims using only the standard library.

Run from the repository root:

```bash
python3 scripts/ncr_001e_sprocket_wire_archive.py
python3 -m unittest discover -s tests -p 'test_ncr_001e_sprocket_wire_archive.py' -v
```

**Original artifact custody limit:** [Actions artifact #12766071456](https://github.com/techrote/neurounner/actions/runs/40573188117/artifacts/12766071456)
was **79,486 bytes**, SHA-256
`7e38a9c8438701c460597fc87016f3a26ece2d12b97482cfbcb44b0e6974a2ab`.
The original full ZIP and its 68,602-byte `source-snapshot.tar.gz` remain
in Actions retention, **not Git**; their original digests and sizes are
recorded without claiming the binary tar was archived. Original executable
source is independently available from immutable Git source head
`905e237136e4ea75208a8ed92392489a9f62849d`.
The full weight bytes are likewise not committed. All original text
evidence, including the failed/negative model outputs, **is** in this archive.

## Original frozen Instruct diagnostic observations

The pinned root `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`
`model.safetensors` (1,002,207,984 bytes, SHA-256
`1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`)
and original frozen hook source `b95c010f9b2ba3cbd9f81317981316bb9c8acd19`
were used. The model executed on the recorded four-vCPU GitHub-hosted AMD EPYC
runner; one **manually activated, untrained** NCA coefficient was compared
against disabled and zero-readout controls.

| Synthetic case | Literal/control observation (all three modes) | Stop |
|---|---|---|
| Tool request | No strict tool-call envelope emitted | 64-unit budget |
| Host tool result 42 | Expected `42` **absent** | 64-unit budget |
| Host tool result 17 | Expected `17` **present** | 64-unit budget |
| Host memory amber-17 | Literal `amber-17` present | Turn-end at 15 units |
| Host memory violet-28 | Literal present, including reasoning-like content | 64-unit budget |
| Memory-write request | No memory-write marker emitted | Turn-end at 6 units |
| Memory-read request | No memory-read action marker emitted | 64-unit budget |

Zero-adapter greedy output matched the frozen baseline in all seven cases.
All 108 positions of the source-structured multi-turn cache test passed at
**3.590583801269531e-4** maximum absolute error, below the unchanged
5e-4 absolute/relative gates. Frozen-vs-zero logits and A→B→A request reset
were exact, and frozen backbone tensor hashes matched before and after.
The 21 complete generated-ID sequences, prompts, decoded text, actual controls,
finish reasons, timings and negative observations are preserved unedited.

Literal copying is not semantic task success, and the observed failures **must not
be repaired, selected against or recast as a quality PASS**. Publisher source
syntax is pinned but not cryptographically attested to the historical model
training build. Neither Chat nor SmolLM2/void.1 inherits the Instruct result.

## Verification history, including failure

- Original actual frozen-model [40573188117](https://github.com/techrote/neurounner/actions/runs/40573188117):
  successful bounded diagnostic; not final admission.
- First archival mutation-test [40584453555](https://github.com/techrote/neurounner/actions/runs/40584453555):
  **FAILED** on three false manifest promotions that the audit did not yet
  reject (training allowed, tool/memory competence, final qualification).
  Original frozen-model/source evidence did **not** fail and was left untouched.
- [Commit `88cf5df`](https://github.com/techrote/neurounner/commit/88cf5df383556ec28246cbb93755014fb91bde95)
  tightened these three manifest flags without changing numerical tolerances,
  original receipts, model code, or sealed fixtures.
- Follow-up [archive-only Actions 40584618404](https://github.com/techrote/neurounner/actions/runs/40584618404):
  **PASS**, all nine negative/positive groups, `FRESH_MODEL_DIAGNOSTIC_REQUIRED=False`.

## Admission and custody limits

`train_authorized=false` stays unchanged; no NCA/MLP comparative training,
no sealed experiment evaluation, no migration of this data into training, no
draft merge or source/fixture gate relaxation. **A separate independent reviewer
must accept the exact final PR head and four model-specific gates before any
training protocol admission.** This diagnostic archive is source/byte evidence,
not a trained model or claim of learned tool/memory competence.
