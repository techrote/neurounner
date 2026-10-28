# NCR-001 void.1 post-lease original physical evidence custody

**DRAFT / no training / no final #72 admission.** Parent #67, #72, #73 and BYTE-EOS-001 #92. This branch is strictly stacked on PR #167 at immutable original frozen source `9aed5ca0574a6885c5d60c07d16561e0f73d5043`, not main.

## What was physically executed

Original [Actions run 40747035876](https://github.com/techrote/neurounner/actions/runs/40747035876) finished **both jobs successfully**, including original pinned `dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563`, full model safetensors SHA `ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3`, exact remote ByteTokenizer source hash, and read-only corrected serializer #163 at `9fbdee880697903aaf5b91cc29442c9e53479e50`. The source job passed nine synthetic/negative tests; the physical job passed Qwen3 native causal shifted CE, zero/disabled frozen-logit parity for NCA and matched MLP, original shared-decoder reference isolation, manually activated *untrained* residual effect, 32/128-token incremental-cache checks, A-B-A and causal suffix isolation, and two separate-interpreter adapter binary cold reloads. The frozen parameter tensor SHA `7673957ba36e8eb5427e5ecc94dd52de71fbf9e6718bb0c2a94e4d7361b3f904` remained unchanged in producer and consumer.

Observed max absolute prefill/incremental cache errors (all within **unchanged** atol/rtol 0.0005): NCA 32 = `0.00001144409`, NCA 128 = `0.00001001358`; MLP 32 = `0.00000762939`, MLP 128 = `0.00001144409`. Native frozen losses on the three original synthetic samples were `8.765803337`, `3.065653801`, `4.957742691`; each exactly matched the independent shifted cross-entropy reported by that original frozen host. This is **not** learned stopping, behavioral competence or comparative effect evidence.

## Immutable original artifact and job boundaries

- Source Actions artifact ID `12842890264` is one complete ZIP, SHA-256 `f87e93a4c54a0a6fee7eda9c4d7db42a92006803134c554e69e9301a3e26646e`. Original job ID `123023455576`.
- Original Qwen3 producer/consumer Actions artifact ID `12840998564` is a complete ZIP, SHA-256 `ca581f11b0f4006170f6565322cff2f7eef72b5e6844e4c0e615961c05773144`, containing **nine members**, including both untrained binary adapter checkpoints and original producer/consumer receipts. Original job ID `123023557881`.
- Both original GitHub Actions jobs and both original artifacts are tied to the original source-head SHA, not the synthetic pull-request merge SHA. Original reported runner identities and separate process IDs are preserved.

The offline auditor `scripts/ncr_001_void_postlease_original_audit.py` independently checks the exact two ZIP byte strings, all ten original member SHA-256s, source, runner and frozen checkpoints, causal byte/EOS labels, three native forward losses, both adapters' untrained full/cache/cold paths and preserved failure/admission boundaries. Eleven adversarial mutations test false source, trainer, EOS, cache, causal mask, cold replay, frozen-integrity and ZIP admissions. It **does not** rerun the model and cannot substitute for independent scientific review.

The dedicated archival workflow only accepts a same-repository PR head descended from the original source, verifies `train_authorized=false`, downloads **the original** ZIPs and full decoded job logs through GitHub's Actions API, verifies those byte hashes and receipts, creates a manifest, and commits this evidence to **this draft branch only**, with a non-force fast-forward push protected by the original expected branch head. The workflow must succeed before asserting permanent Git custody; any 403, unavailable original artifact, changed ZIP byte or push failure is an explicit negative archive result. No scientific gate may be relaxed to make archival green.

## Qualification limit

**No comparative training, backpropagation, optimizer, opened held-out fixture, preregistration change, source pin drift or merge.** The exact matched control differs in parameter count by 472 parameters; neither arm was trained. The probe covered 32/128 rather than full 2048 context, with manually nonzero untrained readouts and only three synthetic EOS target cases. Final four-lane #72 acceptance, distinct independent reviews, source-bound learning/update gates where outstanding, and full void.1 cold/long/serialization qualification remain blocked. `train_authorized=false` remains enforced.
