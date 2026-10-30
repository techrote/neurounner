# NCR-001E — frozen Sprocket all-2,048-token incremental-KV qualification

**Purpose:** evaluate a specific missing #72 causality premise using an original frozen Sprocket checkpoint, rather than infer that last-four cache parity establishes every-token parity.

This is **not** comparative NCA/MLP training, a learned-quality evaluation, generated tool/memory competence, nor full per-model #72 scientific admission.

## Live main and immutable source

- Source branch starts at integrated `main` commit `b085d0cec41473b33d844acbf644b57304526779`. It uses the existing `scripts/ncr_001_postlease_sprocket.py` unchanged, exact Git blob `66e6b4d97eea5e917339b8e7bfb5d1c191db4a2d`. Original base #155 is merged; no original evidence is rewritten. All new changes are limited to this runner, source-only tests, CI, documentation and RAG.
- Four-model protocol Git blob `14ce5fbaa38f8806c8c8738f19c01b91385eebaf`, source pins Git blob `59fff1aed872b5563aa6898171983973e0cdb02c`. The exact preregistered cache comparison is `atol=rtol=5e-4`, no relaxation.
- Original source: `kandivault/sprocket-500m` revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`. Chat HF `chat/model.safetensors` SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97` vs distinct root Instruct `model.safetensors` SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. Both original frozen weight files are 1,002,207,984 bytes. Each job independently fetches originals and the runner hashes full files before loading. The independently pinned ordinary-role formatter checkout is `2271218d625e1462b2cf218445da975efe17fe32`.
- PyTorch `2.14.1+cpu`, Transformers `4.57.6`, huggingface_hub `0.36.2`, four CPU threads and original wrapper hooks/lease. A fresh actual checkpoint is loaded in **each separate Chat/root Instruct Actions job**. These are distinct GitHub hosts, not a shared cross-model compute benchmark.

## New original physical experiment

For each variant independently, serialize a **synthetic ordinary-user message**, take exactly 2,048 genuine non-padding token IDs and construct one full 2,048-position forward through an otherwise frozen model with the same deliberately **manual, untrained** nonzero NCA readout used in prior source tests. Verify original frozen-reference zero/disabled parity on 16 tokens first.

Then, using a fresh KV cache, pass **exactly one true token per call for all positions 0–2047**, preserving `past_key_values`, explicit `position_ids` and full prefix attention masks. At *every* position compare the returned `[1,1,32000]` scores against the matching position of the original full-sequence `[1,2048,32000]` logits, with the unchanged `5e-4` absolute/relative gates. Validate finite scores and one-token depth-state ownership on every call. Report every 512-position bucket and the global maximum, checked position, and raw partial progress every 128 tokens.

Ten **preregistered uncached prefix checkpoints** at 1, 2, 16, 64, 128, 256, 512, 1,024, 1,536 and 2,048 positions compare a fresh no-cache prefix score against the corresponding original full score. A 2,048-position same-length suffix mutation must have **zero** effect on the first 2,047 positions (exact comparison). Check that frozen backbone parameter bytes and gradients are unchanged, and retain peak whole-process RSS and runner/package identity. The experimental NCA forward never opens heldouts or invokes an optimizer.

The test is deliberately **B=1 without artificial padding**. The earlier **separate** original [#202](https://github.com/techrote/neurounner/pull/202) B=2 2,048-position left/right synthetic mask + final-four cached result is permanently in [#203](https://github.com/techrote/neurounner/pull/203); [#204](https://github.com/techrote/neurounner/pull/204) separately reconciles original NCA-only synthetic gradients. Do not transfer evidence from these other cohorts or mistakenly call this a new B=2 test.

**Critical limitation:** comparing all 2,048 *incremental* positions with a full forward is **not** the same as executing an independent, uncached full-prefix run for **each** of the 2,048 positions. Only the above ten uncached prefixes are explicitly executed. The report must set `all_prefixes_run_without_cache=false`, even if every streaming-cache comparison succeeds.

## Failures and evidence

Source-only workflow verifies exact head/source fingerprints and runs twelve adversarial tests, refusing changed model variants, reopened training gates, altered cache tolerances, invalid positions, nonfinite output, fabricated exact suffix parity and incorrect 2,048-token context. Physical jobs preserve **PASS, FAIL and partial receipts** using `if:always()` artifacts, along with original runner/pip/version/source identity; partial progress is recorded every 128 positions.

If an original physical run fails or times out, preserve the original log and partial report; **never** relax numerical tolerances or silently replace it with a later success. Successful source tests alone do not establish model qualification. Complete original Actions ZIP containers and original full job logs must be independently inspected and promoted losslessly into permanent Git custody before ephemeral 30-day Actions retention expires.

Historical negatives remain unchanged: the original root-Instruct generated only an incomplete six-token tool-call prefix; useful generated tool/result/memory semantics remain **NOT ADMITTED**. void.1's real EOS ID is 258, historical TRAIN-002b impact **UNMEASURED**; neither is measured here. No training, heldout, generated tool/memory approval, full four-model #72 PASS or #73 matched learned quality/FLOP parity follows from this test.

`train_authorized=false` · `full_72_conformance_pass=false` · `actual_compute_budget_qualified=false`. Draft/unmerged; owner-accepted same-account scoped reviews meet administrative gates but do not promote scientific evidence.
