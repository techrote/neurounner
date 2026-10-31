# NCR-001E — independent masked-batched 2048 KV + cold-process Sprocket qualification

**Scope:** two frozen source-distinct models, Chat `chat/model.safetensors` and root Instruct `model.safetensors`; no training or generated tool/memory semantic admission. This is a **new physical diagnostic**, not a replacement for original #95/#97/#100/#102/#113/#155/#161 receipts, the earlier negative six-token tool-call prefix, or the four-model #72 scientific decision.

## Immutable source and dependency contract

- Original base: [PR #155](https://github.com/techrote/neurounner/pull/155) `2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80`; use its proven native Sprocket NCA wrapper, frozen decoder-root lock, exact byte-verified model loading and source-pinned ordinary-role formatter. Never modify #155 or #102.
- Model source: `kandivault/sprocket-500m`, revision `eb157d4ddddc4619e661d4a2426bb2dd000fc668`. Chat full weight SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`; root Instruct `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`. Each complete file is 1,002,207,984 bytes and must be freshly hashed in its own CI job.
- Original ordinary-role formatter source `2271218d625e1462b2cf218445da975efe17fe32`. Protocol Git blob `14ce5fbaa38f8806c8c8738f19c01b91385eebaf`; source pins `59fff1aed872b5563aa6898171983973e0cdb02c`. Unchanged four-lane model IDs and preregistered 5e-4 absolute and relative cache tolerances.
- Package pins: PyTorch 2.14.1+cpu; Transformers 4.57.6; huggingface_hub 0.36.2; safetensors 0.8.0. Four CPU threads. Source/test/physical Actions run only on exact draft head with read-only contents permission and nonpersistent credentials.

## New independent observation requested

Each model lane independently executes a 2-row, 2048-physical-position synthetic batch. First row contains 2048 true source-ordinary token IDs. Second row contains 1536 true source-ordinary token IDs and 512 deliberately masked synthetic pads. **Both rows have exactly four true final tokens.** Left and right padding are separate tests, with per-example explicit cumsum-derived position IDs and full-length attention masks passed to every incremental step. The right-pad test intentionally introduces a masked gap before four valid final tokens: it is an adversarial causal/cache *transport* pattern, **not** a normal publisher conversation or generated tool/memory protocol.

The actual frozen NCA wrapper has a deterministic manually activated *untrained* readout. Its full forward's last four scores must match a genuine 2044-position masked prefill followed by **four calls of precisely one token**, preserving `past_key_values` and proving two separate `[2,1,...]` depth states at each step. Fail closed above existing 5e-4 numeric tolerances. A separate 64-position adversarial mask-corruption sentinel must have nonzero effect to demonstrate this harness detects accidental pad admission. The same original model is freshly loaded in a **second Python interpreter/PID on the same runner/cache**: serialized adapter bytes, original source/token IDs and both 2048 left/right last-four score tensors are compared exactly after cold reload. Frozen backbone before/after must be identical and have no gradients; report raw score tensors, unmodified adapter files, source SHA-256s, runner PID, platform/CPU, packages and RSS.

**Not claimed:** a different machine/GitHub account; separate-runner comparison of Chat and Instruct; publisher tool/result/memory competence; training loss/learning/stability; sufficient generation tests; full-context blanket acceptance beyond the two synthetic layouts; #73 matched FLOP or train budget; a four-lane #72 PASS.

## Evidence and refusal policy

Original Actions ZIPs/logs and receipt files from any SUCCESS or FAIL **must** be inspected and permanently Git-custodied on a separate descendant draft before the ordinary 30-day Actions expiry. Until those source-matched receipts and exact-head jobs are verified, status is **PENDING_EXECUTION**; a green source-test badge alone is not a physical-model PASS. Do not suppress/replace failures or loosen tolerances after seeing results. The negative original root-Instruct six-token tool prefix remains preserved. void.1 EOS=258 historical training impact remains **UNMEASURED**.

`train_authorized=false`, `full_72_conformance_pass=false`, `actual_compute_budget_qualified=false`. No optimizer, backward pass, sealed heldout access, comparative training or draft merge.
