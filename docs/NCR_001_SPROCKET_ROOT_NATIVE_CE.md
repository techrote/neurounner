# NCR-001E — actual frozen root Sprocket Instruct causal masked CE witness (NO TRAIN)

**Scope:** stacked draft on [#185](https://github.com/techrote/neurounner/pull/185), preserving the original pinned source-only shifted-mask qualification of [#184](https://github.com/techrote/neurounner/pull/184), [#181](https://github.com/techrote/neurounner/pull/181) and [#182](https://github.com/techrote/neurounner/pull/182), and the original successful and unsuccessful root frozen-host evidence in [#161](https://github.com/techrote/neurounner/pull/161)/[#162](https://github.com/techrote/neurounner/pull/162). Original commits and receipts are not modified. Owner's same-account scoped review override: [#72 checkpoint](https://github.com/techrote/neurounner/issues/72#issuecomment-6483647556).

## Narrow question

Does the **actual pinned root-Instruct frozen HF model** consume the original source-derived assistant-only labels, shifted causal targets, genuine END=5 delimiters and synthetic ID5 left-padding correctly when passed as a **native model loss**, with no-op NCA/MLP depth-hook controls sharing that identical model, on the original 12-turn public synthetic diagnostic?

The upstream source-only checks demonstrated 123 genuine serialized units, 45 supervised assistant positions, 26-token host-only prefix and 97 artificial pads, but did not execute native CE on real model weights. This strictly bounded new host job checks native `forward(labels=...)` CE against an independently coded `logsumexp - selected_logit` reduction over only the 45 causally shifted supervised targets. It also checks exact zero-initialized and disabled NCA/MLP logits/loss relative to the same native backbone, and unchanged backbone parameter fingerprint before and after. No gradients or optimizer are run.

## Source and causality conditions

- Root **Instruct** is `kandivault/sprocket-500m@eb157d4ddddc4619e661d4a2426bb2dd000fc668`, `model.safetensors` SHA-256 `1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864`; sibling Chat is **different** `chat/model.safetensors`, SHA-256 `551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97`. No cross-variant inference.
- Protocol and source-pins Git blobs stay `14ce5fbaa38f8806c8c8738f19c01b91385eebaf` and `59fff1aed872b5563aa6898171983973e0cdb02c`. `train_authorized=false`, four separate keys, no held-outs.
- The tested input is *exactly* the previously reviewed source-defined 12-turn diagnostic extracted from the original frozen-host script, not a new preregistered evaluation fixture. Original token, assistant mask, attention and new shifted target SHA values must match #161/#184, including 12 real END markers and mask hiding 97 synthetic ID-5 pads.
- Full-source weight and tokenizer file bytes are downloaded from the exact revision and re-hashed against permanent pins **before offline construction** with `trust_remote_code=False`. The original role formatter is checked out at its fixed Git tree. Both controls own the same frozen model/decoder lease; no trained adapters or learned quality claims.
- Six new toy-loss adversarial/positive tests reject missing or host-leaked supervised labels, nonfinite logits, out-of-vocabulary targets, and incorrect current-versus-predecessor logit selection. These are separate from the **new physical model** check and cannot substitute for it.
- Native vs independent mean causal CE comparison uses only an explicit **1e-5 absolute float32 reduction tolerance for the *loss diagnostic***. It does **not** change #72's frozen-logit/cache/causal tolerance `5e-4` or its separate acceptance requirements.

## Archive / result policy

Exact-head workflow checks [#185](https://github.com/techrote/neurounner/pull/185) ancestry, runs inherited original ZIP/log custody, then six toy loss tests, downloads and hashes the original 1,002,207,984-byte root Instruct checkpoint and performs frozen masked CE. Every phase persists positive/negative stdout, partial machine-readable receipt, source/runner details and model weight identity as an Actions artifact **including failure**. Passing transient Actions evidence must be separately moved into immutable Git original ZIP/log custody; it is not assumed permanent.

Even successful native CE proves only source-matched *synthetic masked causal loss*, not NCA learning or the validity of generated tool/memory operations. The original physical six-token `<|tool_call|>{"name":"` remains an incomplete negative. Chat, SmolLM2 and void.1 require their own actual-source gates; the historical void.1 BYTE-EOS impact remains **UNMEASURED**.

**Status until checked:** exact-head CI unverified, physical-host PASS not claimed. `full_72_conformance_pass=false`, `train_authorized=false`, optimizer steps=0, held-out access=false, comparative NCA/MLP training=false. No merges or changes to evaluation protocol/fixtures.
