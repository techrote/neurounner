# NCR-001E — Source-anchored Sprocket role formatting and independent frozen-host replay

Parent: [#72](https://github.com/techrote/neurounner/issues/72), with separate Chat integration [#71/#86](https://github.com/techrote/neurounner/pull/86) and root Instruct [#78/#84](https://github.com/techrote/neurounner/pull/84).

This reviewer branch derives from the original read-only four-source receipt audit [#88](https://github.com/techrote/neurounner/pull/88). It adds both **independent, fresh actual-host invocations** and a **source-limited tokenizer/role audit**, not model training.

## Exact source and permitted ordinary format

Source authority is *kandivault/sprocket-500m*, pinned at revision eb157d4ddddc4619e661d4a2426bb2dd000fc668. The published README documents the ordinary chat prefill

    <|user|>your message<|end|><|assistant|>

and lists special tokens: system, user, assistant, end, tool call/result, thinking start/end and memory read/write. The source explicitly distinguishes **Chat** (the tool/memory examples were removed) from **Instruct** (the full tool/memory control-token SFT).

The candidate [ordinary-role formatter](../scripts/ncr_001e_sprocket_format.py) uses **only the documented user/assistant framing**, with no hidden dependency on an HF chat_template (none exists in the pinned tokenizer config). It refuses raw control tokens injected as message content, unsupported system/memory/tool messages, incorrect turn sequence and variant substitution. It does **not** invent the tool-call JSON, result-target grammar or memory continuation format.

The metadata-only [source inventory](../scripts/ncr_001e_sprocket_source_inventory.py) reads the **exact pinned README bytes**, hashes the actual root/Chat tokenizer and configuration files against committed pins, records per-variant actual control-token IDs and verifies the ordinary role framing tokenizes into distinct dedicated user/end/assistant markers. Tool/memory token *recognition* and *behavior* are not interchangeable. Root Instruct and Chat full safetensors must remain independent hashes and caches.

## Fresh frozen-host reviewer test

The [bounded workflow](../.github/workflows/ncr-001e-sprocket-fresh.yml) first validates the model-card formatter and pinned token IDs without model weights, then runs separate 4-CPU Chat/Instruct jobs, each with:

- Original independently authored hook implementation checked out at **exact, immutable PR head**: Chat 83a56835a7476f17a17980d28dd1a7c14bb6f2f6; root Instruct b95c010f9b2ba3cbd9f81317981316bb9c8acd19.
- Actual full 1,002,207,984-byte respective frozen model.safetensors downloaded and rehashed; exact revision/config/tokenizer/README files pinned before an offline runtime.
- Genuine source-formatted user/assistant token IDs, initial NCA 0-readout logit parity at prefixes 16/64/128, left/right masked padded batches and greedy generation on a complete source-formed prompt.
- Manually set a single NCA readout coefficient **without optimizer updates** to prove a nonzero effect, then compare full 48-position logit tensor with 48 **truly incremental one-token KV-cache** forwards at preregistered tolerance 5e-4. Explicit suffix/causality invariant and A→B→A request reset.
- New adapter object on the **same original frozen base** for cold adapter state_dict reload, zero-error parity and deliberate wrong-model-variant, wrong-checkpoint-hash and wrong-revision refusals. No different-process restart claim.
- Source-run, tokenizer-control IDs, model checksum, errors/partial failed receipts and runner RSS committed or staged with immutable SHA when actually verified.

The run deliberately does **not** use the existing held-out 16 Sprocket Chat/Instruct evaluations as prompts, does not score model/tool outputs, and does not optimize or train.

## Interpret carefully

A positive run proves **pinned source ordinary-role framing and a limited extended frozen-host cache/cold-adapter path**. It is not a demonstration that the Sprocket Chat variant can call tools (the publisher explicitly removed those examples), that Instruct's tool/memory semantics work under a production protocol, or that NCA improves anything.

Full independent four-model #72 still requires a separately approved Sprocket tool-and-result grammar, true cold-process reload/replay and accepted source/PR reviews. The four-model protocol remains train_authorized=false. All original unsuccessful attempts and numerical negative results remain evidence, not items to overwrite.
