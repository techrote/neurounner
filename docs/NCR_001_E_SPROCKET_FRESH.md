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


## Verified hosted results — 27 October 2026

The original [GitHub Actions run 40469703065](https://github.com/techrote/neurounner/actions/runs/40469703065) **passed all three jobs**: seven static role/control-injection refusal tests and a pinned actual tokenizer/source inventory, plus independent frozen-host Chat and Instruct source/model checks. The source-bound prompt format is confirmed by actual README SHA-256 **09aec879dc5b387bdc031dc3c74d77a9ba44362a55163f9a8b456761b68142c9** (8,718 bytes). Both tokenizers assign **user=3, assistant=4, end=5**, tool_call=6, tool_result=7, memory_read=10, memory_write=11. These are **token recognition IDs**, not independently qualified action semantics.

The pinned README actually contains the publisher's separate Chat and Instruct variant descriptions. The first source CI revision had an over-rigid prose substring check that failed because of whitespace; the rule was corrected to check the literal role format strictly and **record** revision-specific descriptive claims instead of retroactively imposing them. The original failure remains visible in the Actions history.

| Actual frozen Sprocket source-bound check | Chat subfolder | Root Instruct |
|---|---:|---:|
| Actual checkpoint SHA-256 | 551c6e26…b5acefe97 | 1a9fd106…220d864 |
| Zero-init source-role logit parity at 16/64/128 units | Exactly 0 | Exactly 0 |
| Left/right padded batch parity | Exactly 0 | Exactly 0 |
| Greedy four-token zero-init generation | Identical | Identical |
| Manual nontrained readout's maximum logit effect | 0.0110900 | 0.0236578 |
| 48-position full vs incremental cached logits max abs | **5.48363e-05** | **9.91821e-05** |
| Causal future-suffix influence | 0 | 0 |
| Request A→B→A reset | Exact | Exact |
| New adapter on same frozen base, cold readout reload | Exact | Exact |
| Wrong variant, checkpoint hash or revision | Rejected | Rejected |
| Peak process RSS | 3,293,812 KiB | 3,293,376 KiB |
| Actual optimizer steps and model-quality comparisons | 0 | 0 |

Each full model was fetched and hashed at exact pin, using a **distinct** original integration implementation and its own checkpoint. This is not training and not a comparison of Chat vs Instruct capability.

### Full original source custody

The complete original JSON reports have been **losslessly preserved in Git** as gzip-compressed base64 text, not replaced with only a summary. Their original JSON byte SHA-256 and GitHub Actions artifact ZIP digests are independently archived in [provenance.json](../experiments/ncr-001/evidence-review/sprocket-fresh/provenance.json). The [standard-library auditor](../scripts/ncr_001e_sprocket_receipt_audit.py) decompresses all three originals, verifies byte lengths/SHA-256/source identities/tolerances and refuses an accidental claim that tool semantics or model training passed. A subsequent CI revision re-reads the archived originals before rerunning the live model tests.

| Original report | Exact original JSON size | SHA-256 |
|---|---:|---|
| Pinned README/tokenizer inventory | 8,008 bytes | 1e5dad585bf6ea07827df210dea3b24b43df20a241372525e1b42fa9d1818021 |
| Chat full source/cold/cache receipt | 6,527 bytes | 616d9f0876757b40c79b3c436fa89dbc4a199ad3ad16f3518532d64860309e1c |
| Instruct full source/cold/cache receipt | 6,681 bytes | af1ccdfad8350e5f1dd6e0ba07b6c6dfd8ce1396ab5c33981a9609d3200f87c0 |

To inspect an archived original JSON locally, decode the corresponding .json.gz.b64 file using Python's base64.b64decode then gzip.decompress, or run the committed auditor. The originals have **no modified field values or re-exported/normalised JSON**, so they can be compared byte-for-byte with the original uploaded Actions artifacts.

**Explicit stop:** Model-card claims and dedicated tool/memory tokens are not a verified tool-call argument/result or memory-write grammar; no held-out ability score has been computed. The parent four-model #72 acceptance remains **BLOCKED**, as does all comparative training.
