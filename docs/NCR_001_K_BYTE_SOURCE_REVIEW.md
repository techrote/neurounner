# NCR-001K — independent pinned void.1 ByteTokenizer source review

**Scope:** static source and risk review only, not model/text qualification or permission to train. Review branch [PR #90](https://github.com/techrote/neurounner/pull/90) is stacked on the original Qwen3 integration [PR #85](https://github.com/techrote/neurounner/pull/85).

## Source and evidence

Pinned file: dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563/tokenization_byte.py — **4,458 bytes**, SHA-256 **279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03**. Byte vocabulary SHA-256: **57a1cd2e70dd5b7cc6fd2eaa12ebfcd173c1620fa5d45c2c6282511c29b6a5f6**.

[Actions 40462933022](https://github.com/techrote/neurounner/actions/runs/40462933022), job 122051518627, verified four source-file hashes, parsed AST without importing or executing the downloaded Python, and printed numbered full source lines. Original source-audit artifact 12714879401 is temporary (ZIP SHA-256 aa7ff9d9e8d8ab21ca946e17ec51fe37527a73499c2e18d0c4231dc646c5fcf5). Result: PASS_SOURCE_HASH_AND_AST_INVENTORY_ONLY__RUNTIME_BLOCKED.

## Manual static findings

The 138-line implementation defines ByteTokenizer(PreTrainedTokenizer), importing only Python json, os and transformers.PreTrainedTokenizer.

- **No observed shell, dynamic execution or network path:** no eval, exec, subprocess, socket, requests, reflection-based loader or dynamic import call appears in the complete AST/source.
- **Filesystem operations:** os.makedirs, os.path.join and open(path, "w") occur only in save_vocabulary() at lines 129–137. This method creates/writes a JSON vocabulary file in a caller-provided directory. The runtime probe must not call it.
- **Byte vocabulary:** 0–255 are byte IDs, PAD 256, BOS 257, EOS 258, vocab size 259; no UNK.
- **Encoding:** _tokenize at line 67 maps text.encode("utf-8", errors="replace") to individual byte tokens. Exact for ordinary valid Unicode, **not malformed/lone-surrogate input**.
- **Decoding:** convert_tokens_to_string assembles a byte buffer and decodes with errors="replace". Invalid standalone byte sequences therefore do not losslessly round-trip.
- **BOS/EOS:** build_inputs_with_special_tokens at lines 119–123 returns the provided input IDs unchanged. BOS and EOS are **not inserted automatically**. Callers must confirm the actual EOS ID rather than assume that appending the text "<eos>" creates ID 258.
- **Compatibility trap for runtime testing:** initializer sets split_special_tokens=True by default. Whether appending literal "<eos>" text is encoded as regular UTF-8 bytes or EOS 258 must be tested against the pinned Transformers version. TRAIN-002b appended the literal EOS text and previously reported output budget exhaustion. This is a diagnostic, **not** a retroactive correction to TRAIN-002b.
- **Baseline match:** custom source hash agrees with TRAIN-002b, confirming identical code but not its semantic assumptions.

## Restricted next test

A separate CPU run with Transformers **4.57.6** should independently hash the four pinned tokenizer files and enforce the reviewed import/method surface **before executing the module**. Disable network after downloading the source. Compare ASCII, UTF-8, combining Unicode, padding, explicit BOS/EOS tokens, invalid bytes and literal EOS text; retain raw ID arrays and replacement/truncation measurements.

Passing this narrow tokenizer test must keep train_authorized=false. Actual void.1 LLM prompts, long-context causal/cache replay, checkpoint cold restart and training remain separate #79/#72 gates.

## Verdict

**Source-level review:** no suspicious executable/network path observed, with a scoped vocabulary-save file write. **Tokenizer runtime semantics:** not yet qualified. This review is not a general code-safety certificate or approval to train.
