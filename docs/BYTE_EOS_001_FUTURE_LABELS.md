# BYTE-EOS-001 — correct future void.1 supervised stop-token labels

Issue [#92](https://github.com/techrote/neurounner/issues/92). Draft implementation [PR #94](https://github.com/techrote/neurounner/pull/94) stacks on reviewed pinned ByteTokenizer runtime [PR #91](https://github.com/techrote/neurounner/pull/91).

## Verified historical mismatch

The fixed dotlabs/void.1 ByteTokenizer source SHA-256 is 279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03. It uses individual UTF-8 byte IDs 0–255 and EOS=258, but does not insert BOS/EOS automatically and splits literal special-token strings when ordinary text is encoded.

The earlier TRAIN-002b source encoded its completion by concatenating literal tokenizer.eos_token text and then tokenizing the entire sequence. At the exact pinned model and Transformers 4.57.6, this produces five IDs **[60,101,111,115,62]**, not EOS 258. Thus the original supervised target lacked the dedicated EOS stop token. The previous generation-budget exhaustion could partly reflect this issue, but effect size is **unmeasured**.

Historic TRAIN-002b code/results/weights/evidence remain unchanged. Any retrospective baseline needs a new source-bound experiment and cannot silently overwrite previous work.

## New future-only serializer

[scripts/byte_eos_001_void_labels.py](../scripts/byte_eos_001_void_labels.py) implements a strict, standard-library, model-agnostic label constructor for reviewed void ByteTokenizer only.

- Verify original ByteTokenizer class, vocabulary 259, PAD 256, BOS 257, EOS 258, and **no automatic BOS/EOS insertion**.
- Encode prompt and full prompt+assistant completion as exact valid UTF-8 bytes with no normalization or inserted control IDs.
- Append terminal **integer 258** after tokenization.
- Mask prompt tokens as -100; supervise each completion byte **and EOS 258**.
- Cap total tokenizer units at **1,024** for NCR, refusing examples that would truncate EOS. Do not silently cut response bytes or alter user Unicode.
- Report sequence units, supervised units, prompt units, UTF-8 source bytes and Unicode characters separately. No inference, model weights or optimizer required.
- Provide a strictly read-only legacy comparison that reports the old five-byte suffix, corrected single-ID suffix and **four-unit reduction**, without modifying historical training data.

## Hosted qualification

[Actions 40466048138](https://github.com/techrote/neurounner/actions/runs/40466048138) **PASS** after an initial workflow failure caused by an older stacked protocol snapshot's missing 1,024-unit field. Corrected CI explicitly checks the exact preregistration [PR #77](https://github.com/techrote/neurounner/pull/77) commit a1b41c358d215974ea9ef2748baa13c3872b0fbb. It does not invent a fallback token limit.

**Nine pure-Python negative/positive tests passed**, covering EOS supervision, prompt masking, Unicode, literal stop strings, empty completion, no-truncation boundary, malformed Unicode, wrong tokenizers and historical difference. A separate offline real pinned ByteTokenizer source-check then passed at Transformers 4.57.6, with no model weights or training.

The original [758-byte receipt](../experiments/ncr-001/byte-eos-001/receipt.json) SHA-256 **6c611bdc0f021f6d9e2fa39d91b7e141d937844f2b849f9ae437151254572922** and [artifact/run provenance](../experiments/ncr-001/byte-eos-001/provenance.json) are durable. Original artifact ID 12716463457 has ZIP SHA-256 7c9919714e4c8cc6cbb563e1138a9a6c7915cfc3846840642eda99f1ee223539.

For the fixed test example, the corrected sequence has 84 byte-tokenizer units: 37 prompt units and 47 supervised units including actual EOS ID 258. Historic text-appended-EOS targets would consume four extra units and supervise the wrong terminal representation.

## Limits and dependency

This is a **future serializer candidate, not a switched-on training path**. The code does not change original train/eval logic, create checkpoints or enable any optimizer. It remains to be integrated into void-specific NCR-001 training only after frozen protocol review and independent #72 real-host/text/cache/cold-source approval.

**No measured model-quality or EOS generation-rate improvement, no MLP/NCA comparative training, and train_authorized=false throughout.**
