# NCR-001K — exact reviewed void.1 ByteTokenizer runtime semantics

**Source review:** [PR #90](https://github.com/techrote/neurounner/pull/90). **Restricted runtime:** [PR #91](https://github.com/techrote/neurounner/pull/91). **Related reproduced EOS defect:** [#92](https://github.com/techrote/neurounner/issues/92).

## Results (27 October 2026)

[Runtime CI 40463695879](https://github.com/techrote/neurounner/actions/runs/40463695879) passed after pinned-source AST verification, followed by a stricter regression [40463929003](https://github.com/techrote/neurounner/actions/runs/40463929003), which again passed **37/37 checks**.

The tokenizer was loaded using Transformers 4.57.6 from exact dotlabs/void.1 revision 9c2be5087772867d45961abd11f6b35110617563 with fully offline HF/Transformers runtime. The actual executed Python code was rehashed to the reviewed 4,458-byte source SHA-256 **279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03**. Model checkpoint weights were not loaded and there were **zero optimizer steps**.

| Probe | Verified behaviour |
|---|---|
| Vocab IDs | Bytes 0–255, PAD 256, BOS 257, EOS 258 |
| Ordinary text encoding | One tokenizer ID per UTF-8 byte; BOS/EOS not auto-inserted |
| Valid ASCII, accents, combining marks, Greek, Japanese, emoji | Exact UTF-8 byte/string round-trip |
| Padding | PAD 256, left by default; left and right attention masks correct |
| Invalid standalone UTF-8 byte | Replacement character; not byte-lossless |
| Stream decode one byte at a time for multi-byte emoji | Replacement characters instead of original emoji |
| Explicit EOS appended as integer 258 | Real EOS token |
| Textual string <eos> passed through ordinary tokenizer | **[60, 101, 111, 115, 62]**, five UTF-8 byte IDs, **no EOS 258** |
| Literal EOS text appended to prior text | Does **not** introduce ID 258 |

## Important historical consequence

The committed historical TRAIN-002b encoder forms a target with text concatenation of the tokenizer's EOS string **before** tokenization. Under the source-bound runtime, the tail is **five regular byte target IDs**, not EOS ID 258.

This is a reproduced serialization mismatch, tracked in [BYTE-EOS-001 #92](https://github.com/techrote/neurounner/issues/92). It is a **plausible explanation** for some history of generation exhausting its byte-unit budget, but causality/effect size have **not** been measured; model repetition and poor learned stopping may have additional causes.

Future void.1 corpus preparation must explicitly append token ID 258 **after encoding**, supervise that ID, and prove it is never truncated. Historical TRAIN-002b weights, receipts and scores remain unchanged.

## Evidence custody

Full original static source review report: **6,903 bytes**, SHA-256 **14ac28204837d7615220981c5b3ec3c86b9effe548306047b012fed69e0efa1c**, GitHub Actions run [40462933022](https://github.com/techrote/neurounner/actions/runs/40462933022), artifact #12714879401 (ZIP SHA **aa7ff9d9e8d8ab21ca946e17ec51fe37527a73499c2e18d0c4231dc646c5fcf5**).

Runtime regression original: **9,663 bytes**, SHA-256 **dd12dc7321bb0115bef01b16f5afa67b5d2d325154162bfef469ddfd1c1e4bb1**, Actions [40463929003](https://github.com/techrote/neurounner/actions/runs/40463929003), artifact #12715522343 (ZIP SHA **5cfd97b0bc2ba6232f183fd37516031c7b697f645e0cf372deae5dec8cf15be6**).

[Durable evidence summary](../experiments/ncr-001/void-byte-runtime/evidence.json) records exact source pins, original artifact and raw report hashes, all 37 checks and the observed EOS discrepancy. The code regenerates the original machine-readable reports; Actions archives remain temporary.

## Limits and next test

This qualifies **only pinned tokenizer runtime byte semantics** and the bounded source review. The implementation does not qualify the real void.1 model's text-prefill/KV-cache/cold-reload behaviour, EOS generation outcomes, model quality, or security under arbitrary future tokenizer changes. The next #79/#72 model-side reviewer check must use this exact tokenizer to construct actual text prompts and compare longer masked/cached forwards.

**No NCR-001 training is authorized.** The protocol retains train_authorized=false and every original model-specific qualification remains independently gated.
