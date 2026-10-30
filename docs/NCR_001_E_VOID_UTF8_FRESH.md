# NCR-001E — fresh reviewed void.1 UTF-8, cache and cold-adapter prepass

**Parent:** [#72](https://github.com/techrote/neurounner/issues/72). **Draft PR:** [#93](https://github.com/techrote/neurounner/pull/93), stacked on reviewed ByteTokenizer source/runtime [#90](https://github.com/techrote/neurounner/pull/90)/[#91](https://github.com/techrote/neurounner/pull/91), and original frozen void.1 integration [#85](https://github.com/techrote/neurounner/pull/85).

## Exact source and separation

This is a **new reviewer-owned actual Qwen3 model invocation**, not a readback of the earlier raw-ID-only void.1 prepass. It checks out original source PR #85 at exact commit 5fa3dc8c0de153098d1800f7b0c809819620aef4, verifies the complete downloaded 360,622,744-byte source checkpoint SHA-256, and uses independently reviewed pinned ByteTokenizer code under an offline process.

Model revision: dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563. Checkpoint SHA-256 **ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3**, tokenizer Python SHA-256 **279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03**.

The review deliberately uses real Unicode text (🔧, café, 日本語, πΩ), with **1,144 UTF-8 bytes/tokenizer units** but 1,032 Python Unicode characters in the source prompt. It does not decode incomplete multibyte tokens individually or assume a byte is equivalent to a character. Explicit EOS ID 258 was passed to the generation API.

## Real hosted result — 27 October 2026

[Actions 40464956343](https://github.com/techrote/neurounner/actions/runs/40464956343) **PASS** on pinned CPU-only torch 2.14.1 and Transformers 4.57.6 (four threads).

| Frozen void.1 source-bound check | Measured |
|---|---:|
| Zero-init original logits vs NCA, 16/64/128 UTF-8 units | Exact, max diff 0 |
| Two-request left/right padded batches | Exact, max diff 0 |
| Greedy four-token generation at zero-init | Identical generated token IDs |
| Actual EOS observed in that four-token generation | No; **not a general EOS metric** |
| Manual nontrained NCA readout, maximum logit effect | 0.0622978 |
| True one-unit KV-cached vs full 48-unit forward | **2.86102 × 10⁻⁵** max abs, below frozen tolerance 5 × 10⁻⁴ |
| Future-suffix earlier-logit influence | 0 |
| A→B→A request isolation | Exact |
| New adapter, same frozen base, cold-load logits | Exact |
| Corrupt source revision in adapter metadata | Rejected |
| Frozen base parameter count | 90,148,352 |
| NCA trainable count | 329,944 |
| Peak process RSS | 723,736 KiB |
| Actual optimizer steps | 0 |

[Original 5,288-byte raw receipt](../experiments/ncr-001/evidence-review/void1-utf8-fresh/receipt.json) SHA-256 **a595e5bc61762f35085fbaeb24c1a5a8d4e654ef3a344a1c56232046a51e7e51** and [source-run provenance](../experiments/ncr-001/evidence-review/void1-utf8-fresh/provenance.json) preserve exact model/toolchain/pin identity, checks and explicit limitations. The original transient artifact #12716546149 ZIP digest **6714cbdababf999e1a2b9419609750dc41311d6c924a42ffb0b256247f7d5760** was separately read back, rather than relying only on the green Actions badge. A follow-up CI step independently verifies committed raw evidence against the original SHA-256.

## Important stopping caveat

The ByteTokenizer runtime [review #91](https://github.com/techrote/neurounner/pull/91) reproduced that literal text "<eos>" is **five regular byte IDs**, not terminal token ID 258. The historical TRAIN-002b target-appending pattern therefore did not supervise real EOS. The candidate correction is tracked separately in [BYTE-EOS-001 #92](https://github.com/techrote/neurounner/issues/92), without rewriting historical evidence.

This fresh model prepass **does not** show an actual improvement in EOS emissions or model quality. No adapter parameters were trained: the single nonzero readout value was manually set. Cold reload used a new adapter against the same frozen base, not a new operating-system process. The remaining Sprocket Chat/Instruct formatter and fresh runtime gates are unaffected.

**Verdict:** Fresh source-bound void.1 UTF-8 model/cache/adapter prepass **PASS**, but independent full four-model #72 admission and all NCR-001 comparative training remain **BLOCKED**.
