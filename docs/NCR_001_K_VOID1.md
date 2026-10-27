# NCR-001K — void.1 frozen Qwen3 depth-hook prepass

Issue [#79](https://github.com/techrote/neurounner/issues/79); parent [#67](https://github.com/techrote/neurounner/issues/67). Integration draft stacked on the shared NCA core PR #82.

## Identity and safe preflight

The target is dotlabs/void.1 at revision 9c2be5087772867d45961abd11f6b35110617563. The 24-block **Qwen3ForCausalLM** network has hidden width 640, 259 vocabulary units and NCA tap indices [7,15,23]. It is not a Llama model and does not reuse SmolLM2-specific configuration.

The source-bound full 360,622,744-byte model.safetensors SHA-256 was verified against pinned identity:

    ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3

The custom source file tokenization_byte.py was **downloaded as bytes only, never executed**. Its SHA-256 279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03 matches the already-qualified historical TRAIN-002b source. The byte vocabulary, tokenizer config and special-token-map files were also hashed against pinned metadata. This demonstrates source fidelity, **not a security review or correctness proof of the executable ByteTokenizer**.

The prepass intentionally uses short raw integer vocabulary IDs to exercise the decoder/hook structure without importing custom tokenizer code. There is no meaningful language or semantic tokenisation inference in this check.

## Hosted result

The corrected [Actions run 40424675239](https://github.com/techrote/neurounner/actions/runs/40424675239) **PASS** used a four-thread CPU with PyTorch 2.14.1+cpu and Transformers 4.57.6. Seven offline toy tests (including mismatched-Qwen3-config and source-hash refusals) passed; the real checkpoint was then read/verified and run.

| Check | Result |
|---|---:|
| Zero-initialized logits vs frozen Qwen3 | Exact, max diff 0 |
| Two-sequence batch initial parity | Max diff 0 |
| Greedy three-unit generation initial parity | Identical |
| Nontrained manually perturbed NCA effect on logits | 0.0220349 max |
| Full vs incremental KV-cache error with nonzero readout | **0.0000104904** max, within 0.0005 tolerance |
| Future-suffix causal invariance | 0 max |
| Readout gradient L1 | 4,867.964 |
| Peak process RSS | 749,736 KiB |
| Optimizer steps | 0 |
| Original Qwen3 parameters frozen | 90,148,352 |
| Trainable NCA parameters | 329,944 |

The [raw 4,319-byte source receipt](../experiments/ncr-001/void1-preflight/receipt.json) has SHA-256 **1aaa45cb5942c6d033c8fc69da13d5bd76232a240da1fd4e76f1b31f132399c3**, with independently [recorded source-run and artifact provenance](../experiments/ncr-001/void1-preflight/provenance.json). The preflight is **not** final #72 source/cache qualification.

## Outstanding void.1 admission gates

1. Review the pinned custom ByteTokenizer Python implementation as code before enabling trust_remote_code. The source SHA match to TRAIN-002b is helpful but not a substitute for review.
2. Prove exact byte-vocabulary encoding/decoding, UTF-8 and Unicode roundtrip, BOS/EOS/padding/continuation, byte unit accounting and output-budget exhaustion without changing the fixed holdout.
3. Independently repeat full-vs-KV-cache and request-isolation tests on actual tokenized text, including longer contexts. No source or causal check for another model may waive void's own #72 gate.
4. Only after preregistration review and within-void matched MLP controls could a bounded training experiment be admitted.

No adapter learning, external tokenizer execution or claim of improved perplexity has occurred.
