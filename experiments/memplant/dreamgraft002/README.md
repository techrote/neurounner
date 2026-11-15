# DREAMGRAFT-002 — Two-Rule Transplant (exploratory)

**Question:** Can real Qwen inference state carry **two independently variable fictional-world laws** into a frozen unrelated receiver, and can a single-law counterfactual change the corresponding recipient decision without disturbing the other law?

This is a bounded, real-model research experiment continuing [G0 #230](https://github.com/techrote/neurounner/pull/230), [G1/G1B #231](https://github.com/techrote/neurounner/pull/231) and [G1C #232](https://github.com/techrote/neurounner/pull/232). No existing Draft PR has been merged. This experiment has no automatic approval for live cache editing, generations of artificial descendants or Chimaera synthesis.

## Construction

- Frozen Qwen2.5-0.5B-Instruct donor, revision `7ae557604adf67be50417f59c2c2f167def9a775`: actual layerwise KV reads, matched 2,048-element **pooled V**, contextual hidden, and uncontextualised token-embedding arrays.
- Synthetic physical laws: warming fictional metal **hardens / softens** it; vibration makes its dust **float / sink**. All four combinations appear per world family, each with rejected wrong statements and verified correct statements to reduce lexical shortcuts.
- 32 scenario families; **20 train (80 sources), six validation (24), six sealed (24)**. All four counterfactuals of a world stay in one split. Validation and sealed worlds use disjoint narration patterns and material names.
- Per feature method, a frozen two-output supervised ridge (`lambda=1`) predicts one signed score per law. Neither labels nor original source text go to the recipient for numeric conditions.
- Four continuous eight-position native prefixes are directly optimised against frozen SmolLM2-1.7B-Instruct for each joint law combination. Both models' actual weights stay frozen. Two sigmoid scores then form a bilinear mixture of the four soft prefixes.
- Recipient scoring on held-out worlds records **four-way joint accuracy**, independent heat and vibration accuracy, and source-specific prediction changes for **heat-only**, **vibration-only** and both-law donor swaps. V-state additionally checks no-prefix, eight-zero-prefix, random-prefix, hard-gated and full-source-text reference.
- A small **unedited raw dream gallery** probes machines and organisms for all four law combinations and selected swaps. It is not counted as scientific accuracy evidence.

## Research interpretation

This is intentionally **a supervised two-dimensional numeric communication channel** (two learned scalar outputs controlling four model-native prefixes), not direct cross-family KV-cache migration or a model-neutral memory format. A real advantage of pooled V over equal-capacity hidden or lexical baselines would be task-specific; pooled V cannot contain information absent from its complete underlying hidden state. Ordinary full-text inference remains a serious comparator.

A successful source-specific heat-only swap that flips the heat answer **and preserves the vibration answer** is stronger evidence than changing an output randomly. Correctly reporting two laws is still not proof of a coherent internal physical model; generative usefulness must be tested separately.

**Statistical limit:** only six independent sealed world families. Treat all conclusions as exploratory. Do not count 24 variant cases as 24 independent environments or tune the sealed test after inspecting results.

## Reproduction

[Workflow](../../../.github/workflows/memplant-dg002.yml) runs an isolated donor job, frozen receiver-native prefix fitting, then separate V/hidden/embedding receiver evaluations. Results and negative cases are retained in GitHub Actions artifacts for 30 days. Source texts, manifests, hash records, trained prefix tensors, raw prediction scores, and generated text are archived; original model weights are not committed.

No user conversations, private sources, automatically generated image tasks, or production inference service modifications are involved. **Draft/HOLD. No merge.**
