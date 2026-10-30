# NCR-001E — void.1 post-diagnostic cold-process adapter replay (30 October 2026)

**DRAFT. Source-bound diagnostic only, no comparative training, no scientific admission.**

Parent issues #67 / #72; related controls #73 and BYTE-EOS-001 #92. This draft is stacked directly onto #175 at 515b8cdaa64164b9282051997c76e09eb4f501ee. The original #171 nine-check decision and the later #172 physical diagnostic remain separate. The earlier original frozen-Qwen3 cold replay was pretraining/manual readout; it did NOT test saving and reloading an adapter after the single public-sample SGD diagnostic.

## Narrow question

Can one NCA adapter after exactly one synthetic public teacher-forced SGD step be serialised temporarily and reloaded in a **separate Python process** on the same four-CPU runner, preserving frozen-source identity, bit-exact active logits, A–B–A reset and full-versus-final-four true incremental KV logits at the unchanged 5e-4 absolute/relative tolerance? This is a 31-unit post-step follow-up; it is NOT a fresh 2,048-position post-step conformance test, general learned-quality assessment, independent host replication, generated EOS competence, MLP comparison or final #72 scientific PASS.

## Frozen model, code and fixtures

Source: dotlabs/void.1 revision 9c2be5087772867d45961abd11f6b35110617563, real Qwen3 checkpoint model.safetensors SHA-256 ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3 (360,622,744 bytes), reviewed ByteTokenizer source SHA-256 279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03. Native EOS is the integer 258, never the printable five-byte literal. The original corrected serializer is pinned at commit 9fbdee880697903aaf5b91cc29442c9e53479e50.

This branch fixes and hashes the exact original #172 diagnostic module (Git blob 672fede7cb488bb33315b067bba5292678435644), frozen loader (45d170472f3b37c6e13088212deab60bf2b223d7), void model wrapper (7e6abcbf1b3738094e2b460089babed150c72a9b) and shared cellular core (0427ad0b0f1fc68199369d5094952017bc70f503). Protocol/source-pin Git blobs remain 14ce5fbaa38f8806c8c8738f19c01b91385eebaf / 59fff1aed872b5563aa6898171983973e0cdb02c, with four distinct frozen-model variants and train_authorized=false.

The only example is the previously public synthetic source from #172: prompt “Q: π?\nA: ” (10 UTF-8 bytes) and completion “🔧 café 日本語” (20 UTF-8 bytes), 31 tokenizer units including terminal real EOS 258 and exactly 21 supervised completion/EOS units after causal shifting. No sealed evaluation fixture is inspected or changed.

## Implementation and admission refusal

The offline standard-library sentinel suite rejects malformed counts (including boolean-as-integer), swapped checkpoint identities, false training/review, wrong token/EOS or shifted-label counts, altered source heads, nonfinite/zero active readout effects and incorrect thread provenance. It executes no model and no optimizer.

The source-bound physical workflow fetches the six original pinned Qwen3/tokenizer files by exact revision and verifies their SHA-256s, then goes offline. It uses the original CPU package versions (Torch 2.14.1+cpu, Transformers 4.57.6, huggingface_hub 0.36.2, safetensors 0.8.0). The producer calls the unmodified original #172 one-step NCA-only SGD function: a single readout update at learning rate 0.005 and a later second backward probe only. The source and frozen logits are checked; no MLP optimizer, corpus training or other sample is executed. A temporary post-step adapter state is created only to conduct the cold-replay comparison.

The consumer is an independently started Python interpreter which reloads the same frozen Qwen3 from pinned weights and loads that temporary adapter. It checks source/backbone/adapter hashes, exact active-logit identity, A–B–A request reset, and four genuine one-token cached positions versus the full 31-unit sequence. Effective Torch threads are recorded after loader configuration, not guessed from startup metadata. Both success and failure receipts are retained.

The temporary updated adapter binary and metadata are deleted unconditionally **before** any GitHub Actions artifact upload. Only raw source/host logs, receipts and file digests are retained, never a distributed trained checkpoint. Original ZIPs plus decoded job logs (including failures) must later be permanently archived in Git and independently audited; transient Actions artifacts alone do not satisfy custody. A successful source audit is not a new physical-host test.

## Historical negatives and locked gates

The older 12 void.1 four-step generations without real EOS 258 remain negative. Historical TRAIN-002b effect of incorrect literal EOS text is UNMEASURED. Sprocket root Instruct's incomplete six-token generated tool-call prefix remains unqualified; neither model gets inferred useful tool/memory skills. No model-specific conformance is promoted; no comparative NCA/MLP training, preregistered held-out access, modified decision rules or draft merges.

Status: train_authorized=false; void1_conformance_pass=false; full_72_conformance_pass=false; actual_compute_budget_qualified=false. The owner's same-account COMMENTED scoped-review exception is administrative only, not GitHub APPROVED or scientific acceptance.

The exact-head physical CI, archive custody and scientific interpretation are PENDING until actually observed; do not infer PASS from the existence of this workflow or draft.
