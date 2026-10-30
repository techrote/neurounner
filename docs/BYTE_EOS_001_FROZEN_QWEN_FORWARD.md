# BYTE-EOS-001 — original frozen void.1 causal-loss execution (no training)

Issue #92 (BYTE-EOS-001), epic #67 and conformance #72. This draft is stacked
on #164 and the unchanged future-only corrected EOS-ID serializer from #163.
No historical TRAIN-002b code, model outputs, held-out fixtures or protocol
have been changed.

## Source-locked intent

Does the *actual original pinned* dotlabs/void.1 Qwen3 model return finite
shifted causal loss using this future-only ByteTokenizer serialization, with
EOS ID 258 and strict prompt masking? This is not NCA-vs-MLP evaluation.

- Original dotlabs/void.1 revision 9c2be5087772867d45961abd11f6b35110617563,
  original model.safetensors SHA-256
  ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3
  and size 360622744; source-listed Qwen3 24 layers, hidden 640, vocab 259.
- Original tokenization_byte.py SHA-256
  279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03.
  AST-source review runs *before* dynamic import; the executed Python source
  is checked again after loading. PAD=256, BOS=257, actual EOS=258.
- Immutable four-lane protocol Git blob dddfb416b765ef6cf23e1cdd326f8834f9f1417c
  and source-pins blob 59fff1aed872b5563aa6898171983973e0cdb02c.
  The future-only serializer blob is 4c00c78b6776ade6901a99eee9a42e7ca61c698b.
- Three synthetic non-heldout prompts: single-byte context and empty completion,
  mixed multibyte UTF-8 response, and ordinary literal stop string text. Verify
  all real bytes are attended and only assistant bytes and integer EOS 258 are
  supervised; enforce correct shifted target count.
- In a real frozen Qwen3 forward, compare native outputs.loss against independently
  shifted logits cross-entropy with prompt labels masked (-100). Require finite
  values and <= 1e-5 absolute deviation. Recompute unchanged original model
  state fingerprint, require no gradients or optimizer/backward. Persist
  raw synthetic inputs/IDs/masks/labels, logits hashes, numerical losses,
  runner identity, complete source SHA and partial failures.
- Exact-head source and original model jobs run separately, with always-uploaded
  results. A green test suite is not a model result. Original Actions ZIPs and
  complete job logs require independent readback and Git custody before any
  original-host evidence is considered durable.

## Non-admission boundary

Even a passing actual frozen forward is NOT evidence of trained EOS generation,
historical TRAIN-002b effect size, post-warmup adapter gradients, post-update
backbone integrity, NCA/MLP hook/cache acceptance, independent scientific review,
or final #72 conformance. The untouched SmolLM2, Sprocket Chat and Sprocket
Instruct lanes remain independently gated. No comparative training, no adapter
optimizer or held-out evaluation is authorized; train_authorized=false.
