# NCR-001E — Sprocket Instruct shifted-loss / END-padding alignment (NO TRAIN)

**Scope:** diagnostic source-only addendum stacked above [PR #182](https://github.com/techrote/neurounner/pull/182), which archives [PR #181](https://github.com/techrote/neurounner/pull/181)'s original full-position mask replay. The producer remains [PR #161](https://github.com/techrote/neurounner/pull/161) and its original archive [PR #162](https://github.com/techrote/neurounner/pull/162). The new work does not change original inputs, fixtures, source, masks, frozen weights or the preregistered evaluation suite.

## Newly isolated failure mode

A valid full-position unshifted assistant supervision mask and an individually correct attention mask do not, alone, certify that an eventual next-token cross-entropy consumer shifts targets correctly. Synthetic left padding is additionally ambiguous because the actual Sprocket turn terminator <|end|> has token ID **5**, also used as the transport padding ID. The causal loss must use target at index j and logits at j-1; both positions must be attended.

This draft independently composes the unaltered exact-source 12-turn root Sprocket Instruct fixture into a two-row batch: a 3-turn host-only prefix left-padded to the length of the original 12-turn sequence, plus the full original sequence. Assistant body and actual assistant END labels are retained; system/user/tool/host-memory and dangling assistant-cue tokens remain unsupervised. Any pad-row supervision or naive token-id-not-5 attention masking is rejected. Negative tests also reject target off-by-one, loss on the dangling cue, altered turn delimiters, malformed label types and collapsed source identities.

The original label and token vectors are independently reconstructed from the exact pinned **root Instruct tokenizer**; original model-host receipt hashes for all 123 token IDs, 45 supervised assistant units and explicit padded attention are checked against the archived original #161 receipt via #162/#182 custody. The new receipt hashes the newly composed shifted next-token target tensor and preserves original CI/runner identities. Chat and Instruct checkpoints remain distinct: Chat chat/model.safetensors, Instruct root model.safetensors.

## Evidence and execution classification

- Original actual frozen-model publisher-wire [run 40718706396](https://github.com/techrote/neurounner/actions/runs/40718706396), PR #161 source head aadbfe823c9778066acac0a0869e0c559a533f05; original [failed source run 40718490313](https://github.com/techrote/neurounner/actions/runs/40718490313) retained.
- Original full-position source/tokenizer replay: [run 40782101613](https://github.com/techrote/neurounner/actions/runs/40782101613), commit 40efcb5748894c6c7847c18b1ad442a455bd0aa3; original two-ZIP Git custody is PR #182 commit 8ca28b8aa8c43f97795d1149a7563bee9444763b, [custody CI 40783012701](https://github.com/techrote/neurounner/actions/runs/40783012701).
- New workflow performs a fresh **source-only pinned tokenizer** execution and validates the newly constructed causal target alignment. It downloads SHA-verified small tokenizer/config/card files only and uses the fixed 12-turn synthetic diagnostic from the original physical source. It does not download or execute model weights. The test fake tokenizer is deliberately not asserted to be numerically equivalent to the pinned root tokenizer.
- Every failure, including dependencies, causal/attention mismatch and source drift, is to be preserved in the CI log/receipt. Actions artifacts are temporary staging and require further byte-exact Git filing before permanent custody is claimed.

## Strict non-admission boundary

The original frozen model's incomplete six-token generated tool-call prefix remains a **negative result**, not tool competency. These new causal targets are purely structural; no measured cross-entropy, gradients, learned stopping, adapter update, cached frozen-model invocation, behavioural success or NCA-versus-MLP comparison is claimed. Original #72 acceptance remains blocked pending scientific review and four independent model-specific gates; #73 and BYTE-EOS #92 remain separate. Historical void.1 EOS-byte supervision's effect remains **unmeasured**.

train_authorized=false, full_72_conformance_pass=false. No held-out opening, model training, comparative study, protocol/fixture alteration or draft merges are permitted.
