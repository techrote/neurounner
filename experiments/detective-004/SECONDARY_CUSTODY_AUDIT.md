# DETECTIVE-004A — secondary Stage-A custody check (author-side)

This check supplements [Stage A](STAGE_A.md) and [draft PR #149](https://github.com/techrote/neurounner/pull/149). **It is not independent review and does not grant Stage B, learner-final materialization, production admission or cast expansion.**

## Specific gap addressed

The original producer validator validates the separate `row["view"]` structure and the rendered turn sequence, but does not itself prove that the view used for permission/argument checks exactly matches the JSON embedded in the **actual model-visible user message**. A modified row can therefore contain a broader audit-side view than the model was shown. The committed generator still deterministically produces matched rows; this is an adversarial verifier gap, not evidence that the original 36-row archive was corrupted.

The additional `scripts/detective_004a_custody_audit.py` mechanically cross-checks each of the 36 committed source-bound training rows against the exact user-message projection, the pinned B3 tool manifest, family-bound case IDs, assistant content + closing end-token **character** supervision spans, visible assets/credentials, and host-only receipt separation. It rejects out-of-band view changes, case-family aliasing and forged results.

Run with:

```sh
python3 scripts/detective_004a_custody_audit.py --archive datasets/detective-004a
python3 -m unittest tests.test_detective_004a_custody -v
```

The additional verifier only consumes frozen Stage-A data. It does **not** change the corpus, `training.jsonl` SHA-256 `a8ea42c3cd2344fcc285e35c98dfcf2745282fa5b7738e02722c820c0983e513`, the manifest, generator/config Git blobs, or the sealed 12-task posthash final recipe. No final is generated, and no tokenizer/weights are loaded.

## Unresolved admission gates

A distinct independent reviewer must still qualify the source-native role/assistant loss design, case-family separations, source identities, final custody and the full protocol before Stage A merges. The frozen tokenizer's real token-label arrays and 512-token limit remain **unverified** in Stage A and are mandatory prior to any Stage B optimizer step. Archive-side character spans are not token-level labels.

The current `adapter_dir_sha256` helper binds file-path/size/hash identity; merely naming two files does not prove a genuine trained adapter or correct base ancestry. Any future Stage-B final custodian must independently authenticate the actual adapter, optimizer receipt, root-weight identity and the posthash final before allowing the evaluator to be interpreted as evidence. A naked `admission_gate` result on arbitrary task objects is not qualification.

DETECTIVE-001R1 #121 remains separately reviewer-blocked: no opaque relevance ratings, no population expansion.
