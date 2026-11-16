# VOID provisional DEV60 static-prefix probe v1 — source-only, NOT FROZEN

15 November 2026. This is a bounded, lane-owned contribution to the common DEV60 proposal, not a shared evaluation registry, an admitted DEVELOPMENT suite, new model training, or a model-quality result. Shared issue #568 and the independent reviewer retain allocation authority. Sole VOID PR #574 remains Draft/HOLD.

## Scope and immutable model identity

The probe uses the actual reviewed, pure-Python **CyberSandVoidByteTokenizer** and **encode_prompt** methods from the unchanged VOID adapter at the existing pinned revision dotlabs/void.1@9c2be5087772867d45961abd11f6b35110617563. The source-only workflow downloads no model weights or native corpus and does not import the upstream remote tokenizer Python, Torch or Transformers. It retains the exact 259-symbol strict UTF-8 byte tokenizer, non-chat cybersand-void-task-v1 framing, no implicit BOS, authentic EOS258 and 2048-token context.

It explicitly binds the historical shared system instruction text with SHA-256 2beaa9dbcd9a006c6bdeb4b0f96fd56fdb50eb0cfc4e8056b7f64bb30a21661c. Any future different instruction is a new reviewed framing/budget identity; this source-only version must not silently inherit it.

## Provisional contract and refusal

Input is a **provisional, target-free** UTF-8 JSON object with exactly one key, "rows". Each of its 60 objects has exactly: id, family, group_id, formulation_id, prompt, prompt_sha256. It checks the proposed 3 families x 4 structural-group IDs per family x 5 formulations (F0 through F4), unique request ID and group/form pairs, and actual raw UTF-8 SHA-256 for each literal prompt. It rejects BOM, invalid UTF-8, duplicate JSON keys, nonfinite constants, wrong types, surrogate substitution, missing formulations and any unexpected field, including teacher programs, target images or completion text.

It then computes the 60 real VOID encoded-prefix byte/token IDs and hashes, authentic system/framing identity, token counts and context headroom under one **provisional, fixed** integer new-token ceiling. It refuses any overrun rather than truncating. This tool is incapable of granting TRAIN/DEVELOPMENT admission: it does NOT read reference teacher programs and cannot establish target representability, same-meaning prompts, independent structure identities or visual/native results.

The report's mandatory marker is **PROVISIONAL_PREFIX_ONLY_NOT_DEV60_READY**, including explicit target_representability_checked=false, dev60_frozen=false, model_inference=false, new_native_operations=0, new_train_admissions=0, final_targets_opened=false. A source-only PASS is never DEV60 acceptance.

## Exact source-byte custody and provisional request identity

The source gate independently verifies the **actual complete on-disk bytes** of `adapter.py` and `profile.json` against their Git blob SHA-1 IDs at the immutable qualified foundation source `2a565df45efc568eb96a2166e0b77f299e289625`, **before importing/executing** the lane adapter. A different tokenizer implementation or profile, even if it retains the same declarative model/framing fields, fails closed and needs a newly versioned and hosted-qualified source identity. A symlinked source file is refused. This is local Git-byte identity, not a claim of a security boundary against arbitrary pre-loaded Python modules or compromised runner code.

The report additionally includes `adapter_git_blob_sha1`, `profile_git_blob_sha1` and a canonical `request_set_sha256` over all 60 literal request rows. Permuting row order preserves the manifest identity; changing a prompt and recomputing its individual prompt SHA changes the manifest identity. None of these digests signifies independently admitted structures, reference-target representability, or a frozen benchmark.

## Invocation and test

With a separately supplied target-free provisional request file:

    python3 experiments/agent_painter/lanes/void/dev60_prefix_probe_v1.py --target-free-requests /tmp/requests.json --provisional-ceiling 1024 --output /tmp/void-prefix-only.json

**1024 is illustrative, NOT a selected DEV60 ceiling.** A future fixed ceiling must be chosen before seeing model output, fit every common prompt, and separately fit every authorized DEVELOPMENT reference serialization, including genuine EOS. The historical six-request evaluator and original fresh120 optimizer/receipt evidence remain separate and unchanged.

Run focused model-free adversarial regression:

    python3 experiments/agent_painter/lanes/void/test_dev60_prefix_probe_v1.py -v

All 60 regression prompts are deliberately **synthetic fake input**, not real or admitted DEVELOPMENT. There is no FINAL access. When the shepherd and reviewer approve a *shared* manifest schema and its six new native-admitted DEVELOPMENT structures, adapt and version this provisional parser to the approved manifest rather than presuming this research-only row shape is canonical. Preserve model-specific pinned framing and perform a separate reference-target fit audit without ever exposing references to inference/training.

## Source-import custody correction (15 November 2026)

After verifying the exact committed Git blobs for the original adapter and
profile, the provisional checker now loads the adapter directly from that
checked path with `spec_from_file_location`. A pre-cached
`sys.modules["adapter"]` value can no longer substitute an unreviewed
encoder. A new adversarial test deliberately injects a cached impostor and
verifies that the pinned framing/model is still used. This is bounded source
custody, not protection against a compromised Python interpreter or runner;
it changes no qualified model, TRAIN inputs or native painting behavior.
