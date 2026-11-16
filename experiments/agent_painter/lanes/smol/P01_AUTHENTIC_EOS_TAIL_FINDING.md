# SMOL P01: authenticated upstream chat-template EOS-tail mismatch — 14 November 2026

**Status: FAIL / common training-label contract needs shepherd-owned correction.** This is a model-specific source-and-runtime finding, not a new Art/optimizer result and not positive TRAIN admission.

## Original executed evidence

- Sole Draft/HOLD PR #573, branch `codex/agent-painter-smol-568`. First label proposal commit `24ae20fbd1c6328e1d3c7ed112250c09f1e1e4d9`; diagnostic follow-up `b4b876dcbc4357bb6293019b64001d7cd1848dff`. Shared base remains #569 at `2a565df45efc568eb96a2166e0b77f299e289625`.
- Actual hosted CPU runs [#43633850191](https://github.com/techrote/cybersand/actions/runs/43633850191), job `132566214776`, **FAILED**; and [#43633994935](https://github.com/techrote/cybersand/actions/runs/43633994935), job `132566661413`, **FAILED as designed at strict terminal-EOS gate**. The former retained [partial artifact #14128888895](https://github.com/techrote/cybersand/actions/runs/43633850191/artifacts/14128888895), digest from upload logs `7066cde759b745602e625af4d1a5ff0af0cb88172cfe6fb284a838f42f9050e2`. Both completed **12/12 adversarial source-only tests**, pinned environment installation and actual public tokenizer/config small-file downloads; neither loaded model weights or trained.
- Model revision `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`. The hosted checker verified exact expected pinned `config.json`, `tokenizer.json`, `tokenizer_config.json` SHA-256 bytes from the existing shared model profile, used `AutoTokenizer.from_pretrained(..., trust_remote_code=False)` and the *unchanged* shared `training/common.py:encode_row`. This is not a forged tokenizer or an alternate chat framing.

## Exact witnessed failure

The authentic tokenizer/Jinja output for P01 gives **194 input tokens = 71 prefix-masked tokens + 123 supervised tokens**, and the true model EOS ID is **2**. The only EOS in the supervised suffix occurs at **zero-based token index 192**. However, the template appends a separate supervised **newline ID 198 at index 193, after EOS**. Tail decoded text is:

```text
,"variation":0,"version":2}<|im_end|>\n
```

The model-free loss-mask checks pass exactly: positions `0..70` are `-100`; the whole supervised suffix equals the input suffix. The failure is narrower: **`input_ids[-1] == eos_token_id` is false**, even though `eos_token_id` appears at the penultimate position. The existing shared encoder only checks that EOS *appears somewhere* after the prefix boundary; it does **not** require EOS to be the final supervised token. The new SMOL label witness intentionally fails rather than silently normalizing, repairing or misreporting this.

This discovery does **not** invalidate the fact of previous 120 real optimizer steps, original native Art permissions/readbacks or original source identity. It does qualify prior descriptions of "true terminal EOS": the genuine upstream Jinja path supervises one formatting newline **after** EOS. Historical checkpoint/recipe contracts must remain immutable; old candidates cannot be relabeled as newly terminal-EOS-qualified.

## Required coordinated correction, not a lane takeover

1. Shepherd/reviewer should verify this run and claim the exclusive shared #569 lease before making any source change. Add an explicit upstream-SMOL completion-label policy to shared `encode_row`/its companion regression tests: preserve the authentic upstream system/user/assistant framing and full JSON bytes, but make the **terminal model EOS the final supervised token**; separately document that the template's trailing presentation newline is not part of the model's completion target. Fail if the pinned template's suffix changes beyond the precisely observed single LF or if EOS appears elsewhere in the supervised answer.
2. Requalify a **new recipe/source identity** through the exact pinned 12 TRAIN rows (never FINAL), sample-cursor/token-boundary checks, actual hosted two-update/fresh-VM reproducibility and any later positive common native TRAIN admission. Do not resume old optimizer checkpoints under a new supervision contract; new-stage copied-weight training is separately labeled and uses fresh optimizer state.
3. Repeat P01 pinned tokenizer review at a newly source-bound head, preserving the original failed logs. Global P01 split approval, positive native TRAIN admission and frozen common 60-request DEVELOPMENT suite remain independent prerequisites before full painting training.

The P01 authored graph still has one upward lower-trunk riser, not the held-out three-riser DEVELOPMENT topology. No new native drawings, TRAIN rows, useful model freeze or FINAL inspection occurred here. **Draft/HOLD; no merge.**
