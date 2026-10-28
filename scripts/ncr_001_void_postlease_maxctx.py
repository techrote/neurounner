#!/usr/bin/env python3
"""NCR-001 void.1: post-shared-lease 2048-unit cache and HF.generate qualification.

This is a new source-matched frozen-model execution, NOT a reclassification
of the pre-lease #128 evidence. Synthetic UTF-8 only, no optimizer, backward,
comparison training, held-out evaluation or #72/training admission.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import ncr_001_void_eos_postlease as original

LENGTHS = (512, 1024, 2048)
TAIL = 8
GENERATE_PREFIX = 512
GENERATE_UNITS = 4
ATOL = RTOL = 0.0005
SOURCE = (
    "🔧 café 日本語 πΩ — the workshop robot inspects a rail, records "
    "reversible evidence and leaves the frozen model unchanged. "
) * 32
EXPECTED_IMPLEMENTATION_BLOBS = {
    "scripts/ncr_001_void_eos_postlease.py": "45d170472f3b37c6e13088212deab60bf2b223d7",
    "scripts/ncr_001_void1.py": "7e6abcbf1b3738094e2b460089babed150c72a9b",
    "scripts/ncr_001f_frozen_hooks.py": "5efe8f0690bc1c2793de323d5b972b26bbd7bccb",
}


def require(condition, explanation):
    if not condition:
        raise RuntimeError("NCR-001 post-lease void.1 maxctx REFUSED: " + explanation)


def check_plan(protocol):
    require(LENGTHS == (512, 1024, 2048) and TAIL == 8,
            "predeclared maxctx diagnostic schedule changed")
    require(GENERATE_PREFIX == 512 and GENERATE_UNITS == 4,
            "generation diagnostic schedule changed")
    require(ATOL == RTOL == 0.0005
            and protocol["conformance"]["full_vs_cached_prefill_atol"] == ATOL
            and protocol["conformance"]["full_vs_cached_prefill_rtol"] == RTOL,
            "original cache numeric gate changed")
    require(protocol["train_authorized"] is False
            and set(protocol["models"]) == original.LANES,
            "four-variant training/source gate changed")
    require(protocol["models"]["void1"]["expected_config"]
            ["max_position_embeddings"] == 2048,
            "model max context changed")
    require(len(SOURCE.encode("utf-8")) > 2048
            and any(ord(c) > 127 for c in SOURCE),
            "ordinary synthetic UTF-8 source too short")
    return True


def check_source_blobs():
    for path, expected in EXPECTED_IMPLEMENTATION_BLOBS.items():
        observed = subprocess.check_output(
            ["git", "hash-object", str(original.ROOT / path)], text=True
        ).strip()
        require(observed == expected, "composed wrapper drift: " + path)


def compare_logits(torch, label, full, cached):
    require(tuple(full.shape) == tuple(cached.shape), label + " tensor shape")
    require(bool(torch.isfinite(full).all() and torch.isfinite(cached).all()),
            label + " nonfinite logits")
    delta = float((full - cached).abs().max())
    passed = bool(torch.allclose(full, cached, atol=ATOL, rtol=RTOL))
    original.REPORT["checks"][label] = {
        "pass": passed, "max_abs": delta, "atol": ATOL, "rtol": RTOL,
        "full_sha256": original.tensor_sha(full),
        "cached_sha256": original.tensor_sha(cached),
    }
    original.save("maxctx")
    require(passed, label + " actual frozen-host cache mismatch: " + str(delta))


def compare_generation_scores(torch, label, scores_a, scores_b):
    require(len(scores_a) == len(scores_b) == GENERATE_UNITS,
            label + " generation score count")
    details = []
    for step, (left, right) in enumerate(zip(scores_a, scores_b)):
        require(tuple(left.shape) == tuple(right.shape),
                label + " scores shape")
        finite = torch.isfinite(left)
        require(bool(torch.equal(finite, torch.isfinite(right)))
                and bool(torch.equal(left[~finite], right[~finite]))
                and bool(finite.any()), label + " nonfinite mask")
        maximum = float((left[finite] - right[finite]).abs().max())
        passed = bool(torch.allclose(left[finite], right[finite],
                                     atol=ATOL, rtol=RTOL))
        details.append({"step": step, "max_abs_finite": maximum,
                        "finite_values": int(finite.sum()),
                        "same_nonfinite_mask": True, "pass": passed})
        require(passed, label + " score mismatch on step " + str(step))
    original.REPORT["checks"][label] = {
        "pass": True, "steps": details, "atol": ATOL, "rtol": RTOL,
    }
    original.save("maxctx")


def run_actual_host(torch, base, tok, nca, mlp):
    byte_ids = tok.encode(SOURCE, add_special_tokens=True)
    require(len(byte_ids) >= 2048 and all(0 <= value < 256
                                          for value in byte_ids[:2048]),
            "reviewed ByteTokenizer added special IDs or source too short")
    sequence = torch.tensor([byte_ids[:2048]], dtype=torch.long)
    mask = torch.ones_like(sequence)
    original.REPORT["source_text"] = {
        "utf8_sha256": hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
        "utf8_bytes": len(SOURCE.encode("utf-8")),
        "characters": len(SOURCE),
        "available_tokenizer_units": len(byte_ids),
        "selected_units": 2048,
        "selected_ids_sha256": original.tensor_sha(sequence),
        "supervised_training_case": False,
        "ordinary_byte_ids_only": True,
    }
    original.save("maxctx")

    with torch.inference_mode():
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            # Both zero-readout variants must remain exactly the frozen base.
            for length in LENGTHS:
                ids, attention = sequence[:, :length], mask[:, :length]
                frozen = base(input_ids=ids, attention_mask=attention,
                              use_cache=False).logits
                disabled = wrapper(input_ids=ids, attention_mask=attention,
                                   enabled=False, use_cache=False).logits
                zero = wrapper(input_ids=ids, attention_mask=attention,
                               use_cache=False).logits
                original.compare_exact(frozen, disabled,
                                       kind + "_raw_disabled_" + str(length))
                original.compare_exact(frozen, zero,
                                       kind + "_raw_zero_" + str(length))
                original.save("maxctx")

            with torch.no_grad():
                if kind == "nca":
                    wrapper.nca.readout.weight[0, 0] = 0.03
                else:
                    wrapper.control.down.weight[0, 0] = 0.05

            # Full-context is compared with a real, prefilled KV cache and
            # eight one-unit decoder invocations, not a fake full re-forward.
            for length in LENGTHS:
                ids, attention = sequence[:, :length], mask[:, :length]
                full = wrapper(input_ids=ids, attention_mask=attention,
                               use_cache=False).logits[:, -TAIL:]
                frozen = wrapper(input_ids=ids, attention_mask=attention,
                                 enabled=False, use_cache=False).logits[:, -TAIL:]
                effect = float((full - frozen).abs().max())
                require(effect > 0 and bool(torch.isfinite(full).all()),
                        kind + " untrained readout inactive/nonfinite")
                original.REPORT["checks"][kind + "_untrained_" + str(length)] = {
                    "max_abs": effect, "trained": False,
                }
                prefill = wrapper(input_ids=ids[:, :-TAIL],
                                  attention_mask=attention[:, :-TAIL],
                                  use_cache=True)
                past = prefill.past_key_values
                incremental = []
                for index in range(length - TAIL, length):
                    one = wrapper(input_ids=ids[:, index:index + 1],
                                  attention_mask=attention[:, :index + 1],
                                  past_key_values=past, use_cache=True)
                    incremental.append(one.logits[:, -1:])
                    past = one.past_key_values
                    if kind == "nca":
                        traces = wrapper.last_hook_trace
                        require(len(traces) == 1
                                and traces[0]["tap_count"] == 3
                                and traces[0]["state_shape"][:2] == [1, 1]
                                and traces[0]["finite"] is True,
                                "NCA cache decoder reused field across tokens")
                compare_logits(torch, kind + "_maxctx_cache_" + str(length),
                               full, torch.cat(incremental, dim=1))

            # Suffix causality at the full native 2048 unit context.
            plain = wrapper(input_ids=sequence, attention_mask=mask,
                            use_cache=False).logits
            altered = sequence.clone()
            altered[:, -1] = (altered[:, -1] + 1) % 256
            mutated = wrapper(input_ids=altered, attention_mask=mask,
                              use_cache=False).logits
            original.compare_exact(plain[:, :-1], mutated[:, :-1],
                                   kind + "_2048_strict_future_suffix")

            # GenerationMixin, both cache modes, with actual EOS=258 and
            # explicit mask. These are untrained diagnostics, not EOS quality.
            runs = {}
            prefix = sequence[:, :GENERATE_PREFIX]
            kwargs = dict(
                input_ids=prefix, attention_mask=mask[:, :GENERATE_PREFIX],
                do_sample=False, max_new_tokens=GENERATE_UNITS,
                min_new_tokens=GENERATE_UNITS, eos_token_id=258,
                pad_token_id=256, return_dict_in_generate=True,
                output_scores=True,
            )
            for mode in ("disabled", "zero", "active_untrained"):
                with torch.no_grad():
                    if kind == "nca":
                        wrapper.nca.readout.weight.zero_()
                        if mode == "active_untrained":
                            wrapper.nca.readout.weight[0, 0] = 0.03
                    else:
                        wrapper.control.down.weight.zero_()
                        if mode == "active_untrained":
                            wrapper.control.down.weight[0, 0] = 0.05
                for use_cache in (False, True):
                    response = wrapper.generate(
                        **kwargs, use_cache=use_cache,
                        enabled=(mode != "disabled"),
                    )
                    generated = response.sequences[0].tolist()[
                        GENERATE_PREFIX:]
                    require(len(generated) == GENERATE_UNITS
                            and len(response.scores) == GENERATE_UNITS,
                            kind + " generation bounded output length changed")
                    runs[(mode, use_cache)] = (generated, list(response.scores))
                    original.REPORT.setdefault("generations", []).append({
                        "adapter": kind, "mode": mode,
                        "use_cache": use_cache, "ids": generated,
                        "actual_eos_id_emitted": 258 in generated,
                        "score_sha256": [
                            original.tensor_sha(s) for s in response.scores
                        ],
                        "learned_eos_claim": False, "trained": False,
                    })
                    original.save("maxctx")
            for cache in (False, True):
                off_ids, off_scores = runs[("disabled", cache)]
                zero_ids, zero_scores = runs[("zero", cache)]
                require(off_ids == zero_ids,
                        kind + " zero adapter changed frozen generation IDs")
                # Zero-init outputs must match exact same-cache reference.
                require(all(torch.equal(a, b)
                            for a, b in zip(off_scores, zero_scores)),
                        kind + " zero adapter changed frozen generation scores")
            for mode in ("disabled", "zero", "active_untrained"):
                uncached_ids, uncached_scores = runs[(mode, False)]
                cached_ids, cached_scores = runs[(mode, True)]
                require(uncached_ids == cached_ids,
                        kind + " generation token IDs changed with cache")
                compare_generation_scores(
                    torch, kind + "_generate_" + mode + "_cache_parity",
                    uncached_scores, cached_scores,
                )
            require(not base.model._forward_hooks
                    and not base.model._forward_pre_hooks,
                    "composed Qwen3 decoder left root hooks after generate")
    original.REPORT["status"] = (
        "PASS_VOID1_POSTLEASE_512_1024_2048_AND_GENERATE__72_BLOCKED"
    )


def main():
    original.REPORT.clear()
    original.REPORT.update({
        "_started": time.monotonic(),
        "schema": "ncr001-void-postlease-maxctx-v1",
        "phase": "maxctx", "variant": "void1", "status": "RUNNING",
        "training_authorized": False, "optimizer_steps": 0,
        "backward_executed": False, "comparative_training_performed": False,
        "heldout_opened": False, "full_72_conformance_pass": False,
        "independent_scientific_review": False,
        "historical_byte_eos_effect_measured": False,
        "checks": {}, "generations": [],
        "runner": {
            "pid": os.getpid(), "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "name": os.getenv("RUNNER_NAME"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "expected_head": os.getenv("NCR_EXPECTED_HEAD"),
            "event_merge_sha": os.getenv("GITHUB_SHA"),
        },
    })
    original.save("maxctx")
    try:
        protocol = json.loads((original.ROOT /
                               "experiments/ncr-001/protocol.json").read_text())
        check_plan(protocol)
        check_source_blobs()
        torch, _, base, tok, nca, mlp, _ = original.run()
        run_actual_host(torch, base, tok, nca, mlp)
        after = original.frozen_sha(base)
        original.REPORT["frozen_backbone_sha256_after"] = after
        require(after == original.REPORT["frozen_backbone_sha256_before"],
                "complete original Qwen3 backbone tensor bytes changed")
        require(all(p.grad is None for p in base.parameters()),
                "frozen original Qwen3 accumulated gradients")
        require(time.monotonic() - original.REPORT["_started"] <= 3600,
                "original void.1 3600s CPU lane watchdog breached")
    except BaseException:
        original.REPORT["status"] = "FAIL_VOID1_POSTLEASE_MAXCTX_QUALIFICATION"
        original.REPORT["traceback"] = traceback.format_exc()
        raise
    finally:
        original.save("maxctx")
    print(json.dumps({
        "status": original.REPORT["status"],
        "check_count": len(original.REPORT["checks"]),
        "training_authorized": False,
        "full_72_conformance_pass": False,
    }, sort_keys=True))


if __name__ == "__main__":
    if "--source-test" in sys.argv:
        p = json.loads((original.ROOT /
                        "experiments/ncr-001/protocol.json").read_text())
        check_plan(p)
        check_source_blobs()
        print(json.dumps({"lengths": LENGTHS, "tail": TAIL,
                          "train_authorized": False}, sort_keys=True))
    else:
        main()
