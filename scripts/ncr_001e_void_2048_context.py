#!/usr/bin/env python3
"""NCR-001E independent void.1 512/1024/2048 + HF.generate conformance.

Actual pinned frozen dotlabs/void.1 Qwen3 and independently reviewed
ByteTokenizer. Synthetic mixed UTF-8 source only. No optimizer, backward,
heldout evaluation, BYTE-EOS historical reinterpretation, or training.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "implementation"
OUT = ROOT / "runtime/ncr-001/void1-2048-context.json"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"

REPO = "dotlabs/void.1"
REV = "9c2be5087772867d45961abd11f6b35110617563"
WEIGHT = "ecaf2dfec3caad3cb997995c330b4c9b2581b4754acf78fb0309fbe887f14cc3"
HOOK = "5fa3dc8c0de153098d1800f7b0c809819620aef4"
HOOK_BLOB = "3640624df6dbd994dc11b3ca5cd759bdef8757dd"
TOKENIZER_SHA = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
LENGTHS = (512, 1024, 2048)
TAIL = 16
GENERATE_PREFIX = 512
GENERATE_UNITS = 4
PAD, BOS, EOS = 256, 257, 258
ATOL = RTOL = 0.0005
LANES = {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"}
SOURCE_TEXT = (
    "🔧 café 日本語 πΩ — The workshop hinge shifts across the guide rail while "
    "a mechanic records reversible observations, vibration and load without "
    "changing the frozen baseline. "
) * 24


def require(ok, why):
    if not ok:
        raise RuntimeError("NCR void1 2048 reviewer REFUSED: " + why)


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev(path: Path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def git_blob(path: Path, relative: str):
    return subprocess.check_output(
        ["git", "-C", str(path), "hash-object", relative], text=True).strip()


def fixed_plan():
    require(LENGTHS == (512, 1024, 2048) and TAIL == 16,
            "fixed context/cache lengths changed")
    require(GENERATE_PREFIX == 512 and GENERATE_UNITS == 4,
            "fixed generation source envelope changed")
    require(len(SOURCE_TEXT.encode("utf-8")) > 2048
            and any(ord(c) > 127 for c in SOURCE_TEXT),
            "synthetic source lacks >2048 mixed UTF-8 bytes")
    return True


def compare(report, name, a, b, *, atol=ATOL, rtol=RTOL, exact=False):
    import torch
    require(a.shape == b.shape, name + " shape drift")
    require(bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            name + " nonfinite logits")
    delta = (a.float() - b.float()).abs()
    maximum = float(delta.max())
    passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    if exact:
        passed &= bool(torch.equal(a, b))
    report["checks"][name] = {
        "pass": passed, "max_abs": maximum, "atol": atol, "rtol": rtol,
        "exact_required": exact,
    }
    require(passed, name + " conformance failed")
    return maximum


def generation_compare(report, name, a, b, *,
                       atol=ATOL, rtol=RTOL, exact=False):
    import torch
    require(a.shape == b.shape, name + " score shape drift")
    finite = torch.isfinite(a)
    require(bool(torch.equal(finite, torch.isfinite(b))),
            name + " finite masks differ")
    require(bool(torch.equal(a[~finite], b[~finite])),
            name + " source-masked scores differ")
    require(bool(finite.any()), name + " all scores nonfinite")
    maximum = float((a[finite] - b[finite]).abs().max())
    passed = bool(torch.allclose(a[finite], b[finite], atol=atol, rtol=rtol))
    if exact:
        passed &= bool(torch.equal(a, b))
    report["checks"][name] = {
        "pass": passed, "max_abs_finite": maximum,
        "atol": atol, "rtol": rtol, "exact_required": exact,
        "finite_logits": int(finite.sum()),
        "nonfinite_logits": int((~finite).sum()),
    }
    require(passed, name + " generation score mismatch")
    return maximum


def guard_claims(report):
    require(report["train_authorized"] is False
            and report["optimizer_steps"] == 0
            and report["actual_training"] is False
            and report["heldout_evaluation"] is False
            and report["historical_byte_eos_impact_measured"] is False
            and report["final_four_model_72_conformance"] is False,
            "false training/history/final-admission claim")


def main():
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    report = {
        "schema_version": 1,
        "experiment": "NCR-001E",
        "stage": "void1_exact_512_1024_2048_context_and_HF_generate",
        "status": "RUNNING",
        "model_key": "void1",
        "train_authorized": False,
        "optimizer_steps": 0,
        "actual_training": False,
        "heldout_evaluation": False,
        "historical_byte_eos_impact_measured": False,
        "final_four_model_72_conformance": False,
        "checks": {},
        "generations": [],
        "runner": {
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "runner_name": os.getenv("RUNNER_NAME"),
            "pid": os.getpid(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report["wall_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                  sort_keys=True, allow_nan=False) + "\n",
                       encoding="utf-8")

    def tensor_hash(model):
        h = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            h.update((name + str(tensor.shape) + str(tensor.dtype)).encode())
            h.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    save()
    try:
        require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1",
                "actual frozen model/tokenizer execution must be offline")
        require(fixed_plan(), "fixed source plan changed")
        protocol = json.loads(PROTOCOL.read_text())
        pins = json.loads(PINS.read_text())
        require(protocol["train_authorized"] is False
                and set(protocol["models"]) == set(pins["models"]) == LANES,
                "four-model no-training protocol changed")
        lane = protocol["models"]["void1"]
        pin = pins["models"]["void1"]
        require(lane["repo_id"] == pin["repo_id"] == REPO
                and lane["revision"] == pin["revision"] == REV
                and lane["weight_file"] == pin["hf_weight_file"] == "model.safetensors"
                and pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT
                and pin["weight_bytes"] == 360622744
                and pin["tokenization_byte_py_sha256"] ==
                lane["tokenizer_reference"]["train002b_source_sha256"] ==
                TOKENIZER_SHA
                and lane["expected_config"]["max_position_embeddings"] == 2048
                and lane["taps_after_block_zero_based"] == [7, 15, 23],
                "void1 source/model/tokenizer/context identity changed")
        require(protocol["conformance"]["full_vs_cached_prefill_atol"] ==
                protocol["conformance"]["full_vs_cached_prefill_rtol"] ==
                ATOL == RTOL, "cache tolerances changed")
        require(IMPL.is_dir() and git_rev(IMPL) == HOOK
                and git_blob(IMPL, "scripts/ncr_001_void1.py") == HOOK_BLOB,
                "original Qwen3 NCA hook source changed")
        report["source"] = {
            "repo": REPO, "revision": REV,
            "checkpoint_file": "model.safetensors",
            "checkpoint_sha256": WEIGHT, "checkpoint_bytes": 360622744,
            "original_hook_commit": HOOK, "original_hook_blob": HOOK_BLOB,
            "byte_tokenizer_source_sha256": TOKENIZER_SHA,
            "protocol_sha256": sha(PROTOCOL),
            "source_pins_sha256": sha(PINS),
            "taps": [7, 15, 23],
            "max_context_units": 2048,
        }
        save()

        for filename, field in (
            ("model.safetensors", "weight_declared_lfs_sha256_not_downloaded"),
            ("config.json", "config_sha256"),
            ("tokenization_byte.py", "tokenization_byte_py_sha256"),
            ("byte_vocab.json", "byte_vocab_sha256"),
            ("tokenizer_config.json", "tokenizer_config_sha256"),
            ("special_tokens_map.json", "special_tokens_map_sha256"),
        ):
            path = Path(hf_hub_download(REPO, filename=filename, revision=REV,
                                        local_files_only=True))
            observed = sha(path)
            require(observed == pin[field], "actual pinned source drift: " + filename)
            if filename == "model.safetensors":
                require(path.stat().st_size == 360622744,
                        "complete model source length changed")
            report.setdefault("actual_source_files", {})[filename] = {
                "sha256": observed, "bytes": path.stat().st_size}
            save()

        sys.path.insert(0, str(ROOT / "scripts"))
        from ncr_001k_byte_runtime import verify_reviewed_shape
        source_file = Path(hf_hub_download(
            REPO, filename="tokenization_byte.py", revision=REV,
            local_files_only=True))
        report["reviewed_tokenizer_shape"] = verify_reviewed_shape(
            ast.parse(source_file.read_text(encoding="utf-8"),
                      filename="tokenization_byte.py"))
        tokenizer = AutoTokenizer.from_pretrained(
            REPO, revision=REV, trust_remote_code=True, use_fast=False,
            local_files_only=True)
        executed = inspect.getsourcefile(tokenizer.__class__)
        require(bool(executed) and sha(Path(executed)) == TOKENIZER_SHA,
                "executed ByteTokenizer differs from reviewed source")
        require(tokenizer.vocab_size == 259
                and (tokenizer.pad_token_id, tokenizer.bos_token_id,
                     tokenizer.eos_token_id) == (PAD, BOS, EOS)
                and tokenizer.encode("<eos>", add_special_tokens=True) ==
                [60, 101, 111, 115, 62],
                "ByteTokenizer special/literal EOS semantics changed")
        ids = tokenizer(SOURCE_TEXT, return_tensors="pt",
                        add_special_tokens=True)["input_ids"]
        require(ids.shape[1] >= 2048
                and all(0 <= int(v) < 256 for v in ids[:, :2048].flatten()),
                "mixed UTF-8 plain source does not provide 2048 ordinary byte units")
        ids = ids[:, :2048].contiguous()
        report["input"] = {
            "utf8_source_bytes": len(SOURCE_TEXT.encode("utf-8")),
            "python_characters": len(SOURCE_TEXT),
            "available_tokenizer_units": int(
                tokenizer(SOURCE_TEXT, return_tensors="pt",
                          add_special_tokens=True)["input_ids"].shape[1]),
            "selected_units": 2048,
            "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            "contains_multibyte_utf8": True,
            "pad_id": PAD, "bos_id": BOS, "eos_id": EOS,
        }
        save()

        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        torch.manual_seed(20261027)
        base = AutoModelForCausalLM.from_pretrained(
            REPO, revision=REV, torch_dtype=torch.float32,
            attn_implementation="eager", trust_remote_code=False,
            local_files_only=True).eval()
        sys.path.insert(0, str(IMPL / "scripts"))
        from ncr_001_void1 import FrozenVoid1DepthNCA, POST_BLOCK_TAPS
        require(tuple(POST_BLOCK_TAPS) == (7, 15, 23),
                "original Qwen3 depth taps changed")
        wrapper = FrozenVoid1DepthNCA(
            base, verified_weight_sha256=WEIGHT).eval()
        wrapper.verify_frozen_base()
        require(wrapper.nca.trainable_parameters() == 329944
                and all(not p.requires_grad for p in base.parameters()),
                "void1 frozen/adapter geometry changed")
        report["frozen_base_parameters"] = wrapper.frozen_parameters()
        report["adapter_parameters"] = wrapper.nca.trainable_parameters()
        report["backbone_sha256_before"] = tensor_hash(base)
        save()

        with torch.inference_mode():
            # Source/no-op parity at all three fixed context sizes.
            for n in LENGTHS:
                x = ids[:, :n]
                disabled = wrapper(input_ids=x, enabled=False,
                                   use_cache=False).logits
                zero = wrapper(input_ids=x, enabled=True,
                               use_cache=False).logits
                compare(report, f"zero_exact_{n}", disabled, zero,
                        atol=0, rtol=0, exact=True)

            with torch.no_grad():
                wrapper.nca.readout.weight[0, 0] = 0.03

            # Full versus genuine prefill + 16 sequential cached byte units.
            for n in LENGTHS:
                x = ids[:, :n]
                full = wrapper(input_ids=x, use_cache=False).logits
                disabled = wrapper(input_ids=x, enabled=False,
                                   use_cache=False).logits
                effect = float((full[:, -TAIL:] -
                                disabled[:, -TAIL:]).abs().max())
                require(effect > 0 and bool(torch.isfinite(full).all()),
                        f"untrained adapter inactive/nonfinite at {n}")
                report["checks"][f"manual_untrained_effect_{n}"] = {
                    "max_abs": effect, "trained": False, "pass": True}
                prefill = wrapper(input_ids=x[:, :n-TAIL], use_cache=True)
                past = prefill.past_key_values
                pieces = []
                for index in range(n-TAIL, n):
                    out = wrapper(input_ids=x[:, index:index+1],
                                  past_key_values=past, use_cache=True)
                    pieces.append(out.logits[:, -1:])
                    past = out.past_key_values
                    require(len(wrapper.last_hook_trace) == 1
                            and wrapper.last_hook_trace[0]["state_shape"][:2] == [1, 1]
                            and wrapper.last_hook_trace[0]["finite"] is True,
                            "cached byte field did not reset/finish at one token")
                cached = torch.cat(pieces, dim=1)
                compare(report, f"full_vs_{n-TAIL}_prefill_{TAIL}_cached_{n}",
                        full[:, -TAIL:], cached)
                save()

            # Strict future suffix and A→B→A at max context.
            full2048 = wrapper(input_ids=ids[:, :2048],
                               use_cache=False).logits
            changed = ids[:, :2048].clone()
            changed[:, -1] = (changed[:, -1] + 1) % 256
            altered = wrapper(input_ids=changed, use_cache=False).logits
            compare(report, "strict_future_suffix_2048",
                    full2048[:, :-1], altered[:, :-1],
                    atol=0, rtol=0, exact=True)
            a = wrapper(input_ids=ids[:, :128], use_cache=False).logits
            wrapper(input_ids=ids[:, 128:256], use_cache=False)
            again = wrapper(input_ids=ids[:, :128], use_cache=False).logits
            compare(report, "A_B_A_request_reset_exact",
                    a, again, atol=0, rtol=0, exact=True)

            # Actual GenerationMixin cache on/off, disabled/zero/manual-untrained.
            prompt = ids[:, :GENERATE_PREFIX]
            kwargs = dict(
                input_ids=prompt, do_sample=False,
                max_new_tokens=GENERATE_UNITS,
                min_new_tokens=GENERATE_UNITS,
                eos_token_id=EOS, pad_token_id=PAD,
                return_dict_in_generate=True, output_scores=True,
            )
            runs = {}
            for mode in ("disabled", "zero", "active_untrained"):
                with torch.no_grad():
                    wrapper.nca.readout.weight.zero_()
                    if mode == "active_untrained":
                        wrapper.nca.readout.weight[0, 0] = 0.03
                for cache in (False, True):
                    response = wrapper.generate(
                        **kwargs, use_cache=cache,
                        enabled=(mode != "disabled"))
                    sequence = response.sequences[0].tolist()
                    scores = list(response.scores)
                    require(len(sequence) == GENERATE_PREFIX + GENERATE_UNITS
                            and len(scores) == GENERATE_UNITS,
                            "GenerationMixin did not honor 4-unit reviewer cap")
                    if mode != "disabled":
                        traces = wrapper.last_hook_trace
                        require(len(traces) == GENERATE_UNITS
                                and all(t["tap_count"] == 3 and t["finite"]
                                        for t in traces),
                                "void generation did not complete/reset three taps")
                        lengths = [t["state_shape"][1] for t in traces]
                        expected = ([GENERATE_PREFIX, 1, 1, 1] if cache else
                                    [GENERATE_PREFIX, GENERATE_PREFIX+1,
                                     GENERATE_PREFIX+2, GENERATE_PREFIX+3])
                        require(lengths == expected,
                                "void generate NCA state does not reset per forward")
                    runs[(mode, cache)] = (sequence, scores)
                    report["generations"].append({
                        "mode": mode, "use_cache": cache,
                        "source_prompt_units": GENERATE_PREFIX,
                        "generated_ids": sequence[GENERATE_PREFIX:],
                        "score_sha256": [
                            hashlib.sha256(
                                s.contiguous().cpu().numpy().tobytes()).hexdigest()
                            for s in scores],
                        "actual_eos_258_emitted": EOS in sequence[GENERATE_PREFIX:],
                        "historical_byte_eos_impact_measured": False,
                        "optimizer_steps": 0,
                    })
                    save()

            for cache in (False, True):
                a_ids, a_scores = runs[("disabled", cache)]
                b_ids, b_scores = runs[("zero", cache)]
                require(a_ids == b_ids,
                        "zero adapter changed frozen generation IDs")
                for step in range(GENERATE_UNITS):
                    generation_compare(
                        report, f"generate_zero_exact_cache_{cache}_step_{step}",
                        a_scores[step], b_scores[step],
                        atol=0, rtol=0, exact=True)

            for mode in ("disabled", "zero", "active_untrained"):
                a_ids, a_scores = runs[(mode, False)]
                b_ids, b_scores = runs[(mode, True)]
                require(a_ids == b_ids,
                        "generation IDs changed across cache mode: " + mode)
                for step in range(GENERATE_UNITS):
                    generation_compare(
                        report, f"generate_full_vs_cache_{mode}_step_{step}",
                        a_scores[step], b_scores[step])

        report["backbone_sha256_after"] = tensor_hash(base)
        require(report["backbone_sha256_before"] ==
                report["backbone_sha256_after"],
                "frozen Qwen3 tensor bytes changed")
        require(all(p.grad is None for p in base.parameters()),
                "frozen Qwen3 accumulated gradients")
        require(all(not m._forward_hooks and not m._forward_pre_hooks
                    for m in base.modules()),
                "void1 reviewer left hooks installed")
        guard_claims(report)
        report["status"] = "PASS_VOID1_512_1024_2048_AND_HF_GENERATE_PRETRAIN_ONLY"
        report["limits"] = [
            "Actual pinned frozen Qwen3/ByteTokenizer synthetic UTF-8 source only",
            "No BYTE-EOS historical effect or model stopping propensity conclusion",
            "No optimizer/backward/heldout/comparative training",
            "Not final four-model #72 admission or future serializer integration",
        ]
    except BaseException:
        report["status"] = "FAIL_VOID1_512_1024_2048_OR_HF_GENERATE"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()

    print(json.dumps({"status": report["status"],
                      "checks": report["checks"],
                      "train_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    if "--source-test" in sys.argv:
        print(json.dumps({"fixed_plan": bool(fixed_plan()),
                          "lengths": LENGTHS,
                          "train_authorized": False}))
    else:
        main()
