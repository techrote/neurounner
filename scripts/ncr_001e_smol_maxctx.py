#!/usr/bin/env python3
"""NCR-001E SmolLM2: separate-process frozen 2048-context/cache/HF.generate witness.

This is source/runtime PRETRAIN ONLY. No optimizer, backward, held-out dataset,
comparative NCA/MLP work, or permission to train. Original #89 receipts are immutable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
IMPL = ROOT / "implementation"
OUT = ROOT / "runtime/ncr-001/smol-maxctx"
PROTOCOL = ROOT / "experiments/ncr-001/protocol.json"
PINS = ROOT / "experiments/ncr-001/source-pins.json"
HOOK_COMMIT = "ec0d4a80e15f203a06c26ee9c23858f78976370b"
WEIGHT_SHA = "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
MODEL_REV = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
CACHE_LENGTHS = (512, 1024, 2048)
TAIL = 16


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok: bool, why: str) -> None:
    if not ok:
        raise RuntimeError(why)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def base_digest(model) -> str:
    # Original tensor bytes, not serialized metadata or a model-output surrogate.
    h = hashlib.sha256()
    for name, param in model.named_parameters():
        h.update(name.encode("utf-8") + b"\x00")
        h.update(param.detach().contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


def process_info() -> dict:
    cpu = "unknown"
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "pid": os.getpid(), "cpu_model": cpu, "platform": platform.platform(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "threads": torch.get_num_threads(),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
        "github_head_sha": os.getenv("NCR_HEAD_SHA"),
        "runner_name": os.getenv("RUNNER_NAME"),
    }


def assert_pair(result: dict, key: str, a: torch.Tensor, b: torch.Tensor, *,
                atol: float, rtol: float) -> None:
    require(a.shape == b.shape, key + ": shape mismatch")
    d = (a - b).abs()
    entry = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
             "atol": atol, "rtol": rtol,
             "within": bool(torch.allclose(a, b, atol=atol, rtol=rtol))}
    result["checks"][key] = entry
    require(entry["within"], key + ": " + str(entry))


def source_and_model(result: dict):
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    require(p["experiment"] == "NCR-001" and p["train_authorized"] is False,
            "NCR train admission must remain false")
    require(set(p["models"]) == {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
            "Four frozen lane identities changed")
    pin = pins["models"]["smollm2"]
    require(pin["repo_id"] == MODEL_ID and pin["revision"] == MODEL_REV
            and pin["hf_weight_file"] == "model.safetensors"
            and pin["weight_declared_lfs_sha256_not_downloaded"] == WEIGHT_SHA
            and pin["weight_bytes"] == 269060552
            and p["models"]["smollm2"]["taps_after_block_zero_based"] == [9, 19, 29],
            "SmolLM2 source/tap/weight pin mismatch")
    head = subprocess.check_output(["git", "-C", str(IMPL), "rev-parse", "HEAD"],
                                   text=True).strip()
    require(head == HOOK_COMMIT, "Original hook is not exact reviewed PR #83 source")
    sys.path.insert(0, str(IMPL / "scripts"))
    from ncr_001_smollm2 import FrozenSmolLM2DepthNCA, POST_BLOCK_TAPS
    require(list(POST_BLOCK_TAPS) == [9, 19, 29], "Hook tap indices changed")
    weight = Path(hf_hub_download(MODEL_ID, filename="model.safetensors",
                                  revision=MODEL_REV))
    require(weight.stat().st_size == 269060552 and sha(weight) == WEIGHT_SHA,
            "Actual complete model weight bytes mismatch")
    # Explicitly hash the original source files, not only the declared LFS digest.
    from huggingface_hub import hf_hub_download as fetch
    verified = {}
    for filename, field in (("config.json", "config_sha256"),
                            ("tokenizer.json", "tokenizer_sha256"),
                            ("tokenizer_config.json", "tokenizer_config_sha256")):
        source_path = Path(fetch(MODEL_ID, filename=filename, revision=MODEL_REV))
        actual = sha(source_path)
        require(actual == pin[field], filename + " source pin mismatch")
        verified[filename] = {"sha256": actual, "bytes": source_path.stat().st_size}
    result["source"] = {
        "repo": MODEL_ID, "revision": MODEL_REV, "hook_commit": HOOK_COMMIT,
        "weight_file": "model.safetensors", "actual_full_weight_sha256": WEIGHT_SHA,
        "actual_weight_bytes": weight.stat().st_size, "source_files": verified,
        "protocol_sha256": sha(PROTOCOL), "source_pins_sha256": sha(PINS),
        "taps": [9, 19, 29],
    }
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, torch_dtype=torch.float32,
        attn_implementation="eager"
    ).eval()
    wrapper = FrozenSmolLM2DepthNCA(model).eval()
    wrapper.verify_frozen_base()
    require(all(not x.requires_grad and x.grad is None
                for x in model.parameters()), "Backbone not frozen")
    require(wrapper.nca.trainable_parameters() == 297048, "Wrong NCA parameter count")
    return p, model, wrapper, AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REV
    )


def synthetic_ids(tokenizer) -> torch.Tensor:
    # This is a deliberately repetitive diagnostics-only string, NOT a held-out
    # behavioural or natural-language evaluation fixture.
    words = ("Workshop latch spring guide rail metal bracket torque "
             "inspection record lever alignment observation. ")
    ids = tokenizer(words * 700, add_special_tokens=False,
                    return_tensors="pt")["input_ids"]
    require(ids.shape[1] >= max(CACHE_LENGTHS), "Synthetic diagnostic text too short")
    return ids[:, :max(CACHE_LENGTHS)]


def producer(result: dict, p: dict, model, wrapper, tokenizer):
    atol = p["conformance"]["frozen_base_noop_atol"]
    rtol = p["conformance"]["frozen_base_noop_rtol"]
    ct = p["conformance"]["full_vs_cached_prefill_atol"]
    cr = p["conformance"]["full_vs_cached_prefill_rtol"]
    before_sha = base_digest(model)
    ids_all = synthetic_ids(tokenizer)
    with torch.inference_mode():
        for length in CACHE_LENGTHS:
            ids = ids_all[:, :length]
            disabled = wrapper(input_ids=ids, enabled=False, use_cache=False).logits
            zero = wrapper(input_ids=ids, enabled=True, use_cache=False).logits
            assert_pair(result, f"zero_noop_{length}", disabled, zero, atol=atol, rtol=rtol)
            require(torch.equal(disabled, zero), "Zero-init parity not exact at " + str(length))
            result["checks"][f"zero_noop_{length}"]["exact"] = True
            require(len(wrapper.last_hook_trace) == 1
                    and wrapper.last_hook_trace[0]["finite"],
                    "NCA trace state not finite/complete")
            del disabled, zero
            gc.collect()
        # Manual nonlearned readout; NEVER an optimizer step.
        wrapper.nca.readout.weight[0, 0] = 0.03
        probe = ids_all[:, :64]
        delta = (wrapper(input_ids=probe, enabled=True, use_cache=False).logits -
                 wrapper(input_ids=probe, enabled=False, use_cache=False).logits).abs().max()
        require(float(delta) > 0, "Manual NCA probe has zero effect")
        result["checks"]["manually_activated_untrained_effect_max_abs"] = float(delta)
        for length in CACHE_LENGTHS:
            ids = ids_all[:, :length]
            full = wrapper(input_ids=ids, use_cache=False).logits[:, -TAIL:, :].clone()
            pre = wrapper(input_ids=ids[:, :-TAIL], use_cache=True)
            past = pre.past_key_values
            pieces = []
            for index in range(length - TAIL, length):
                nxt = wrapper(input_ids=ids[:, index:index+1],
                              past_key_values=past, use_cache=True)
                past = nxt.past_key_values
                pieces.append(nxt.logits[:, -1:, :])
                require(len(wrapper.last_hook_trace) == 1
                        and wrapper.last_hook_trace[0]["state_shape"][1] == 1,
                        "Cached token reused spatial field from a prior token")
            incremental = torch.cat(pieces, dim=1)
            assert_pair(result, f"active_tail_{length}_prefill_{length-TAIL}_plus_{TAIL}",
                        full, incremental, atol=ct, rtol=cr)
            del full, pre, past, pieces, incremental, nxt
            gc.collect()
        a = ids_all[:, :256]
        changed = a.clone()
        changed[:, -1] = (changed[:, -1] + 17) % model.config.vocab_size
        earlier = wrapper(input_ids=a, use_cache=False).logits[:, :-1]
        altered = wrapper(input_ids=changed, use_cache=False).logits[:, :-1]
        assert_pair(result, "future_suffix_256", earlier, altered, atol=ct, rtol=cr)
        require(result["checks"]["future_suffix_256"]["max_abs"] == 0,
                "Future token changed earlier logits")
        a = ids_all[:, :80]
        b = ids_all[:, 80:160]
        first = wrapper(input_ids=a, use_cache=False).logits.clone()
        wrapper(input_ids=b, use_cache=False)
        again = wrapper(input_ids=a, use_cache=False).logits
        require(torch.equal(first, again), "A/B/A request field leaked")
        result["checks"]["request_A_B_A_exact"] = True
        # The wrapper delegates into the ORIGINAL Hugging Face generate() path.
        # At zero NCA readout, model output must exactly equal the frozen model.
        original_weight = wrapper.nca.readout.weight[0, 0].clone()
        saved = {}
        prefix = ids_all[:, :128]
        for mode in ("disabled", "zero", "active"):
            wrapper.nca.readout.weight[0, 0] = (0 if mode == "zero" else original_weight)
            for use_cache in (False, True):
                output = wrapper.generate(
                    input_ids=prefix, enabled=(mode != "disabled"),
                    max_new_tokens=4, do_sample=False, use_cache=use_cache,
                    output_scores=True, return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                require(len(output.scores) > 0, "HF generate yielded no processed scores")
                saved[(mode, use_cache)] = (
                    output.sequences.clone(),
                    torch.stack(output.scores).clone(),
                )
        wrapper.nca.readout.weight[0, 0] = original_weight
        for mode in ("disabled", "zero", "active"):
            off, on = saved[(mode, False)], saved[(mode, True)]
            require(torch.equal(off[0], on[0]), mode + ": generation token IDs cache-dependent")
            assert_pair(result, "hf_generate_" + mode + "_cache_on_off_scores",
                        off[1], on[1], atol=ct, rtol=cr)
        for use_cache in (False, True):
            disabled = saved[("disabled", use_cache)]
            zero = saved[("zero", use_cache)]
            require(torch.equal(disabled[0], zero[0]), "HF zero readout tokens mismatch")
            assert_pair(result, "hf_generate_frozen_vs_zero_scores_cache_" + str(use_cache),
                        disabled[1], zero[1], atol=atol, rtol=rtol)
            require(torch.equal(disabled[1], zero[1]),
                    "HF zero readout scores not exactly frozen")
        result["checks"]["hf_generate_raw_ids"] = {
            k[0] + "_cache_" + str(k[1]): v[0][0, -4:].tolist()
            for k, v in saved.items()
        }
        # Persistent only across DEPTH, not successive output tokens/requests.
        ref_ids = ids_all[:, :80].clone()
        ref_logits = wrapper(input_ids=ref_ids, use_cache=False).logits[:, -1:, :128].clone()
        wrapper.save_adapter(OUT / "adapter")
        torch.save({"ids": ref_ids, "logits": ref_logits}, OUT / "cold-reference.pt")
    result["checks"]["backbone_before_after_tensor_sha256"] = {
        "before": before_sha, "after": base_digest(model),
    }
    require(before_sha == result["checks"]["backbone_before_after_tensor_sha256"]["after"],
            "Frozen backbone tensor bytes changed")
    require(all(x.grad is None for x in model.parameters()), "Backbone gradients observed")
    result["cold_reference"] = {
        "sha256": sha(OUT / "cold-reference.pt"),
        "adapter_state_sha256": sha(OUT / "adapter/adapter.pt"),
        "adapter_metadata_sha256": sha(OUT / "adapter/metadata.json"),
    }


def consumer(result: dict, p: dict, model, wrapper, _tokenizer):
    source = json.loads((OUT / "producer.json").read_text(encoding="utf-8"))
    require(source["status"] == "PASS_PRETRAIN_ONLY"
            and source["optimizer_steps"] == 0
            and source["process"]["pid"] != os.getpid(),
            "No valid separate Python producer interpreter")
    require(source["source"] == result["source"], "Producer/consumer source identity drift")
    for field, relative in (("sha256", "cold-reference.pt"),
                            ("adapter_state_sha256", "adapter/adapter.pt"),
                            ("adapter_metadata_sha256", "adapter/metadata.json")):
        require(sha(OUT / relative) == source["cold_reference"][field],
                "Cross-process handoff bytes mismatch: " + relative)
    before = base_digest(model)
    wrapper.load_adapter(OUT / "adapter")
    reference = torch.load(OUT / "cold-reference.pt", map_location="cpu", weights_only=True)
    require(set(reference) == {"ids", "logits"}, "Invalid cold reference schema")
    with torch.inference_mode():
        got = wrapper(input_ids=reference["ids"], use_cache=False).logits[:, -1:, :128]
    assert_pair(result, "separate_process_reloaded_active_adapter_exact",
                reference["logits"], got, atol=0, rtol=0)
    # Negative revision refusal does not modify the original state permanently.
    metadata = OUT / "adapter/metadata.json"
    original = metadata.read_bytes()
    try:
        altered = json.loads(original)
        altered["model_revision"] = "wrong-variant-must-fail"
        metadata.write_text(json.dumps(altered), encoding="utf-8")
        rejected = False
        try:
            wrapper.load_adapter(OUT / "adapter")
        except RuntimeError:
            rejected = True
        require(rejected, "Cold loading with wrong revision did not refuse")
        result["checks"]["wrong_revision_cold_load_refused"] = True
    finally:
        metadata.write_bytes(original)
    require(sha(metadata) == source["cold_reference"]["adapter_metadata_sha256"],
            "Original adapter metadata not restored")
    require(base_digest(model) == before, "Frozen consumer base changed")
    require(all(x.grad is None for x in model.parameters()), "Consumer base gradients observed")
    result["producer_run_id"] = source["process"]["github_run_id"]
    result["producer_pid"] = source["process"]["pid"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("producer", "consumer"))
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261027)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1, "experiment": "NCR-001E", "lane": "smollm2",
        "phase": args.phase, "status": "RUNNING", "process": process_info(),
        "checks": {}, "training_authorized": False, "training_performed": False,
        "optimizer_steps": 0, "heldout_evaluation": False,
        "comparison_training": False, "per_model_gate": False,
        "full_four_model_conformance_pass": False,
        "limitations": [
            "Frozen SmolLM2 source/cache/HF.generate and one same-runner two-Python-process cold replay only",
            "Diagnostic manually activated readout, not learned weights or optimizer updates",
            "No supervised learning-step/gradient-after-warmup/finite-loss qualification",
            "No cross-host reboot or model-generated behavioural evaluation",
        ],
    }
    destination = OUT / (args.phase + ".json")
    save_json(destination, result)
    start = time.monotonic()
    try:
        protocol, base, adapter, tokenizer = source_and_model(result)
        save_json(destination, result)
        if args.phase == "producer":
            producer(result, protocol, base, adapter, tokenizer)
        else:
            consumer(result, protocol, base, adapter, tokenizer)
        result["status"] = "PASS_PRETRAIN_ONLY"
    except Exception as exc:
        result["status"] = "FAIL_RETAIN_RAW_ERROR"
        result["error"] = type(exc).__name__ + ": " + str(exc)
        result["traceback"] = traceback.format_exc()[-12000:]
        raise
    finally:
        result["wall_seconds"] = round(time.monotonic() - start, 3)
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save_json(destination, result)
        print(json.dumps({"phase": args.phase, "status": result["status"],
                          "checks": result["checks"],
                          "peak_rss_kib": result["peak_rss_kib"]},
                         sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
