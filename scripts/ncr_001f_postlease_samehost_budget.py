#!/usr/bin/env python3
"""NCR-001F: same-frozen-host NCA/MLP CPU forward budget, NO TRAINING.

Only synthetic ordinary-role inputs. Measures complete frozen-host forward time,
not adapter-only time, FLOPs, learned quality or scientific admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time
import traceback

import torch
import transformers
import huggingface_hub
import ncr_001_postlease_sprocket as host

ROOT = Path(__file__).resolve().parents[1]
PARENT = "2d5c0ed72c3d1c44f78ba58493bfffd8fab26d80"
FORMAT_HEAD = "2271218d625e1462b2cf218445da975efe17fe32"
PROTOCOL_BLOB = "14ce5fbaa38f8806c8c8738f19c01b91385eebaf"
PINS_BLOB = "59fff1aed872b5563aa6898171983973e0cdb02c"
LENGTHS = (512, 2048)
WARMUPS = 1
REPEATS = 3
KINDS = ("frozen", "nca", "mlp")
STATUS = "MEASURED_SAME_FROZEN_HOST_SYNTHETIC_FORWARD_ONLY__BUDGET_NOT_ADMITTED"
FAILURE = "FAILED_SAME_FROZEN_HOST_SYNTHETIC_FORWARD__BUDGET_NOT_ADMITTED"


def refuse(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("NCR001F_SAMEHOST_REFUSED: " + reason)


def gitblob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\x00" + data).hexdigest()


def validate_budget(lengths=LENGTHS, warmups=WARMUPS, repeats=REPEATS) -> None:
    refuse(type(lengths) is tuple and lengths == LENGTHS,
           "only fixed 512/2048-unit synthetic contexts are allowed")
    refuse(type(warmups) is int and warmups == WARMUPS and
           type(repeats) is int and repeats == REPEATS,
           "fixed warmup/repeat schedule changed")


def check_contract(variant: str) -> tuple[dict, dict, dict]:
    refuse(variant in ("chat", "instruct"), "unsupported or collapsed model lane")
    protocol_path = ROOT / "experiments/ncr-001/protocol.json"
    pins_path = ROOT / "experiments/ncr-001/source-pins.json"
    refuse(gitblob(protocol_path.read_bytes()) == PROTOCOL_BLOB and
           gitblob(pins_path.read_bytes()) == PINS_BLOB,
           "preregistered protocol/source Git blob drift")
    p = json.loads(protocol_path.read_text(encoding="utf-8"))
    q = json.loads(pins_path.read_text(encoding="utf-8"))
    refuse(p["train_authorized"] is False and
           set(p["models"]) == set(q["models"]) ==
           {"smollm2", "sprocket_chat", "sprocket_instruct", "void1"},
           "four-lane training closure or source inventory changed")
    refuse(p["conformance"]["full_vs_cached_prefill_atol"] == 0.0005 and
           p["conformance"]["full_vs_cached_prefill_rtol"] == 0.0005,
           "causal cache numeric gate changed")
    spec = host.verify_contract(variant, p, q)
    return spec, p, q["models"]["sprocket_" + variant]


def require_same_backbone(model, nca, mlp, variant: str) -> None:
    refuse(nca.backbone is model and mlp.backbone is model,
           "NCA/MLP are not attached to the SAME frozen model")
    refuse(nca._backbone_lock is mlp._backbone_lock,
           "NCA/MLP do not share the existing decoder-root ownership lease")
    refuse(nca.nca.trainable_parameters() == 658904 and
           mlp.control.trainable_parameters() == 658432,
           "preregistered matched parameter counts changed")
    refuse(all(not p.requires_grad for p in model.parameters()),
           "original frozen checkpoint became trainable")
    refuse(variant in ("chat", "instruct"), "invalid variant")
    refuse(not getattr(model, "is_gradient_checkpointing", False),
           "checkpointed model path is not qualified")


def measure_schedule(callables: dict, *, repeats=REPEATS, warmups=WARMUPS) -> dict:
    """Interleaved full forward calls. No assertions/finite scans INSIDE timed spans."""
    validate_budget(LENGTHS, warmups, repeats)
    refuse(set(callables) == set(KINDS) and
           all(callable(fn) for fn in callables.values()),
           "missing frozen/NCA/MLP same-input paths")
    for _ in range(warmups):
        for kind in KINDS:
            callables[kind]()
    nanos = {kind: [] for kind in KINDS}
    for iteration in range(repeats):
        order = KINDS[iteration:] + KINDS[:iteration]
        for kind in order:
            start = time.perf_counter_ns()
            callables[kind]()
            elapsed = time.perf_counter_ns() - start
            refuse(type(elapsed) is int and elapsed > 0,
                   "nonpositive monotonic full-host wall sample")
            nanos[kind].append(elapsed)
    medians = {kind: statistics.median(nanos[kind]) / 1e6 for kind in KINDS}
    return {
        "forward_call_order": [
            list(KINDS[i:] + KINDS[:i]) for i in range(repeats)
        ],
        "warmup_rounds": warmups, "measured_rounds": repeats,
        "raw_complete_frozen_host_forward_ns": nanos,
        "median_complete_host_forward_ms": medians,
        "nca_over_mlp_total_host_wall_ratio":
            medians["nca"] / medians["mlp"],
        "baseline_is_same_original_frozen_model": True,
        "wall_time_includes_full_backbone": True,
        "timed_interval_includes_posthoc_tensor_validation": False,
        "measured_flops": False, "trained_quality": False,
        "actual_compute_budget_qualified": False,
    }


def require_runtime_threads(report: dict) -> None:
    """Fail closed on the actual timing-path threads, not startup metadata."""
    effective = torch.get_num_threads()
    refuse(effective == 4 and
           report["runner"].get("cpu_threads") == effective,
           "four-thread CPU timing contract is not active or report is stale")


def save(destination: Path, report: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    report["peak_process_ru_maxrss_kib"] = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_report(variant: str) -> dict:
    return {
        "experiment": "NCR-001F", "issue": 73,
        "variant": variant, "status": "STARTED_NO_TRAIN",
        "parent_source_head": PARENT, "formatter_source_head": FORMAT_HEAD,
        "train_authorized": False, "full_72_conformance_pass": False,
        "actual_compute_budget_qualified": False,
        "scientific_admission": False, "heldouts_opened": False,
        "training_performed": False, "optimizer_steps": 0,
        "comparative_nca_mlp_training": False,
        "generated_tool_memory_semantics_qualified": False,
        "untrained_manual_adapter_readouts": True,
        "frozen_host_loaded": False, "synthetic_cases": [],
        "limitations": [
            "Complete-host inference time, not measured FLOPs or isolated adapter overhead",
            "Single process/CPU per lane; host scheduling and cache effects are uncontrolled",
            "ru_maxrss is the SHARED PROCESS peak, never attributable to one adapter",
            "Chat and root Instruct source identities separate; never compare quality across lanes",
            "Synthetic ordinary-role text, untrained manual readouts, no held-out fixtures",
            "No publisher tool/result/memory generated competence or full four-lane #72 acceptance",
        ],
        "runner": {
            "github_run_id": os.getenv("GITHUB_RUN_ID"),
            "github_run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "github_sha": os.getenv("GITHUB_SHA"),
            "expected_pr_head": os.getenv("NCR_EXPECTED_HEAD"),
            "pid": os.getpid(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "cpu_threads_at_report_start": torch.get_num_threads(),
            "cpu_threads": None,  # set only after the actual four-thread contract is active
            "cpu_model": next(
                (line.partition(":")[2].strip()
                 for line in Path("/proc/cpuinfo").read_text().splitlines()
                 if line.startswith("model name")), "unknown"
            ) if Path("/proc/cpuinfo").exists() else "unknown",
        },
    }


def run(variant: str, destination: Path, report: dict) -> None:
    validate_budget()
    expected = os.getenv("NCR_EXPECTED_HEAD")
    actual = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    refuse(os.getenv("GITHUB_EVENT_NAME") == "pull_request" and
           expected and expected == actual and
           os.getenv("HF_HUB_OFFLINE") == "1" and
           os.getenv("TRANSFORMERS_OFFLINE") == "1",
           "must run exact PR-head fully offline after pinned prefetch")
    refuse(subprocess.check_output(
        ["git", "-C", str(ROOT / "format-authority"), "rev-parse", "HEAD"],
        text=True).strip() == FORMAT_HEAD,
        "immutable independent publisher ordinary-role formatter checkout changed")
    torch.set_num_threads(4)
    report["runner"]["cpu_threads"] = torch.get_num_threads()
    require_runtime_threads(report)
    torch.manual_seed(20261029)
    spec, protocol, pin = check_contract(variant)
    report["source_identity"] = {
        "repo": spec["model_id"], "revision": spec["revision"],
        "weight_path": spec["weight_path"], "weight_sha256": spec["weight_sha256"],
        "source_protocol_git_blob": PROTOCOL_BLOB, "source_pins_git_blob": PINS_BLOB,
    }
    save(destination, report)
    formatter = host.load_formatter()
    model, tok, nca, mlp, actual_source_files, serialize = host.load_actual(
        variant, spec, pin, formatter)
    require_same_backbone(model, nca, mlp, variant)
    report["frozen_host_loaded"] = True
    report["original_source_file_sha256"] = actual_source_files
    report["frozen_backbone_parameters"] = sum(p.numel() for p in model.parameters())
    report["nca_trainable_parameters"] = nca.nca.trainable_parameters()
    report["mlp_trainable_parameters"] = mlp.control.trainable_parameters()
    before = host.fingerprint_backbone(model)
    report["frozen_backbone_before_sha256"] = before
    save(destination, report)

    ordinary = (
        "Inspect the bearing, verify the rail and check the drive. "
        "Record reversible observations before a shutdown. "
    ) * 180
    tokens = host.ids_for(tok, serialize, variant, ordinary)
    refuse(tokens.ndim == 2 and tokens.shape[0] == 1 and
           tokens.shape[1] >= 2048 and tokens[0, 0].item() == 3,
           "pinned ordinary-role synthetic input shorter than 2048 or wrong role")
    tokens = tokens[:, :2048].contiguous()
    report["input_token_ids_sha256"] = host.tensor_hash(tokens)
    report["actual_context_units"] = list(LENGTHS)
    with torch.inference_mode():
        short = tokens[:, :64]
        bare = model(input_ids=short, use_cache=False).logits
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            zero = wrapper(input_ids=short, use_cache=False).logits
            refuse(torch.equal(bare, zero),
                   kind + " zero initialized output mismatches own frozen reference")
        with torch.no_grad():
            nca.nca.readout.weight[0, 0] = 0.03
            mlp.control.down.weight[0, 0] = 0.05
        for kind, wrapper in (("nca", nca), ("mlp", mlp)):
            active = wrapper(input_ids=short, use_cache=False).logits
            effect = float((active - bare).abs().max())
            refuse(effect > 0 and bool(torch.isfinite(active).all()),
                   kind + " manually activated untrained host not finite/nonzero")
            report[kind + "_manual_untrained_effect"] = effect
        save(destination, report)
        for length in LENGTHS:
            require_runtime_threads(report)
            x = tokens[:, :length]
            callables = {
                "frozen": lambda x=x: model(input_ids=x, use_cache=False).logits,
                "nca": lambda x=x: nca(input_ids=x, use_cache=False).logits,
                "mlp": lambda x=x: mlp(input_ids=x, use_cache=False).logits,
            }
            row = measure_schedule(callables)
            require_runtime_threads(report)
            row["synthetic_shape"] = [1, length]
            report["synthetic_cases"].append(row)
            save(destination, report)
            # Explicit untimed postflight: no leaked hooks, no NaNs, stable requests.
            for kind, fn in callables.items():
                logits = fn()
                refuse(bool(torch.isfinite(logits).all()),
                       kind + " nonfinite postflight full-host output")
            refuse(not model.model._forward_hooks and
                   not model.model._forward_pre_hooks,
                   "original decoder retains leaked hooks after timing")
    after = host.fingerprint_backbone(model)
    report["frozen_backbone_after_sha256"] = after
    refuse(before == after and
           all(p.grad is None for p in model.parameters()),
           "original frozen backbone was modified or accumulated gradients")
    report["status"] = STATUS
    save(destination, report)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("chat", "instruct"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = base_report(args.variant)
    save(args.out, report)
    try:
        run(args.variant, args.out, report)
    except BaseException as exc:
        report["status"] = FAILURE
        report["exception_type"] = type(exc).__name__
        report["traceback"] = traceback.format_exc()[-10000:]
        save(args.out, report)
        raise
    print(json.dumps({"status": STATUS, "variant": args.variant,
                      "cases": len(report["synthetic_cases"]),
                      "train_authorized": False,
                      "full_72_conformance_pass": False}, sort_keys=True))


if __name__ == "__main__":
    main()
