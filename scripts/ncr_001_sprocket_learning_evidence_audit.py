#!/usr/bin/env python3
"""NCR-001: independently read back byte-exact original Sprocket diagnostic receipts.

Only Python standard library. This auditor NEVER executes frozen models,
touches sealed fixtures, performs optimization or grants #72 admission.
The original physical job HEAD, immutable source files, model variants and
original positive/negative artifacts must all agree. Fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/ncr-001/controls-evidence/sprocket-learning-20261028"
SOURCE_HEAD = "df8479aae34550cba014d910a73e30e59cd493e2"
PINNED_REVISION = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
EXPECTED_ZIPS = {
    "negative_source": "c597e65b9bc095af3382646b76b7663693cd91838df36174df15c79bf3634cc2",
    "positive_source": "c66706d3e541925638e901d37232f204c480f0016aba393023fa925c67a61242",
    "sprocket_chat": "e040e1ce982a1c0f0c5e99e014cada04cc41b334f1755d60cf37fd0bb909a05b",
    "sprocket_instruct": "9656ff3f8019f497cff62d816276f44bdc33f149c07072357bd5cfd9c90d0f93",
}
WEIGHTS = {
    "sprocket_chat": ("chat/model.safetensors",
        "551c6e262ec4561dda7c74c9ef7cb73bd1a555ad1e05e37e6600c84a5acefe97"),
    "sprocket_instruct": ("model.safetensors",
        "1a9fd1066f3f0873c4f5ae99b75beab3b0721100f45b036f9e255611b220d864"),
}
UPSTREAM = ("observe.weight", "perception.weight", "rule.weight")
REQUIRED_CHECKS = {
    "assistant_only_finite_loss", "readout_first_step_gradient",
    "post_warmup_upstream_gradients",
    "one_diagnostic_update_changes_only_nca", "source_baseline_unchanged",
    "three_distinct_ordered_depth_states",
}


def require(test, reason: str) -> None:
    if not test:
        raise RuntimeError(reason)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha(raw: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw)
    return h.hexdigest()


def guarded_positive(value, desc: str) -> None:
    require(type(value) in (float, int) and math.isfinite(value) and value > 0,
            desc + ": finite positive value required")


def audit_receipt(receipt: dict, *, lane: str, artifact: dict, root: Path) -> dict:
    """Physical artifact decision independent from CI green badge."""
    require(lane in WEIGHTS, "Unknown Sprocket variant")
    require(receipt.get("experiment") == "NCR-001"
            and receipt.get("lane") == lane, "Sprocket lane identity drift")
    require(receipt.get("status")
            == "PASS_DIAGNOSTIC_NCA_LEARNING_GATES__FINAL_72_BLOCKED",
            "Original diagnostic status is not PASS")
    for key in ("train_authorized", "comparative_training_performed",
                "preregistered_evaluation_performed", "heldout_opened",
                "model_training_experiment", "full_72_conformance_pass",
                "tool_memory_semantics_qualified", "independent_review_complete"):
        require(receipt.get(key) is False, "Unauthorized admission flag: " + key)
    require(receipt.get("diagnostic_optimizer_steps") == 1
            and receipt.get("original_frozen_model_executed") is True,
            "One original-model diagnostic update required; no comparisons")

    runner = receipt.get("runner", {})
    require(runner.get("run_id") == str(artifact["run_id"])
            and runner.get("head_sha") == SOURCE_HEAD
            and runner.get("attempt") == "1"
            and runner.get("runner_os") == "Linux"
            and runner.get("runner_arch") == "X64"
            and runner.get("torch") == "2.14.1+cpu"
            and runner.get("transformers") == "4.57.6"
            and type(runner.get("pid")) is int and runner["pid"] > 0
            and runner.get("runner_name")
            and runner.get("host")
            and runner.get("python")
            and runner.get("threads") == 4, "Runner/source-head/host mismatch")

    src = receipt.get("source_identity", {})
    weight_path, weight_sha = WEIGHTS[lane]
    require(src.get("model_id") == "kandivault/sprocket-500m"
            and src.get("revision") == PINNED_REVISION
            and src.get("weight_path") == weight_path
            and src.get("weight_sha256") == artifact["weight_sha256"] == weight_sha
            and src.get("model_type") == "llama"
            and src.get("layers") == 26
            and src.get("width") == 1280
            and src.get("taps") == [7, 16, 25], "Distinct pinned model identity wrong")
    original_weight = receipt.get("verified_original_source_files", {}).get(weight_path, {})
    require(original_weight.get("sha256") == weight_sha
            and original_weight.get("bytes") == 1002207984,
            "Original downloaded complete model bytes not established")
    fmt = receipt.get("formatter", {})
    require(fmt.get("commit") == "2271218d625e1462b2cf218445da975efe17fe32"
            and fmt.get("sha256") ==
            "281a5d8ac2c788912db8b9ef80183b93186f457563a713ff1435ae5617e91fbd4"
            and "tool/memory semantics not inferred" in fmt.get("scope", ""),
            "Original source-bound ordinary-role formatter changed")

    source_hashes = receipt.get("source_sha256", {})
    require(set(source_hashes) == {
        "scripts/ncr_001_sprocket_learning_gate.py",
        "scripts/ncr_001_postlease_sprocket.py",
        "scripts/ncr_001_core.py",
        "scripts/ncr_001_sprocket_chat.py",
        "scripts/ncr_001_sprocket_instruct.py",
        "scripts/ncr_001_backbone_lease.py",
        "experiments/ncr-001/protocol.json",
        "experiments/ncr-001/source-pins.json",
    }, "Source manifest incomplete or expanded")
    for name, expected in source_hashes.items():
        require(sha256((root / name).read_bytes()) == expected,
                "Original physical source no longer matches checked code: " + name)
    spec = json.loads((root / "experiments/ncr-001/protocol.json").read_bytes())
    require(spec.get("train_authorized") is False and
            set(spec.get("models", {})) == {
                "smollm2", "sprocket_chat", "sprocket_instruct", "void1"
            }, "Original four-lane train gate no longer immutable")

    sample = receipt.get("synthetic_sample", {})
    require(sample.get("synthetic_fixture") is True
            and sample.get("heldout") is False
            and sample.get("assistant_eos_id") == 5
            and sample.get("prefix_units") == 12
            and sample.get("supervised_units") == 7
            and sample.get("sequence_units") == 19,
            "Synthetic assistant-only EOS label/mask boundary changed")
    for name in ("ids_sha256", "labels_sha256", "attention_mask_sha256"):
        require(bool(re.fullmatch("[a-f0-9]{64}", sample.get(name, ""))),
                "Synthetic sample hash missing: " + name)
    require(receipt.get("initial_zero_readout_source_parity_exact") is True,
            "Initial source reference contaminated")
    original_frozen = receipt.get("backbone_before_sha256")
    require(bool(re.fullmatch("[a-f0-9]{64}", original_frozen or ""))
            and receipt.get("backbone_after_sha256") == original_frozen,
            "Original backbone mutated during diagnostic update")
    require(receipt.get("checks") == dict.fromkeys(REQUIRED_CHECKS, True),
            "Original finite-loss/gradient/frozen/state assertions not all passed")

    warmup = receipt.get("warmup", {})
    post = receipt.get("post_warmup", {})
    guarded_positive(warmup.get("loss"), "First synthetic supervised loss")
    guarded_positive(post.get("loss"), "Post-warmup supervised loss")
    require(warmup.get("readout_before_sha256") != warmup.get("readout_after_sha256")
            and all(re.fullmatch("[a-f0-9]{64}", warmup.get(k, ""))
                    for k in ("readout_before_sha256", "readout_after_sha256")),
            "Readout update absent")
    early_grad = warmup.get("gradient_abs_sums", {})
    late_grad = post.get("gradient_abs_sums", {})
    guarded_positive(early_grad.get("readout.weight"), "First-step readout gradient")
    for name in UPSTREAM:
        require(early_grad.get(name) == 0,
                "Upstream gradient was nonzero through zero readout: " + name)
    for name in (*UPSTREAM, "readout.weight"):
        guarded_positive(late_grad.get(name), "After-warmup NCA gradient " + name)
    trace = post.get("ordered_field_trace", [])
    require(len(trace) == 3 and
            [row.get("tap_index") for row in trace] == [0, 1, 2]
            and [row.get("next_tap") for row in trace] == [1, 2, 3]
            and all(row.get("shape") == [1, 19, 4, 8, 8] for row in trace)
            and len({row.get("field_sha256") for row in trace}) == 3
            and all(re.fullmatch("[a-f0-9]{64}", row.get("field_sha256", ""))
                    for row in trace), "Original persistent-depth field not observed")
    return {
        "lane": lane, "original_host_executed": True,
        "diagnostic_learning_gates_pass": True,
        "backbone_unchanged": True,
        "pre_update_loss": warmup["loss"], "post_update_loss": post["loss"],
        "post_warmup_required_gradient_abs_sums": {
            k: late_grad[k] for k in (*UPSTREAM, "readout.weight")
        },
        "source_head": SOURCE_HEAD,
        "final_72_conformance_pass": False,
        "train_authorized": False,
    }


def audit(root: Path = ROOT, evidence: Path = DIR) -> dict:
    manifest = json.loads((evidence / "manifest.json").read_bytes())
    require(manifest.get("schema_version") == 1
            and manifest.get("physical_source_head") == SOURCE_HEAD
            and manifest.get("original_success_run_id") == 40710171622
            and manifest.get("original_failed_run_id") == 40709934406
            and manifest.get("train_authorized") is False
            and manifest.get("comparative_training_performed") is False
            and manifest.get("heldout_opened") is False
            and manifest.get("final_72_admission") is False,
            "Original evidence provenance/admission changed")
    archives = manifest.get("archives", {})
    require(set(archives) == set(EXPECTED_ZIPS),
            "Original positive/negative archive set altered")
    members: dict[str, dict[str, bytes]] = {}
    for key, fixed_sha in EXPECTED_ZIPS.items():
        spec = archives[key]
        require(spec.get("sha256") == fixed_sha and
                spec.get("file", "").endswith(".zip")
                and Path(spec["file"]).name == spec["file"],
                "Artifact manifest or file path not immutable: " + key)
        raw = (evidence / spec["file"]).read_bytes()
        require(sha256(raw) == fixed_sha and len(raw) == spec["bytes"],
                "Original Actions ZIP bytes drift: " + key)
        with zipfile.ZipFile(evidence / spec["file"]) as z:
            require(len(z.namelist()) == len(set(z.namelist()))
                    and set(z.namelist()) == set(spec["members"]),
                    "Original ZIP members altered: " + key)
            pieces = {}
            for name, item in spec["members"].items():
                require(Path(name).name == name, "ZIP member path unsafe")
                piece = z.read(name)
                require(len(piece) == item["bytes"]
                        and sha256(piece) == item["sha256"],
                        "Original ZIP member SHA mismatch: " + key + "/" + name)
                pieces[name] = piece
            members[key] = pieces
    require(archives["negative_source"]["run_id"] == 40709934406
            and archives["positive_source"]["run_id"] == 40710171622,
            "Original first failure and final success conflated")
    negatives = members["negative_source"]["source-tests.log"].decode("utf-8")
    require("FAILED (failures=2)" in negatives
            and "test_four_lane_protocol_and_no_comparative_runner" in negatives
            and "test_supervised_loss_requires_nonzero_finite_differentiable_scalar"
                in negatives, "First source test failures falsely reclassified")
    for filename in ("source-tests.log", "lease-tests.log"):
        log = members["positive_source"][filename].decode("utf-8")
        require("Ran 10 tests" in log and "OK" in log and "FAILED" not in log,
                "Positive source job did not pass ten checks: " + filename)
    original_logs = manifest.get("complete_actions_job_logs", {})
    require(set(original_logs) == set(archives), "Full original job log set missing")
    for key, spec in original_logs.items():
        require(spec.get("job_id") == archives[key].get("job_id"),
                "Job log linked to wrong original job: " + key)
        name = spec.get("file", "")
        require(name.endswith(".log") and Path(name).name == name,
                "Original job log name invalid")
        log = (evidence / name).read_bytes()
        require(git_blob_sha(log) == spec["blob_sha"],
                "Full decoded GitHub job log unexpectedly changed: " + key)
        require(str(spec["job_id"]).isdigit() and len(log) > 1000,
                "Full original CI job transcript missing: " + key)
    report = {"experiment": "NCR-001", "source_head": SOURCE_HEAD,
              "original_success_run_id": 40710171622,
              "original_failed_run_id": 40709934406,
              "source_positive": True, "source_negative_preserved": True,
              "original_archive_integrity": True, "lanes": {},
              "all_four_model_72_pass": False, "training_authorized": False,
              "scientific_independent_review_complete": False}
    for lane in ("sprocket_chat", "sprocket_instruct"):
        require(archives[lane]["run_id"] == 40710171622,
                "Sprocket physical lane from another source job")
        physical = json.loads(members[lane]["receipt.json"])
        report["lanes"][lane] = audit_receipt(
            physical, lane=lane, artifact=archives[lane], root=root)
        host_log = members[lane]["host.log"].decode("utf-8")
        require("PASS_DIAGNOSTIC_NCA_LEARNING_GATES__FINAL_72_BLOCKED"
                in host_log and '"diagnostic_optimizer_steps": 1' in host_log
                and '"train_authorized": false' in host_log,
                "Original physical host log does not corroborate receipt")
    chat = json.loads(members["sprocket_chat"]["receipt.json"])
    instruct = json.loads(members["sprocket_instruct"]["receipt.json"])
    require(chat["source_identity"]["weight_sha256"] !=
            instruct["source_identity"]["weight_sha256"],
            "Wrong single-checkpoint two-lane substitution")
    require(chat["synthetic_sample"] == instruct["synthetic_sample"],
            "Same synthetic diagnostic fixture not used in both frozen hosts")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
