#!/usr/bin/env python3
"""Verify and publish one immutable Smol DEVELOPMENT evidence packet.

No model is instantiated here. Checkpoint verification reads CPU tensors with
``weights_only=True`` and safetensors; native verification reduces actual saved
application receipts. Rejected model proposals remain publishable research.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

import contracts
from contracts import common

LANE = Path(__file__).resolve().parent
PAINTER = LANE.parents[1]
REPOSITORY = "techrote/cybersand"
CORPUS_RUN = 43506373969
CORPUS_ARTIFACT = "agent-painter-native-" + contracts.FOUNDATION_SOURCE
ARMS = ("baseline", "candidate")
PUBLICATION_FILES = {
    "experiments/agent_painter/lanes/smol/package.py",
    "experiments/agent_painter/lanes/smol/tests/test_package.py",
    "experiments/agent_painter/lanes/smol/publication.py",
    "experiments/agent_painter/lanes/smol/publication-request.json",
    "experiments/agent_painter/lanes/smol/README.md",
    ".github/workflows/agent-painter-smol-publication.yml",
    ".github/workflows/agent-painter-smol-native.yml",
}
PACKAGING_STEP = "Verify checkpoint bytes and package transparent research evidence"
NATIVE_STEP = "Rebuild unchanged native authority and verify actual raw model drawings"
NATIVE_JOB = "Current-source native replay and immutable evidence"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return common.read_json(path)


def sha(path):
    return common.sha256_file(path)


def verify_environments(directory, source, tree, run_id, attempt, request):
    """Bind both matrix jobs to this run and distinct ordinary hosted VMs."""
    receipts, downloads = {}, {}
    selected = common.profile("smol", require_supported=True)
    for arm in ARMS:
        folder = directory / arm / "environment"
        receipt = read(folder / "run-receipt.json")
        expected = {
            "schema_version": 1, "lane": "smol", "arm": arm,
            "repository": REPOSITORY, "source_sha": source, "source_tree": tree,
            "run_id": run_id, "run_attempt": attempt,
            "runner_environment": "github-hosted", "os": "Linux", "arch": "X64",
            "corpus": {"source_sha": contracts.FOUNDATION_SOURCE, "run_id": CORPUS_RUN,
                       "artifact": CORPUS_ARTIFACT},
            "model": {"model_id": contracts.MODEL_ID, "revision": contracts.MODEL_REVISION},
            "request": request,
        }
        require(all(receipt.get(key) == value for key, value in expected.items()),
                arm + ": environment/source/run/input custody mismatch")
        require(all(isinstance(receipt.get(key), str) and receipt[key]
                    for key in ("job_id", "runner_name", "hostname")),
                arm + ": missing VM identity")
        for name in ("cpu.txt", "pip-freeze.txt", "requirements.sha256"):
            require((folder / name).is_file() and (folder / name).stat().st_size > 0,
                    arm + ": missing runner installation evidence: " + name)
        lock = (folder / "requirements.sha256").read_text().split()
        require(bool(lock) and lock[0] == sha(PAINTER / "training/requirements.lock"),
                arm + ": installed requirement lock differs from source")
        download = read(folder / "model-download.json")
        require(download.get("model_id") == selected["model_id"]
                and download.get("revision") == selected["revision"]
                and download.get("trust_remote_code") is False,
                arm + ": model download pin mismatch")
        files = download.get("files", {})
        require(all(files.get(name, {}).get("sha256") == digest
                    for name, digest in selected["required_hashes"].items()),
                arm + ": model download bytes differ from reviewed pin")
        require(files and all(common.SHA256.fullmatch(str(item.get("sha256", "")))
                              and type(item.get("bytes")) is int and item["bytes"] > 0
                              for item in files.values()),
                arm + ": incomplete model download inventory")
        receipts[arm], downloads[arm] = receipt, download
    # Hosted VM image hostnames can repeat. The API registration proof below,
    # not the guest hostname or shared matrix key, establishes separate jobs.
    require(receipts["baseline"]["runner_name"] != receipts["candidate"]["runner_name"],
            "baseline and candidate must use separate runner registrations")
    require(downloads["baseline"]["files"] == downloads["candidate"]["files"],
            "baseline and candidate model bytes differ")
    return receipts, downloads


def verify_job_assignments(record, environments, source, run_id, attempt, *, recovery):
    """Bind immutable model receipts to saved authoritative Jobs API records."""
    expected = {"schema_version": 1, "repository": REPOSITORY, "source_sha": source,
                "run_id": run_id, "run_attempt": attempt}
    require(all(record.get(key) == value for key, value in expected.items()),
            "job assignment source/run/attempt mismatch")
    base = f"/repos/{REPOSITORY}/actions/runs/{run_id}"
    require(record.get("api_path") in (base + "/jobs", base + f"/attempts/{attempt}/jobs"),
            "job assignment API route differs from producer")
    response = record.get("response", {})
    jobs = response.get("jobs", [])
    require(isinstance(jobs, list) and jobs and response.get("total_count") == len(jobs)
            and len({row.get("id") for row in jobs}) == len(jobs), "incomplete Jobs API response")
    selected = {}
    for arm in (*ARMS, "native"):
        name = arm + " on independent CPU VM" if arm in ARMS else NATIVE_JOB
        matches = [row for row in jobs if row.get("name") == name]
        require(len(matches) == 1, "missing/duplicate authoritative job assignment: " + arm)
        job = matches[0]
        require(job.get("run_id") == run_id and job.get("run_attempt") == attempt
                and job.get("head_sha") == source and type(job.get("id")) is int and job["id"] > 0
                and type(job.get("runner_id")) is int and job["runner_id"] > 0
                and job.get("runner_group_name") == "GitHub Actions"
                and "ubuntu-24.04" in job.get("labels", []), arm + ": wrong producer job/runner assignment")
        if arm in ARMS:
            require(job.get("status") == "completed" and job.get("conclusion") == "success"
                    and job.get("runner_name") == environments[arm]["runner_name"],
                    arm + ": successful hosted assignment does not match environment receipt")
        else:
            steps = job.get("steps", [])
            native_steps = [row for row in steps if row.get("name") == NATIVE_STEP]
            require(len(native_steps) == 1 and native_steps[0].get("status") == "completed"
                    and native_steps[0].get("conclusion") == "success", "producer native execution/build step failed")
            failures = [row for row in steps if row.get("conclusion") in ("failure", "cancelled", "timed_out")]
            require(all(row.get("name") == PACKAGING_STEP and row.get("conclusion") == "failure" for row in failures),
                    "producer failed outside the preserved publication step")
            if recovery:
                require(job.get("status") == "completed" and job.get("conclusion") in ("success", "failure")
                        and (job.get("conclusion") != "failure" or bool(failures)),
                        "recovery requires a completed producer native job with an explained publication failure")
            else:
                require(job.get("status") in ("in_progress", "completed")
                        and job.get("conclusion") in (None, "success"), "direct publication native job is not active/successful")
        selected[arm] = {key: job[key] for key in ("id", "run_id", "run_attempt", "head_sha", "name",
                                                  "runner_id", "runner_name", "runner_group_name", "status", "conclusion")}
    require(len({selected[arm]["id"] for arm in ARMS}) == 2
            and len({selected[arm]["runner_id"] for arm in ARMS}) == 2,
            "baseline and candidate share a job or runner registration")
    return selected


def workflow_without_packaging(text):
    marker = "      - name: " + PACKAGING_STEP
    require(text.count(marker) == 1, "publication workflow packaging step is ambiguous")
    start = text.index(marker)
    next_step = re.search(r"(?m)^      - (?:name|uses):", text[start + len(marker):])
    require(next_step is not None, "publication workflow packaging step has no boundary")
    end = start + len(marker) + next_step.start()
    return text[:start] + text[end:]


def validate_publication_delta(producer_files, publisher_files, workflow_before=None, workflow_after=None):
    """The execution source cannot drift during an independently pinned repair."""
    changed = sorted(name for name in set(producer_files) | set(publisher_files)
                     if producer_files.get(name) != publisher_files.get(name))
    require(set(changed) <= PUBLICATION_FILES, "publisher changed non-publication execution source: "
            + ", ".join(sorted(set(changed) - PUBLICATION_FILES)))
    workflow = ".github/workflows/agent-painter-smol-native.yml"
    if workflow in changed:
        require(workflow_before is not None and workflow_after is not None
                and workflow_without_packaging(workflow_before) == workflow_without_packaging(workflow_after),
                "publisher changed workflow execution outside the packaging step")
    return changed


def git_tree(root):
    output = subprocess.check_output(["git", "ls-tree", "-rz", "HEAD"], cwd=root)
    return {entry.split(b"\t", 1)[1].decode(): entry.split(b"\t", 1)[0].decode()
            for entry in output.split(b"\0") if entry}


def verify_publication_sources(producer_root, producer_source, publisher_root, publisher_source):
    from native.runtime import source_identity
    producer_root, publisher_root = Path(producer_root).resolve(), Path(publisher_root).resolve()
    producer = source_identity(producer_root, producer_source)
    publisher = source_identity(publisher_root, publisher_source)
    allowed_binary = "godot/addons/cybersand_native/bin/libcybersand_native.linux.x86_64.so"
    for root in {producer_root, publisher_root}:
        dirty = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=root, text=True).splitlines()
        require(set(dirty) <= {allowed_binary}, "publication checkout has uncommitted tracked changes")
    if producer_source != publisher_source:
        subprocess.run(["git", "merge-base", "--is-ancestor", producer_source, publisher_source],
                       cwd=publisher_root, check=True)
    workflow = ".github/workflows/agent-painter-smol-native.yml"
    changed = validate_publication_delta(git_tree(producer_root), git_tree(publisher_root),
                                        (producer_root / workflow).read_text(), (publisher_root / workflow).read_text())
    return producer, publisher, changed


def verify_artifact_custody(directory, record, source, run_id, attempt, *, recovery):
    """Re-read downloaded ZIP bytes and match every extracted producer member."""
    expected = {"schema_version": 1, "repository": REPOSITORY, "source_sha": source,
                "run_id": run_id, "run_attempt": attempt}
    require(all(record.get(key) == value for key, value in expected.items()),
            "artifact custody source/run/attempt mismatch")
    entries = record.get("artifacts", [])
    components = [entry.get("component") for entry in entries]
    required = {"baseline", "candidate", "native", "build"} if recovery else {"baseline", "candidate"}
    require(len(components) == len(set(components)) and required <= set(components)
            and set(components) <= {"baseline", "candidate", "native", "build"},
            "producer artifact component inventory incomplete")
    ids, result = set(), []
    for entry in entries:
        component, artifact = entry["component"], entry["artifact"]
        identity = artifact.get("workflow_run", {})
        require(type(artifact.get("id")) is int and artifact["id"] > 0 and artifact["id"] not in ids
                and artifact.get("name") == f"smol-{component}-{source}-{run_id}-{attempt}"
                and identity.get("id") == run_id and identity.get("head_sha") == source
                and artifact.get("expired") is False, "producer artifact API identity mismatch: " + component)
        ids.add(artifact["id"])
        archive_file = common.relative_file(directory, entry["archive_path"])
        measured = sha(archive_file)
        require(common.SHA256.fullmatch(str(entry.get("archive_sha256", "")))
                and measured == entry["archive_sha256"]
                and archive_file.stat().st_size == entry.get("archive_bytes") == artifact.get("size_in_bytes"),
                "producer ZIP hash/size mismatch: " + component)
        require(artifact.get("digest") in (None, "sha256:" + measured),
                "producer artifact API digest differs from downloaded ZIP")
        prefix = entry.get("member_prefix", "")
        require(prefix in ("", "evidence/") and (not prefix or component == "build"),
                "unsupported producer ZIP prefix")
        target = directory / {"baseline": "baseline", "candidate": "candidate", "native": "smol-native", "build": ""}[component]
        members = []
        with zipfile.ZipFile(archive_file) as stream:
            require(stream.testzip() is None, "producer ZIP CRC failure")
            for item in stream.infolist():
                if item.is_dir():
                    continue
                require(item.filename.startswith(prefix) and (item.external_attr >> 16) & 0o170000 in (0, 0o100000),
                        "unsafe producer ZIP member")
                name = item.filename[len(prefix):]
                require(name and not Path(name).is_absolute() and ".." not in Path(name).parts
                        and "\\" not in name and name not in members, "unsafe/duplicate producer ZIP member")
                path = common.relative_file(target, name)
                require(path.stat().st_size == item.file_size
                        and sha(path) == hashlib.sha256(stream.read(item)).hexdigest(),
                        "extracted producer bytes differ from original ZIP: " + name)
                members.append(name)
        require(bool(members), "empty producer artifact ZIP")
        result.append({"component": component, "artifact_id": artifact["id"], "artifact_name": artifact["name"],
                       "archive_sha256": measured, "archive_bytes": entry["archive_bytes"],
                       "member_count": len(members), "api_digest_present": artifact.get("digest") is not None})
    return result


def verify_evaluation(directory, arm, source, run_id, attempt, download):
    from lane_adapters import adapter_identity
    report = read(directory / "evaluation.json")
    require(report.get("schema_version") == 1 and report.get("kind") == "smol-development-first-pass"
            and report.get("arm") == arm and report.get("source_sha") == source,
            arm + ": evaluation identity mismatch")
    require(report.get("final_targets_opened") is False
            and report.get("native_execution") is False and report.get("visual_quality_claim") is False,
            arm + ": evaluation authority/custody claim mismatch")
    runner = report.get("runner", {})
    require(str(runner.get("github_run_id")) == str(run_id)
            and str(runner.get("github_run_attempt")) == str(attempt),
            arm + ": evaluation run/attempt mismatch")
    require(report.get("model") == common.profile("smol", require_supported=True)
            and report.get("model_id") == contracts.MODEL_ID
            and report.get("model_revision") == contracts.MODEL_REVISION
            and report.get("compiler_sha256") == contracts.COMPILER_SHA256,
            arm + ": evaluation model/compiler mismatch")
    corpus = report.get("corpus", {})
    require(corpus.get("manifest_sha256") == contracts.MANIFEST_SHA256
            and corpus.get("development_sha256") == contracts.DEVELOPMENT_SHA256
            and corpus.get("train_sha256") == contracts.TRAIN_SHA256
            and corpus.get("native_source_sha") == contracts.FOUNDATION_SOURCE,
            arm + ": frozen corpus custody mismatch")
    comparison = report["comparison_identity"]
    require(common.digest_json(comparison) == report.get("comparison_identity_sha256")
            and comparison.get("source_sha") == source and comparison.get("model") == report["model"]
            and comparison.get("model_files") == download["files"]
            and comparison.get("adapter_contract") == adapter_identity(report["model"])
            and comparison.get("corpus") == corpus
            and comparison.get("evaluator_sha256") == sha(LANE / "evaluate_native.py")
            and comparison.get("contracts_sha256") == sha(LANE / "contracts.py"),
            arm + ": comparison source/config custody mismatch")
    require(common.digest_json({"comparison": comparison, "arm": arm, "checkpoint": report["checkpoint"]})
            == report.get("evaluation_identity_sha256"), arm + ": arm/checkpoint identity mismatch")
    decoding = report["decoding"]
    require(comparison.get("decoding") == decoding and decoding.get("do_sample") is False
            and decoding.get("num_beams") == 1 and decoding.get("max_new_tokens") == 1024
            and decoding.get("max_seconds") == 1200
            and all(decoding.get(key) is False for key in ("repair", "retry", "fallback")),
            arm + ": first-pass decoding protocol changed")
    cases, generations, counts = report["cases"], report["generations"], report["counts"]
    require(len(cases) == len({row["id"] for row in cases}) == 12
            and len({row["group_id"] for row in cases}) == 6
            and len({row["prompt"] for row in cases}) == 6
            and len(generations) == len({row["id"] for row in generations}) == 6,
            arm + ": six-group/twelve-view denominator changed")
    expected_counts = {
        "case_views": 12, "structural_groups": 6, "distinct_prompts": 6,
        "model_generations_attempted": sum(row["attempted"] for row in generations),
        "attempted_case_views": sum(row["generation_attempted"] for row in cases),
        "accepted_case_views": sum(row["accepted"] for row in cases),
        "accepted_generations": len({row["generation_id"] for row in cases if row["accepted"]}),
        "family_matching_accepted_case_views": sum(row["accepted"] and row["requested_family_matches"] for row in cases),
    }
    require(counts == expected_counts, arm + ": evaluation counts differ from retained rows")
    by_generation = {row["id"]: row for row in generations}
    for case in cases:
        require(case.get("split") == "DEVELOPMENT" and case.get("native_execution") is False,
                arm + ": non-DEVELOPMENT or invented native evaluation row")
        require(set(case["paths"]) == set(case["sha256"]), arm + ": incomplete case inventory")
        for name, path in case["paths"].items():
            common.verify_file(directory, path, case["sha256"][name])
        receipt = read(common.relative_file(directory, case["paths"]["receipt"]))
        expected_receipt = {**case, "sha256": {key: value for key, value in case["sha256"].items() if key != "receipt"}}
        require(receipt == expected_receipt, arm + ": case receipt relabeled")
        raw = read(common.relative_file(directory, case["paths"]["raw_output"]))
        generation = by_generation[case["generation_id"]]
        require(raw == case["verdict"]
                and common.relative_file(directory, case["paths"]["completion"]).read_text() == raw["raw_text"]
                and common.digest_json(raw["token_ids"]) == generation["token_ids_sha256"]
                and case["generation_attempted"] == generation["attempted"]
                and case["id"] in generation["request_ids"]
                and case["prompt"] == generation["prompt"], arm + ": raw generation custody mismatch")
    common.verify_file(directory, "accepted-candidates.json", report["accepted_candidates_sha256"])
    accepted = read(directory / "accepted-candidates.json")
    require(accepted["source_sha"] == source and accepted["arm"] == arm
            and accepted["evaluation_identity_sha256"] == report["evaluation_identity_sha256"]
            and {row["id"] for row in accepted["examples"]} == {row["id"] for row in cases if row["accepted"]}
            and len(accepted["examples"]) == counts["accepted_case_views"],
            arm + ": accepted subset differs from full denominator")
    return report


def read_checkpoint(directory):
    """Read saved tensors and primitive optimizer state; never execute a model."""
    import torch
    from safetensors.torch import load_file
    receipt = common.verify_checkpoint(directory)
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=True)
    for key in ("global_step", "sample_cursor", "config", "frozen_base_sha256", "history"):
        require(state.get(key) == receipt.get(key), "checkpoint state/receipt differ: " + key)
    step = receipt["global_step"]
    require(type(step) is int and 1 <= step <= 120 and state["sample_cursor"] == step,
            "checkpoint optimizer/sampler step is outside this first iteration")
    history = state["history"]
    require(len(history) == step and [row["global_step"] for row in history] == list(range(1, step + 1))
            and [row["sample_cursor"] for row in history] == list(range(1, step + 1)),
            "checkpoint optimizer history is incomplete")
    order = state["config"]["sample_order"]
    require(len(order) == len(set(order)) == 12
            and [row["sample_id"] for row in history] == [order[index % len(order)] for index in range(step)],
            "checkpoint sampler history differs from frozen order")
    adapters = load_file(str(directory / "adapter/adapter_model.safetensors"), device="cpu")
    require(len(adapters) == 420 and sum(value.numel() for value in adapters.values()) == 2442240
            and all("lora_" in name and value.dtype == torch.float32 and torch.isfinite(value).all().item()
                    for name, value in adapters.items()), "checkpoint adapter tensor readback invalid")
    optimizer = state["optimizer"]
    require(len(optimizer["state"]) == len(adapters) and len(optimizer["param_groups"]) == 1,
            "checkpoint optimizer inventory incomplete")
    moments_nonzero = False
    for item in optimizer["state"].values():
        require(int(item["step"].item()) == step, "optimizer parameter step differs")
        for name in ("exp_avg", "exp_avg_sq"):
            value = item[name]
            require(value.is_floating_point() and torch.isfinite(value).all().item(),
                    "optimizer moment is nonfinite")
            moments_nonzero |= bool(torch.count_nonzero(value).item())
    require(moments_nonzero and state["scheduler"]["last_epoch"] == step
            and isinstance(state["torch_rng"], torch.Tensor) and state["torch_rng"].numel() > 0
            and isinstance(state["python_rng"], tuple), "optimizer/scheduler/RNG readback incomplete")
    return receipt


def verify_training(directory, evaluation, source, run_id, attempt, request):
    summary, wrapper = read(directory / "summary.json"), read(directory / "lane-receipt.json")
    step = summary.get("global_step")
    require(type(step) is int and 1 <= step <= request["candidate_steps"]
            and summary.get("actual_training") is True and summary.get("resume") is None
            and summary.get("optimizer_steps_this_job") == step and summary.get("sample_cursor") == step
            and summary.get("checkpoint_readback_exact") is True and summary.get("adapter_changed") is True
            and summary.get("frozen_base_unchanged") is True
            and summary.get("adapter_before_sha256") != summary.get("adapter_after_sha256"),
            "candidate lacks a bounded real optimizer/checkpoint receipt")
    expected_status = "completed" if step == request["candidate_steps"] else "stopped-checkpoint-saved"
    require(summary.get("status") == expected_status and wrapper.get("status") == expected_status
            and wrapper.get("exit_code") == (0 if expected_status == "completed" else 2)
            and wrapper.get("source_sha") == source and wrapper.get("actual_global_step") == step
            and wrapper.get("target_global_steps") == request["candidate_steps"]
            and wrapper.get("checkpoint_every") == request["checkpoint_every"]
            and wrapper.get("max_seconds") == request["training_seconds"]
            and wrapper.get("summary_sha256") == sha(directory / "summary.json")
            and wrapper.get("wrapper_sha256") == sha(LANE / "train.py")
            and wrapper.get("contracts_sha256") == sha(LANE / "contracts.py")
            and wrapper.get("failure") is None and wrapper.get("final_targets_opened") is False,
            "training wrapper/summary custody mismatch")
    config, comparison = summary["config"], evaluation["comparison_identity"]
    required_config = {
        "source_sha": source, "profile": "smol", "model": evaluation["model"],
        "native_source_sha": contracts.FOUNDATION_SOURCE, "corpus_manifest_sha256": contracts.MANIFEST_SHA256,
        "train_sha256": contracts.TRAIN_SHA256, "compiler_sha256": contracts.COMPILER_SHA256,
        "trainer_sha256": sha(PAINTER / "training/train.py"), "contract_sha256": sha(PAINTER / "training/common.py"),
        "model_files": comparison["model_files"], "numerics": comparison["numerics"],
        "adapter_contract": comparison["adapter_contract"],
        "system": comparison["system"], "system_sha256": comparison["system_sha256"],
        "chat_template_sha256": comparison["chat_template_sha256"],
        "device": "cpu", "dtype": "float32", "threads": 1, "seed": 568, "max_length": 2048,
        "rank": 8, "alpha": 16, "dropout": 0.05, "learning_rate": 0.0005,
        "native_execution_by_trainer": False, "final_targets_opened": False,
    }
    require(all(config.get(key) == value for key, value in required_config.items()),
            "training configuration differs from frozen lane/evaluation")
    require(str(summary["runner"].get("github_run_id")) == str(run_id)
            and str(summary["runner"].get("github_run_attempt")) == str(attempt), "training run/attempt differs")
    checkpoints = sorted(directory.glob("checkpoint-*"))
    require(bool(checkpoints) and all(path.is_dir() and not path.is_symlink() for path in checkpoints),
            "checkpoint directories missing or unsafe")
    receipts = {path.name: read_checkpoint(path) for path in checkpoints}
    require(all(receipt["config"] == config and receipt["global_step"] <= step for receipt in receipts.values()),
            "published checkpoint configuration differs")
    latest = common.relative_file(directory, summary["checkpoint"] + "/inventory.json").parent
    receipt = receipts[latest.name]
    require(receipt["global_step"] == step
            and receipt["frozen_base_sha256"] == summary["frozen_base_sha256"]
            and sha(latest / "inventory.json") == summary["checkpoint_inventory_sha256"]
            and sum(path.stat().st_size for path in latest.rglob("*") if path.is_file()) == summary["checkpoint_bytes"],
            "latest checkpoint differs from training summary")
    expected_checkpoint = {"global_step": step, "source_sha": source,
                           "inventory_sha256": sha(latest / "inventory.json"),
                           "adapter_sha256": sha(latest / "adapter/adapter_model.safetensors"),
                           "receipt_sha256": sha(latest / "receipt.json")}
    require(evaluation["checkpoint"] == expected_checkpoint, "candidate evaluation used a different checkpoint")
    return {"actual_optimizer_steps": step, "target_optimizer_steps": request["candidate_steps"],
            "status": expected_status, "checkpoint": latest.name,
            "checkpoint_inventory_sha256": expected_checkpoint["inventory_sha256"],
            "serialized_checkpoints_read": len(receipts), "checkpoint_tensor_readback_verified": True}


def inventory_files(root):
    """Publication never includes FINAL targets, symlinks or transient caches."""
    require(root.is_dir() and not root.is_symlink(), "missing/unsafe evidence component")
    result = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "symlink in evidence: " + str(path))
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        require("final.jsonl" not in {part.lower() for part in relative.parts}, "FINAL targets in publication")
        if any(part in {"__pycache__", ".godot", "userdata", "profile"} for part in relative.parts):
            continue
        result.append((relative.as_posix(), path))
    require(bool(result), "empty evidence component")
    return result


def archive(path, entries):
    require(not path.exists(), "refusing to overwrite published archive")
    require(len({name for name, _ in entries}) == len(entries), "duplicate archive member")
    with zipfile.ZipFile(path, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for name, file in entries:
            require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                    "unsafe archive member")
            require(not file.is_symlink() and file.is_file(), "unsafe archive source")
            require("final.jsonl" not in {part.lower() for part in Path(name).parts}, "FINAL targets in publication")
            stream.write(file, name)
    with zipfile.ZipFile(path) as stream:
        require(stream.testzip() is None, "archive readback CRC failed")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha(path), "files": len(entries)}


def verify_regressions(directory):
    expected = {"art_batch", "stored_material", "deferred_manual", "undo_1", "undo_4",
                "combined_author", "combined_reopen"}
    results = read(directory / "results.json")
    require(isinstance(results, list) and len(results) == 7
            and {row.get("label") for row in results} == expected,
            "retained native regression suite inventory differs")
    checks = 0
    for row in results:
        completion = row.get("completion", {})
        require(row.get("status") == "passed" and row.get("exit_code") == 0
                and type(completion.get("checks")) is int and completion["checks"] > 0
                and completion.get("failures") in (0, []),
                "retained native regression failed: " + row["label"])
        require((directory / (row["label"] + ".log")).is_file(),
                "retained regression execution log missing: " + row["label"])
        checks += completion["checks"]
    return {"suites": 7, "checks": checks, "failures": 0,
            "results_sha256": sha(directory / "results.json")}


def package(directory, output, source, *, producer_root=None, producer_run_id=None,
            producer_run_attempt=None, job_assignments=None, artifact_custody=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    require(not output.exists(), "immutable release output already exists")
    require(os.environ.get("GITHUB_ACTIONS") == "true"
            and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and os.environ.get("RUNNER_OS") == "Linux", "publication requires a Linux GitHub-hosted runner")
    from native_replay import verify_packet
    from native.runtime import ROOT
    require(os.environ.get("GITHUB_REPOSITORY") == REPOSITORY, "publication repository environment mismatch")
    publisher_source = os.environ["GITHUB_SHA"]
    publisher_run, publisher_attempt = int(os.environ["GITHUB_RUN_ID"]), int(os.environ["GITHUB_RUN_ATTEMPT"])
    run_id = publisher_run if producer_run_id is None else producer_run_id
    attempt = publisher_attempt if producer_run_attempt is None else producer_run_attempt
    require(all(type(value) is int and value > 0 for value in (run_id, attempt, publisher_run, publisher_attempt)),
            "positive producer/publisher run/attempt required")
    recovery = (source, run_id, attempt) != (publisher_source, publisher_run, publisher_attempt)
    require(not recovery or (producer_root is not None and producer_run_id is not None
                             and producer_run_attempt is not None), "recovery needs an explicit original producer checkout/run/attempt")
    producer_root = Path(producer_root or ROOT).resolve()
    source_record, publisher_record, changed_files = verify_publication_sources(producer_root, source, ROOT, publisher_source)
    tree = source_record["source_tree"]
    producer = {"source_sha": source, "source_tree": tree, "run_id": run_id, "run_attempt": attempt,
                "source_inputs_sha256": source_record["source_inputs_sha256"]}
    publisher = {"source_sha": publisher_source, "source_tree": publisher_record["source_tree"],
                 "run_id": publisher_run, "run_attempt": publisher_attempt,
                 "source_inputs_sha256": publisher_record["source_inputs_sha256"],
                 "job_id": os.environ.get("GITHUB_JOB"), "runner_name": os.environ.get("RUNNER_NAME"),
                 "verifier_path": "experiments/agent_painter/lanes/smol/package.py", "verifier_sha256": sha(__file__),
                 "publication_only_changes": changed_files,
                 "publication_files_sha256": {name: sha(ROOT / name) for name in changed_files if (ROOT / name).is_file()},
                 "producer_execution_reused": recovery}
    request = read(producer_root / "experiments/agent_painter/lanes/smol/run-request.json")
    require(request == read(directory / "build/request.json") and request["candidate_steps"] == 120
            and request["checkpoint_every"] == 40 and request["training_seconds"] == 1200
            and request["evaluation_seconds"] == 1200 and request["generation_tokens"] == 1024
            and request["final_access"] is False, "published request differs from frozen bounded iteration")
    environments, downloads = verify_environments(directory, source, tree, run_id, attempt, request)
    publication = directory / "publication"
    jobs_path = Path(job_assignments or publication / "job-assignments.json").resolve()
    artifacts_path = Path(artifact_custody or publication / "producer-artifacts.json").resolve()
    for path in (jobs_path, artifacts_path):
        require(path.is_relative_to(publication.resolve()) and path.is_file() and not path.is_symlink(),
                "publication custody records must be saved under evidence/publication")
    publisher_receipt = read(publication / "publisher-receipt.json")
    expected_publisher = {"schema_version": 1, "repository": REPOSITORY,
                          "operation": "recover" if recovery else "capture",
                          "source_sha": publisher_source, "source_tree": publisher_record["source_tree"],
                          "run_id": publisher_run, "run_attempt": publisher_attempt,
                          "job_id": os.environ.get("GITHUB_JOB"), "runner_name": os.environ.get("RUNNER_NAME"),
                          "runner_environment": "github-hosted", "os": "Linux", "arch": "X64"}
    require(all(publisher_receipt.get(key) == value for key, value in expected_publisher.items()),
            "publication downloader identity differs from actual publisher")
    assignments = verify_job_assignments(read(jobs_path), environments, source, run_id, attempt, recovery=recovery)
    artifacts = verify_artifact_custody(directory, read(artifacts_path), source, run_id, attempt, recovery=recovery)
    evaluations = {arm: verify_evaluation(directory / arm / "evaluation", arm, source, run_id, attempt, downloads[arm])
                   for arm in ARMS}
    require(evaluations["baseline"]["checkpoint"] is None
            and evaluations["baseline"]["comparison_identity"] == evaluations["candidate"]["comparison_identity"],
            "baseline is not untuned or arm evaluation protocols differ")
    training = verify_training(directory / "candidate/training", evaluations["candidate"], source, run_id, attempt, request)
    proof = verify_packet(directory / "smol-native", source, details=True)
    native = {**proof["summary"], "runtime_sha256": proof["summary"]["runtime"]["native_sha256"]}
    require(native.get("native_passed") is True and native.get("native_global_gates")
            and all(value is True for value in native["native_global_gates"].values()),
            "publication requires completed source-qualified native gates")
    from native_gallery import verify_gallery
    gallery = verify_gallery(directory / "smol-native", directory / "smol-native/gallery", proof)
    require(gallery["case_views"] == 24 and gallery["native_images"] == native["counts"]["native_verified_views"],
            "gallery image/view counts differ from actual native results")
    for arm in ARMS:
        require(native["evaluation_sha256"][arm] == sha(directory / arm / "evaluation/evaluation.json"),
                arm + ": native replay consumed a different evaluation")
    build = read(directory / "native-build.json")
    require(all(build.get(key) == value for key, value in source_record.items())
            and build.get("source_tree") == tree
            and build.get("kind") == "agent-painter-source-build"
            and build.get("native_sha256") == native["runtime_sha256"], "native build/source publication mismatch")
    binary = directory / "build/runtime/libcybersand_native.linux.x86_64.so"
    require(sha(binary) == build["native_sha256"] and binary.stat().st_size == build["native_size"],
            "published native runtime bytes differ")
    require((directory / "build/source-commit.txt").read_text().strip() == source
            and (directory / "build/source-tree.txt").read_text().strip() == tree,
            "build log source identities differ")
    for name in ("godot-cpp.log", "native.log", "abi-floor.log", "import.log"):
        require((directory / "build" / name).is_file(), "missing source build log: " + name)
    regressions = verify_regressions(directory / "regressions")
    verification = publication / "verification.json"
    require(not verification.exists(), "refusing to overwrite a publication verification record")
    common.write_json(verification, {
        "schema_version": 1, "producer": producer, "publisher": publisher,
        "job_assignments_sha256": sha(jobs_path), "producer_artifact_custody_sha256": sha(artifacts_path),
        "verified_job_assignments": assignments, "verified_producer_artifacts": artifacts,
        "training": training, "native_verification_sha256": native["native_verification_sha256"],
        "gallery": gallery, "retained_native_regressions": regressions,
        "hostname_policy": "Guest hostnames may repeat; distinct numeric GitHub job and runner registrations are required.",
        "model_training_or_inference_during_publication": False, "native_execution_during_publication": False,
    })
    components = {arm: inventory_files(directory / arm) for arm in ARMS}
    components["native"] = inventory_files(directory / "smol-native")
    components["build"] = [("build/" + name, path) for name, path in inventory_files(directory / "build")]
    components["build"].append(("native-build.json", directory / "native-build.json"))
    components["build"].extend(("regressions/" + name, path) for name, path in inventory_files(directory / "regressions"))
    components["build"].extend(("publication/" + name, path) for name, path in inventory_files(publication))
    output.mkdir(parents=True)
    assets = [archive(output / f"smol-{name}-{source[:12]}-{run_id}-{attempt}.zip", files)
              for name, files in components.items()]
    standalone = output / "smol-gallery.html"
    standalone.write_bytes((directory / "smol-native/gallery/index.html").read_bytes())
    require(sha(standalone) == gallery["index_sha256"], "standalone gallery differs from verified original")
    assets.append({"file": standalone.name, "bytes": standalone.stat().st_size, "sha256": sha(standalone)})
    manifest = {
        "schema": "agent-painter-smol-release-v2", "repository": REPOSITORY,
        "source_sha": source, "source_tree": tree, "workflow_run_id": run_id, "workflow_run_attempt": attempt,
        "producer": producer, "publisher": publisher,
        "publication_verification_sha256": sha(verification),
        "verified_job_assignments": assignments, "verified_producer_artifacts": artifacts,
        "request": request, "corpus": {"source_sha": contracts.FOUNDATION_SOURCE, "run_id": CORPUS_RUN,
                                      "artifact": CORPUS_ARTIFACT, "manifest_sha256": contracts.MANIFEST_SHA256,
                                      "development_sha256": contracts.DEVELOPMENT_SHA256},
        "model": {"model_id": contracts.MODEL_ID, "revision": contracts.MODEL_REVISION},
        "independent_hosted_model_vms": {arm: {key: environments[arm][key] for key in ("hostname", "runner_name", "job_id")}
                                         for arm in ARMS},
        "training": training, "development_counts": {arm: evaluations[arm]["counts"] for arm in ARMS},
        "native": native, "gallery": gallery, "retained_native_regressions": regressions,
        "final_targets_opened": False, "artistic_model_qualification": False,
        "draft_hold": True, "assets": assets,
        "claim_limits": ["Twelve host-seed views per arm are six distinct first-pass requests.",
                         "A successful protocol publication can include rejected outputs and zero native model images.",
                         "Native structural checks do not certify requested-layout fulfillment or artistic quality.",
                         "Original training-corpus native admission remains attributed to source633; replay uses this exact source."],
    }
    common.write_json(output / "MANIFEST.json", manifest)
    sums = [{"file": "MANIFEST.json", "sha256": sha(output / "MANIFEST.json")}, *assets]
    (output / "SHA256SUMS.txt").write_text("".join(item["sha256"] + "  " + item["file"] + "\n" for item in sums))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--producer-root", type=Path, help="clean original checkout for publication-only recovery")
    parser.add_argument("--producer-run-id", type=int)
    parser.add_argument("--producer-run-attempt", type=int)
    parser.add_argument("--job-assignments", type=Path, help="saved authoritative Jobs API record")
    parser.add_argument("--artifact-custody", type=Path, help="saved downloaded producer artifact custody")
    args = parser.parse_args()
    result = package(args.directory, args.output, args.source_sha, producer_root=args.producer_root,
                     producer_run_id=args.producer_run_id, producer_run_attempt=args.producer_run_attempt,
                     job_assignments=args.job_assignments, artifact_custody=args.artifact_custody)
    print(json.dumps({"source_sha": result["source_sha"], "training": result["training"],
                      "development_counts": result["development_counts"], "assets": result["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
