#!/usr/bin/env python3
"""Read-only acceptance of the frozen VILLAGE-001F source, inputs and results.

No network, model, tokenizer, optimizer or provider execution is permitted.
Actual phases consume previously downloaded release assets and GitHub metadata.
Temporary extraction/replay stays outside the repository. Source mode never
reads weights and never claims that a diagnostic experiment was executed.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import contextlib
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
SOURCE_COMMIT = "78a72238a8403c4cddb9e8651242a89e17ee21be"
REL = Path("experiments/village-001/diagnostic-20261119")
REPO = "techrote/neurounner"
INPUT_TAG = "village-001f-pilot-inputs-20261119"
FINAL_TAG = "village-001f-mask-sprocket-20261119"
INPUT_TAR = "village001f-complete-inputs.tar"
FINAL_TAR = "village001f-complete-evidence.tar"
FINAL_JSON = {name + ".json" for name in ("summary", "worker-paired-effects", "sprocket-grounding", "judgments")}
CONTRACTS = ("worker-config.json", "worker-eval.json", "sprocket-config.json", "judge-config.json")
F_SCRIPTS = ("village_001f_operator", "village_001f_worker_diagnostic", "village_001f_sprocket_diagnostic", "village_001f_judge")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/nonregular JSON: " + str(path))
    return json.loads(path.read_text())


def canonical(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "missing/nonregular file: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args, raw=False):
    value = subprocess.check_output(["git", "-C", str(repo), *args])
    return value if raw else value.decode().strip()


def ancestor(repo, older, newer):
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", older, newer], check=True)


def inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "missing/nonregular directory: " + str(root))
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "unsafe custody entry: " + str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def forbidden(*args, **kwargs):
    raise RuntimeError("model/tokenizer/provider/network execution is forbidden in this acceptance audit")


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def source_preflight(repo, scratch):
    """Freeze the executable dependency closure, contracts and native plans."""
    ancestor(repo, SOURCE_COMMIT, "HEAD")
    pending = list(F_SCRIPTS)
    seen = set()
    while pending:
        name = pending.pop()
        path = Path("scripts") / (name + ".py")
        if path in seen:
            continue
        original = git(repo, "show", SOURCE_COMMIT + ":" + path.as_posix(), raw=True)
        same((repo / path).read_bytes(), original, "frozen executable source changed: " + str(path))
        seen.add(path)
        for node in ast.walk(ast.parse(original)):
            names = ([item.name.split(".")[0] for item in node.names] if isinstance(node, ast.Import)
                     else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
            pending.extend(module for module in names if (repo / "scripts" / (module + ".py")).is_file())
    fixed = [REL / name for name in (*CONTRACTS, "worker-plan.json")]
    fixed.append(Path(".github/workflows/village-001f-diagnostics.yml"))
    for path in fixed:
        same((repo / path).read_bytes(), git(repo, "show", SOURCE_COMMIT + ":" + path.as_posix(), raw=True),
             "frozen contract/plan/workflow changed: " + str(path))
    sys.path.insert(0, str(repo / "scripts"))
    modules = [importlib.import_module(name) for name in F_SCRIPTS]
    op, worker, sprocket, judge = modules
    for module in modules:
        same(Path(module.__file__).resolve().parent, repo / "scripts", "import resolved outside audited checkout")
    socket.create_connection = forbidden
    socket.socket.connect = forbidden
    socket.socket.connect_ex = forbidden
    for key in ("NVIDIA_API_KEY", "NVKEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(key, None)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for name in ("runtime_modules", "tokenizer_for", "model_for", "infer", "score", "train", "token_audit"):
        setattr(worker, name, forbidden)
    sprocket.verify_runtime = forbidden
    sprocket.LocalHTTP.request = forbidden
    judge.call_api = forbidden
    importlib.import_module("village_001b_generate_member").call_api = forbidden
    op.gh = forbidden
    op.get_bytes = forbidden
    jcfg, wcfg, suite, campaign, pref = judge.load_contract(repo / REL / "judge-config.json")
    plan_dir = scratch / "worker-plan-replay"
    with quiet():
        worker.plan(SimpleNamespace(config=repo / REL / "worker-config.json", out=plan_dir), wcfg, suite, campaign, pref)
    same((plan_dir / "worker-plan.json").read_bytes(), (repo / REL / "worker-plan.json").read_bytes(),
         "frozen worker plan does not reconstruct")
    scfg, splan = sprocket.build_plan(repo / REL / "sprocket-config.json")
    summary = {"pass": True, "source_commit": SOURCE_COMMIT,
        "executable_sources_sha256": {str(path): sha(repo / path) for path in sorted(seen)},
        "contract_sha256": {name: sha(repo / REL / name) for name in CONTRACTS},
        "worker_plan_sha256": sha(repo / REL / "worker-plan.json"),
        "sprocket_plan_sha256": hashlib.sha256(sprocket.canonical(splan)).hexdigest(),
        "planned_training_lanes": 12, "planned_optimizer_steps": 1440,
        "planned_member_training_jobs": 6,
        "planned_member_arm_order": {member: worker.training_arm_order(wcfg, member) for member in worker.PANEL},
        "planned_worker_responses": 300, "planned_sprocket_responses": 36,
        "planned_token_audit_sequences": 720, "planned_forward_batches": 288,
        "provider_logical_limit": 114}
    return SimpleNamespace(op=op, worker=worker, sprocket=sprocket, judge=judge,
        jcfg=jcfg, wcfg=wcfg, suite=suite, campaign=campaign, pref=pref, scfg=scfg, splan=splan), summary


def safe_adapter_extract(archive, destination):
    require(not destination.exists(), "adapter extraction destination already exists")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        require(0 < len(members) <= 1000 and sum(item.size for item in members) <= 256 * 1024 * 1024,
                "adapter TAR exceeds bounded inventory")
        seen = set()
        for item in members:
            name = item.name.rstrip("/")
            path = PurePosixPath(name)
            require(name and not path.is_absolute() and all(p not in ("", ".", "..") for p in name.split("/"))
                    and "\\" not in name and ":" not in name and "\x00" not in name,
                    "unsafe adapter TAR path")
            require(name not in seen and (item.isdir() or item.isfile()), "duplicate/link/special adapter TAR entry")
            seen.add(name)
            require(0 <= item.size <= 128 * 1024 * 1024, "adapter member exceeds bound")
        destination.mkdir(parents=True)
        for item in members:
            target = destination / item.name.rstrip("/")
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = tar.extractfile(item)
                require(stream is not None, "adapter TAR file has no stream")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
    return inventory(destination)


def raw_safetensors_fingerprint(directory):
    """Hash exact contiguous tensor bytes; never import or instantiate a model.

    Safetensors stores a little-endian uint64 JSON-header length followed by
    the header and contiguous tensor data. The frozen F adapters are F32; the
    small explicit dtype table also handles ordinary integer/half tensors.
    Unknown dtypes fail closed. Canonical framing matches worker.json_bytes.
    """
    weights = directory / "adapter_model.safetensors"
    file_size = weights.stat().st_size
    types = {"F64": ("torch.float64", 8), "F32": ("torch.float32", 4),
        "F16": ("torch.float16", 2), "BF16": ("torch.bfloat16", 2),
        "I64": ("torch.int64", 8), "I32": ("torch.int32", 4), "I16": ("torch.int16", 2),
        "I8": ("torch.int8", 1), "U8": ("torch.uint8", 1), "BOOL": ("torch.bool", 1)}
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate safetensors header key")
            result[key] = value
        return result
    with weights.open("rb") as stream:
        raw = stream.read(8)
        require(len(raw) == 8, "truncated safetensors length")
        length = struct.unpack("<Q", raw)[0]
        require(1 <= length <= 16 * 1024 * 1024 and 8 + length <= file_size, "invalid safetensors header length")
        header = json.loads(stream.read(length), object_pairs_hook=unique)
        require(isinstance(header, dict), "safetensors header must be an object")
        if "__metadata__" in header:
            require(isinstance(header["__metadata__"], dict), "invalid safetensors metadata")
        payload_size, intervals, descriptors = file_size - 8 - length, [], {}
        for name, row in header.items():
            if name == "__metadata__":
                continue
            require(isinstance(name, str) and name and isinstance(row, dict), "invalid tensor descriptor")
            require(set(row) == {"dtype", "shape", "data_offsets"} and row["dtype"] in types, "unsupported tensor descriptor/dtype")
            shape, offsets = row["shape"], row["data_offsets"]
            require(isinstance(shape, list) and all(type(x) is int and x >= 0 for x in shape), "invalid tensor dimensions")
            require(isinstance(offsets, list) and len(offsets) == 2 and all(type(x) is int for x in offsets), "invalid tensor offsets")
            start, end = offsets
            dtype, width = types[row["dtype"]]
            require(0 <= start <= end <= payload_size and end - start == math.prod(shape) * width,
                    "tensor byte range does not match dtype/shape")
            intervals.append((start, end))
            descriptors[name] = (dtype, shape, start, end)
        require(descriptors, "empty tensor inventory")
        cursor = 0
        for start, end in sorted(intervals):
            require(start == cursor, "tensor payload has overlaps or unaccounted bytes")
            cursor = end
        same(cursor, payload_size, "unaccounted trailing safetensors payload")
        entries, aggregate = {}, hashlib.sha256()
        for name, (dtype, shape, start, end) in sorted(descriptors.items()):
            stream.seek(8 + length + start)
            remaining, digest = end - start, hashlib.sha256()
            while remaining:
                data = stream.read(min(remaining, 1024 * 1024))
                require(data, "truncated tensor payload")
                digest.update(data)
                remaining -= len(data)
            row = {"dtype": dtype, "shape": shape, "sha256": digest.hexdigest()}
            entries[name] = row
            aggregate.update(canonical({"name": name, **row}))
    return {"safetensors_file_sha256": sha(weights), "canonical_tensor_sha256": aggregate.hexdigest(),
            "tensor_count": len(entries), "tensors": entries}


def git_tar_readback(repo, archive, target, omitted_receipt):
    raw = git(repo, "ls-tree", "-r", "-z", target, "--", str(REL), raw=True)
    expected = {}
    for entry in raw.split(b"\0"):
        if entry:
            meta, name = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            require(mode in ("100644", "100755") and kind == "blob", "nonregular Git entry in F evidence")
            expected[name.decode()] = oid
    require(str(REL / omitted_receipt) in expected, "release receipt is absent from publication commit")
    expected.pop(str(REL / omitted_receipt))  # Written after the workflow constructs its complete TAR.
    object_format = git(repo, "rev-parse", "--show-object-format")
    seen, seen_members, total = set(), set(), 0
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            name = member.name.rstrip("/")
            path = PurePosixPath(name)
            require(name and not path.is_absolute() and "\\" not in name and "\x00" not in name
                    and all(x not in ("", ".", "..") for x in name.split("/")), "unsafe complete TAR path")
            require(name not in seen_members, "duplicate complete TAR member")
            seen_members.add(name)
            require(len(seen_members) <= 50000 and 0 <= member.size <= 1024 * 1024 * 1024, "complete TAR member exceeds bound")
            require(path == PurePosixPath(REL) or PurePosixPath(REL) in path.parents, "complete TAR escapes F scope")
            if member.isdir():
                continue
            require(member.isfile() and name in expected, "unexpected/link/special complete TAR file: " + name)
            total += member.size
            require(total <= 1024 * 1024 * 1024, "complete TAR exceeds bound")
            stream = tar.extractfile(member)
            require(stream is not None, "complete TAR member has no content")
            digest = hashlib.new(object_format)
            digest.update(f"blob {member.size}\0".encode())
            with stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            same(digest.hexdigest(), expected[name], "TAR bytes differ from tagged Git blob: " + name)
            seen.add(name)
    same(seen, set(expected), "complete TAR omits tagged evidence")
    return {"tagged_git_files": len(seen), "uncompressed_file_bytes": total, "exact_git_blob_readback": True}


def bind_worktree_to_tag(repo, target, prefixes):
    """Tie the exact bytes passed to validators to the release's Git tree."""
    expected = {}
    raw = git(repo, "ls-tree", "-r", "-z", target, "--", *map(str, prefixes), raw=True)
    for entry in raw.split(b"\0"):
        if entry:
            meta, name = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            require(mode in ("100644", "100755") and kind == "blob", "nonregular tagged evidence")
            expected[name.decode()] = oid
    actual = set()
    for prefix in prefixes:
        path = repo / prefix
        if path.is_file():
            actual.add(str(prefix))
        elif path.exists():
            actual.update((prefix / name).as_posix() for name in inventory(path))
    same(actual, set(expected), "validated worktree evidence population differs from publication tag")
    object_format = git(repo, "rev-parse", "--show-object-format")
    for name, oid in expected.items():
        path = repo / name
        require(not path.is_symlink(), "validated evidence is a symlink")
        digest = hashlib.new(object_format)
        digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        same(digest.hexdigest(), oid, "validated worktree file differs from tagged bytes: " + name)
    return len(expected)


def recovery_manifest(path, request):
    if not path.exists():
        require(request["resume_run_id"] is None, "resumed phase lacks permanent recovery manifest")
        return None
    value = read(path)
    same(value.get("schema"), "village-001f-permanent-recovery-custody-v1", "wrong permanent recovery schema")
    same(value.get("requested_source_run_id"), request["resume_run_id"], "permanent recovery source differs from request")
    same(value.get("original_zip_files_in_release_only"), True, "recovery ZIP placement changed")
    require(isinstance(value.get("sources"), list), "missing recovery source list")
    sources = value["sources"]
    if request["resume_run_id"] is None:
        same(sources, [], "fresh phase has unsolicited recovery sources")
    else:
        require(any(row["run_id"] == request["resume_run_id"] for row in sources), "latest recovery source is absent")
    return value


def release_readback(repo, directory, api_path, target, tag, baseline, recovery, manifest_name, archive_name):
    require(directory and api_path and target, "actual audit requires explicit release directory, API JSON and target commit")
    require(re.fullmatch(r"[0-9a-f]{40}", target), "release target must be an exact commit")
    same(git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}"), target, "fetched release tag target differs")
    ancestor(repo, target, "HEAD")
    receipt_path = repo / REL / manifest_name
    same(receipt_path.read_bytes(), git(repo, "show", target + ":" + (REL / manifest_name).as_posix(), raw=True),
         "release receipt changed after publication")
    receipt, published = read(receipt_path), read(api_path)
    same(receipt["release_tag"], tag, "release receipt tag differs")
    same(published["tag_name"], tag, "GitHub release tag differs")
    require(published["prerelease"] is True and published["draft"] is False, "diagnostic release is not a published prerelease")
    same(published["target_commitish"], target, "GitHub release target differs from expected bot commit")
    same(git(repo, "show", "-s", "--format=%P", target).split(), [receipt["source_head"]], "publication commit is not the direct child of recorded execution source")
    ancestor(repo, SOURCE_COMMIT, receipt["source_head"])
    require(type(receipt.get("run_id")) is int and receipt["run_id"] > 0, "release run ID is invalid")
    rows = receipt["assets"]
    expected = {row["file"]: row for row in rows}
    remote = {row["name"]: row for row in published["assets"]}
    same(len(expected), len(rows), "duplicate receipt asset")
    same(len(remote), len(published["assets"]), "duplicate GitHub asset")
    extras = {row["archive_file"] for row in recovery["sources"]} if recovery else set()
    same(set(expected), set(baseline) | extras, "release asset set differs from baseline plus declared recovery ZIPs")
    same(set(remote), set(expected), "GitHub release asset population differs")
    same(set(inventory(directory)), set(expected), "local release directory must contain exactly the published assets")
    for name, row in expected.items():
        require(Path(name).name == name and type(row["bytes"]) is int and row["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "invalid release asset descriptor")
        path, asset = directory / name, remote[name]
        same((path.stat().st_size, sha(path)), (row["bytes"], row["sha256"]), "downloaded release bytes differ: " + name)
        same((asset["state"], asset["size"], asset["digest"]), ("uploaded", row["bytes"], "sha256:" + row["sha256"]),
             "GitHub API asset digest/size/state differs: " + name)
        if name in FINAL_JSON:
            same(path.read_bytes(), (repo / REL / "results" / name).read_bytes(), "standalone result asset differs from Git evidence")
            same(path.read_bytes(), git(repo, "show", target + ":" + (REL / "results" / name).as_posix(), raw=True),
                 "standalone result asset differs from tagged Git result")
    tar_report = git_tar_readback(repo, directory / archive_name, target, manifest_name)
    return receipt, {"pass": True, "release_id": published["id"], "tag": tag, "tag_commit": target,
        "api_snapshot_sha256": sha(api_path), "assets": len(expected), "baseline_assets": len(baseline),
        "recovery_zip_assets": len(extras), "verified_assets": expected, **tar_report}


def request_at(repo, commit):
    raw = git(repo, "show", commit + ":" + (REL / "request.json").as_posix(), raw=True)
    value = json.loads(raw)
    return value, hashlib.sha256(raw).hexdigest()


def verify_phase_authority(repo, request, phase, source, context):
    same(request.get("phase"), phase, "wrong execution phase in source request")
    for key, expected in {"schema": "village-001f-post-e-request-v1", "publish_branch": context.op.BRANCH,
        "input_release_tag": INPUT_TAG, "final_release_tag": FINAL_TAG,
        "maximum_training_lanes": 12, "maximum_optimizer_steps": 1440,
        "new_worker_generations": 300, "new_sprocket_generations": 36,
        "maximum_provider_logical_requests": 114}.items():
        same(request.get(key), expected, "source request authority/budget differs: " + key)
    same(request["contract_sha256"], {name: sha(repo / REL / name) for name in CONTRACTS}, "source request contract hashes differ")
    accepted = request["stage_e_main_commit"]
    require(re.fullmatch(r"[0-9a-f]{40}", accepted), "accepted E commit is invalid")
    ancestor(repo, accepted, source)
    raw = git(repo, "show", accepted + ":" + context.op.E_SUMMARY, raw=True)
    same(hashlib.sha256(raw).hexdigest(), request["stage_e_summary_sha256"], "accepted E summary differs from source request")
    same(sha(repo / context.op.E_SUMMARY), request["stage_e_summary_sha256"], "accepted E evidence changed in audited checkout")
    require(type(request["execution_revision"]) is int and request["execution_revision"] > 0, "source request revision is invalid")
    require(request["resume_run_id"] is None or (type(request["resume_run_id"]) is int and request["resume_run_id"] > 0), "source recovery run ID is invalid")


def verify_actions_history(repo, root, receipt, request, context):
    """Replay the frozen launch-history guard from saved complete API pages.

    Later runs are excluded to reconstruct the inventory as of this launch.
    The only substituted git operation is fetch: its exact object must already
    exist locally. No object, ref, checkout or remote is changed by this audit.
    """
    require(root is not None, "actual audit requires --history-api-dir")
    used = {}
    def snapshot(relative):
        path = root / relative
        value = read(path)
        used[relative] = sha(path)
        return value
    run_id = receipt["run_id"]
    run = snapshot(f"runs/{run_id}.json")
    same(context.op.validate_recovery_run(run, run_id), receipt["source_head"], "execution run source differs from release receipt")
    same(run.get("event"), "push", "execution run event differs")
    same(run.get("run_attempt"), 1, "published phase must identify its first Actions attempt")
    full = snapshot("workflow-runs.json")
    runs = full.get("workflow_runs", [])
    require(type(full.get("total_count")) is int and full["total_count"] == len(runs) <= 100,
            "complete bounded workflow history snapshot required")
    require(len({row["id"] for row in runs}) == len(runs) and sum(row["id"] == run_id for row in runs) == 1,
            "workflow history omits or duplicates the publication run")
    relevant = [row for row in runs if row["id"] <= run_id]
    current = next(row for row in relevant if row["id"] == run_id)
    same(current["head_sha"], run["head_sha"], "run-list source differs from run detail")
    def gh(resource):
        if resource.startswith("actions/workflows/village-001f-diagnostics.yml/runs?"):
            return {**full, "workflow_runs": relevant, "total_count": len(relevant)}
        match = re.fullmatch(r"actions/runs/([0-9]+)/(jobs|artifacts)\?.+", resource)
        require(match is not None, "unexpected frozen history API request: " + resource)
        return snapshot(f"{match[2]}/{match[1]}.json")
    original_subprocess, original_gh = context.op.subprocess, context.op.gh
    def run_command(command, **kwargs):
        if command[:4] == ["git", "fetch", "--no-tags", "origin"]:
            require(len(command) == 5, "unexpected offline fetch shape")
            git(repo, "cat-file", "-e", command[4] + "^{commit}")
            return subprocess.CompletedProcess(command, 0)
        require(command[:3] == ["git", "merge-base", "--is-ancestor"], "unexpected history subprocess mutation")
        return subprocess.run(command, cwd=repo, **kwargs)
    def output_command(command, **kwargs):
        require(command[:2] == ["git", "show"], "unexpected history subprocess read")
        return subprocess.check_output(command, cwd=repo, **kwargs)
    saved = {key: os.environ.get(key) for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")}
    try:
        context.op.gh = gh
        context.op.subprocess = SimpleNamespace(run=run_command, check_output=output_command, DEVNULL=subprocess.DEVNULL)
        os.environ.update(GITHUB_RUN_ID=str(run_id), GITHUB_RUN_ATTEMPT="1")
        context.op.verify_execution_history(request)
    finally:
        context.op.gh, context.op.subprocess = original_gh, original_subprocess
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return {"pass": True, "run_id": run_id, "source_head": run["head_sha"],
        "completed_run_conclusion": run.get("conclusion"), "phase": request["phase"],
        "latest_spending_resume_run_id": request["resume_run_id"], "as_of_launch_runs": len(relevant),
        "api_snapshots_sha256": used}


def verify_recovery(repo, scratch, context, manifest, evidence, release_dir, request, source_head):
    if manifest is None:
        return {"verified_source_zips": 0, "immutable_prior_files_preserved": 0}
    stage = scratch / ("recovery-" + request["phase"])
    stage.mkdir()
    rows = manifest["sources"]
    require(len(rows) <= 256 and len({row["artifact_id"] for row in rows}) == len(rows), "recovery population is duplicate or unbounded")
    same(set(inventory(evidence)), {"recovery-manifest.json"} | {row["receipt_file"] for row in rows},
         "permanent recovery evidence has missing or unbound files")
    preserved = 0
    for row in rows:
        for field in ("receipt_file", "archive_file"):
            require(Path(row[field]).name == row[field], "unsafe recovery asset basename")
        receipt_path, archive_path = evidence / row["receipt_file"], release_dir / row["archive_file"]
        same(sha(receipt_path), row["receipt_sha256"], "permanent recovery receipt digest differs")
        same(sha(archive_path), row["archive_sha256"], "permanent recovery archive digest differs")
        shutil.copy2(receipt_path, stage / row["receipt_file"])
        shutil.copy2(archive_path, stage / row["archive_file"])
        ancestor(repo, row["source_head"], source_head)
        old_request, _ = request_at(repo, row["source_head"])
        for key in ("phase", "contract_sha256", "stage_e_main_commit", "stage_e_summary_sha256", "input_release_tag", "final_release_tag"):
            same(old_request.get(key), request[key], "inherited recovery authority differs: " + key)
    checked = context.op.recovery_sources(stage, request["phase"])
    same([{k: v for k, v in row.items() if not k.endswith("_path")} for row in checked], rows,
         "release ZIPs/receipts do not reconstruct permanent recovery manifest")
    for row in checked:
        prior = scratch / ("prior-" + str(row["artifact_id"]))
        context.op.safe_extract(row["archive_path"], prior)
        name = row["artifact_name"]
        prior_files = inventory(prior)
        if name == "village001f-judgments":
            latest = repo / REL / "results"
            selected = {p: meta for p, meta in prior_files.items()
                if p.startswith(("provider/", "sources/")) or p in {"provider-identity.json", "blinded-inputs.json", "private-mapping.json", "source-custody.json"}}
        elif name == "village001f-sprocket":
            latest = repo / REL / "inputs/sprocket"
            selected = {p: meta for p, meta in prior_files.items() if p.startswith(("http/", "rows/", "wire/")) or p == "plan.json"}
        elif name.startswith("village001f-infer-"):
            latest = repo / REL / "inputs/worker" / name
            selected = {p: meta for p, meta in prior_files.items()
                if "/rows/" in "/" + p or Path(p).name in {"generation-identity.json", "result.json"}}
        else:
            match = re.fullmatch(r"village001f-train-(V-G[0-9]{2}-[ABC])-(historical_token_id_mask|length_derived_mask)", name)
            require(match is not None, "unexpected training recovery artifact")
            latest = repo / REL / "inputs/training" / match[1] / match[2]
            selected = {p: meta for p, meta in prior_files.items()
                if p in {"training-identity.json", "result.json", "training.json"} or p.startswith("training-invocations/")}
            checkpoint = prior / "checkpoint/progress.json"
            if checkpoint.exists():
                old, final = read(checkpoint), read(latest / "result.json")
                require(0 <= old["next_step"] <= 120, "prior training checkpoint exceeds fixed step budget")
                for key in ("losses", "preference_losses", "anchor_losses", "train_margins"):
                    same(old[key], final["training"][key][:old["next_step"]], "training resume discarded prior optimizer trace: " + key)
        current = inventory(latest)
        require(all(current.get(path) == meta for path, meta in selected.items()), "recovery discarded immutable prior evidence: " + name)
        preserved += len(selected)
    return {"verified_source_zips": len(checked), "immutable_prior_files_preserved": preserved,
        "source_artifact_ids": [row["artifact_id"] for row in checked]}


def validate_paired_optimizer_provenance(context, training, training_sources):
    """Check native proofs against retained segments, not the terminal copy host."""
    proofs, counts = {}, Counter()
    for member in context.worker.PANEL:
        proof = context.judge.paired_execution_provenance(member, training)
        same(proof["predeclared_arm_order"], context.worker.training_arm_order(context.wcfg, member),
            "reported training arm order differs from prospective contract")
        executions, unknown, fresh = [], [], []
        for arm, stage in context.judge.ARM_STAGE.items():
            native = training[(member, stage)]["execution_provenance"]
            reported = proof["arms"][stage]
            same(reported["optimizer_segments"], native["optimizer_segments"], "paired proof changed optimizer hosts")
            same(reported["unattributed_optimizer_segments"], native["unattributed_optimizer_segments"],
                "paired proof discarded unknown interrupted work")
            same(sum(row["steps"] for row in [*native["optimizer_segments"], *native["unattributed_optimizer_segments"]]),
                120, "paired proof does not account for120 retained steps")
            for name, meta in native["files"].items():
                path = training_sources[f"training/{member}/{arm}/{name}"]
                same({"sha256": sha(path), "bytes": path.stat().st_size}, meta,
                    "native invocation sidecar is not retained in judge sources")
            copied_ordinals = set(native["zero_step_copies"])
            require(not copied_ordinals.intersection(row["invocation_ordinal"] for row in native["optimizer_segments"]),
                "zero-step copy was miscounted as optimizer execution")
            invocations, segments = native["invocations"], native["optimizer_segments"]
            fresh.append(len(invocations) == len(segments) == 1 and not native["unattributed_optimizer_segments"]
                and not any(native[key] for key in ("zero_step_copies", "resume_invocations", "unfinished_invocations"))
                and invocations[0]["start"]["source_kind"] == "fresh"
                and not invocations[0]["start"]["resume_requested"]
                and invocations[0]["finish"] is not None and invocations[0]["finish"]["status"] == "completed"
                and (segments[0]["start_step"], segments[0]["end_step"]) == (0, 120))
            executions.extend(row["execution"] for row in segments)
            unknown.extend(native["unattributed_optimizer_segments"])
            counts["sidecar_files"] += len(native["files"])
            counts["invocations"] += len(invocations)
            counts["zero_step_copies"] += len(copied_ordinals)
            counts["unattributed_retained_optimizer_steps"] += sum(row["steps"] for row in native["unattributed_optimizer_segments"])
        def shared(fields):
            columns = [[row.get(field) for row in executions] for field in fields]
            if any(len({value for value in column if value is not None and value != ""}) > 1 for column in columns):
                return False
            if unknown or not executions or any(any(value is None or value == "" for value in column) for column in columns):
                return None
            return True
        same(proof["same_job_for_optimizer_segments"], shared(("repository", "run_id", "run_attempt", "job",
            "member_id", "paired_job_nonce", "source_head")), "shared optimizer job claim differs from native invocations")
        same(proof["same_runner_instance_for_optimizer_segments"], shared(("runner_name", "boot_id", "hostname")),
            "shared optimizer runner claim differs from native invocations")
        same(proof["fresh_uninterrupted_pair"], all(fresh), "fresh pairing claim ignores recovery history")
        if all(fresh):
            first, second = [training[(member, context.judge.ARM_STAGE[arm])]["execution_provenance"]["invocations"][0]
                for arm in proof["predeclared_arm_order"]]
            require(first["start"]["execution"]["process_id"] != second["start"]["execution"]["process_id"],
                "fresh arms did not use distinct recorded processes")
            require(first["finish"]["finished_monotonic_ns"] <= second["start"]["execution"]["started_monotonic_ns"],
                "fresh paired processes did not follow the predeclared sequential order")
        proofs[member] = proof
    return proofs, {"pass": True, "member_pairs": len(proofs), **dict(counts),
        "classifications": dict(Counter(proof["classification"] for proof in proofs.values())),
        "original_D_has_equivalent_invocation_history": False,
        "same_runner_implies_numerical_determinism": False}


def validate_native_inputs(repo, scratch, context):
    inputs = repo / REL / "inputs"
    context.op.validate_input_sources(inputs)
    training, training_sources = context.judge.load_training(inputs / "training", context.jcfg, context.wcfg, context.campaign, context.pref)
    _, paired_provenance = validate_paired_optimizer_provenance(context, training, training_sources)
    workers, _ = context.judge.load_workers(inputs / "worker", training, context.jcfg, context.wcfg, context.suite)
    sprocket_rows, _, _ = context.judge.load_sprocket(inputs / "sprocket", context.jcfg)
    audit_path = next((inputs / "token-audit").rglob("token-mask-audit.json"))
    audit = read(audit_path)
    groups = defaultdict(list)
    for row in audit["records"]:
        prompt_count = len(row["prompt_token_ids"])
        common = row["prompt_common_prefix_tokens"]
        same(row["prompt_token_count"], prompt_count, "token audit prompt-count arithmetic differs")
        same(row["prompt_tokens_supervised_by_historical_prefix_rule"], prompt_count - common, "token audit prefix arithmetic differs")
        same(row["supervised_tokens"], len(row["input_token_ids"]) - common, "token audit supervised count differs")
        require(isinstance(row["rendered_native_training_sequence"], str) and row["rendered_native_training_sequence"], "missing native training prompt")
        require(isinstance(row["real_masked_token_strings"], list)
                and len(row["real_masked_token_strings"]) == len(row["real_tokens_masked_by_historical_rule"]), "masked-token text count differs")
        groups[row["member_id"]].append(row)
    per_member = [{"member_id": member, "sequences": len(groups[member]),
        "sequences_with_real_tokens_masked": sum(bool(row["real_tokens_masked_by_historical_rule"]) for row in groups[member]),
        "real_tokens_masked": sum(len(row["real_tokens_masked_by_historical_rule"]) for row in groups[member]),
        "prompt_prefix_mismatch_sequences": sum(row["prompt_tokens_supervised_by_historical_prefix_rule"] != 0 for row in groups[member])}
        for member in context.worker.PANEL]
    same(audit["per_member"], per_member, "token-audit aggregate counters differ from 720 rows")
    for lane in training.values():
        result = lane["result"]
        for key in ("losses", "preference_losses", "anchor_losses", "train_margins"):
            trace = result["training"][key]
            require(len(trace) == 120 and all(type(value) in (int, float) and math.isfinite(value) for value in trace),
                    "matched training has incomplete/nonfinite optimizer trace")
        same(result["recovery"]["final_checkpoint_step"], 120, "terminal training checkpoint count differs")
    replay = scratch / "sprocket-replay"
    context.sprocket.prepare_output(replay, inputs / "sprocket", context.splan)
    with quiet():
        context.sprocket.execute(replay, context.scfg, context.splan, scratch / "unused-model-cache", 18950, offline=True)
    same(inventory(replay), inventory(inputs / "sprocket"), "offline Sprocket reconstruction changes archived bytes")
    scores = list((inputs / "worker").rglob("score.json"))
    same(sum(read(path)["model_forward_batches"] for path in scores), 288, "terminal forward batch total differs")
    return training, workers, sprocket_rows, {"pass": True, "training_lanes": len(training),
        "terminal_optimizer_steps": 1440, "native_worker_lanes": len(workers), "native_worker_responses": 300,
        "paired_training_execution_provenance": paired_provenance,
        "sprocket_responses": len(sprocket_rows), "sprocket_offline_reconstruction": "byte-identical",
        "token_audit_sequences": len(audit["records"]), "forward_score_lanes": len(scores),
        "terminal_forward_batches": 288, "token_audit_aggregate_counters_recomputed": True,
        "limitations": ["Recorded token IDs, prompts, labels and native chains are checked by the frozen validators; this audit does not rerun the tokenizer.",
            "Archived forward scores are source-bound and their arithmetic is checked; model likelihoods are not recalculated.",
            "Terminal workload counts do not estimate computation lost before an interrupted operation could create a durable receipt."]}


def validate_adapter_bytes(repo, scratch, context, training, workers, release_dir, reference_dir):
    require(reference_dir is not None, "actual acceptance requires --reference-assets-dir for historical C/D fingerprint checks")
    results = []
    plan = read(repo / REL / "worker-plan.json")
    for member in context.worker.PANEL:
        for arm, stage in context.judge.ARM_STAGE.items():
            name = f"village001f-{member}-{arm}.tar.gz"
            adapter = scratch / "new-adapters" / member / arm
            contents = safe_adapter_extract(release_dir / name, adapter)
            receipt_dir = repo / REL / "inputs/training" / member / arm
            _, directory_sha = context.worker.checked_adapter(context.wcfg, member, stage, adapter, receipt_dir)
            fingerprint = raw_safetensors_fingerprint(adapter)
            context.judge.validate_fingerprint(fingerprint)
            same(fingerprint, training[(member, stage)]["receipt"]["adapter_weight_fingerprint"], "published pilot tensor bytes differ from training receipt")
            same(fingerprint, workers[(member, stage)]["result"]["adapter_weight_fingerprint"], "published pilot tensors differ from inference receipt")
            same(sum(row["bytes"] for row in contents.values()), training[(member, stage)]["result"]["adapter"]["bytes"], "published pilot directory byte count differs")
            results.append({"member_id": member, "stage": stage, "asset": name, "directory_sha256": directory_sha,
                "safetensors_file_sha256": fingerprint["safetensors_file_sha256"],
                "canonical_tensor_sha256": fingerprint["canonical_tensor_sha256"],
                "tensor_count": fingerprint["tensor_count"], "all_tensor_bytes_match_native_receipts": True})
    expected_references = {row["file"] for source in plan["adapter_downloads"].values() for row in source["assets"]}
    same(set(inventory(reference_dir)), expected_references, "historical reference directory must contain exactly the 12 frozen TARs")
    for stage, source in plan["adapter_downloads"].items():
        for row in source["assets"]:
            member, archive = row["member_id"], reference_dir / row["file"]
            same((archive.stat().st_size, sha(archive)), (row["bytes"], row["sha256"]), "historical reference archive differs from frozen plan")
            adapter = scratch / "historical-adapters" / stage / member
            contents = safe_adapter_extract(archive, adapter)
            _, directory_sha = context.worker.checked_adapter(context.wcfg, member, stage, adapter, None)
            same(directory_sha, row["adapter_directory_sha256"], "historical reference directory hash differs from frozen plan")
            same(sum(value["bytes"] for value in contents.values()), row["adapter_directory_bytes"], "historical reference directory size differs")
            fingerprint = raw_safetensors_fingerprint(adapter)
            context.judge.validate_fingerprint(fingerprint)
            same(fingerprint, workers[(member, stage)]["result"]["adapter_weight_fingerprint"], "historical tensor bytes differ from F native inference receipt")
            results.append({"member_id": member, "stage": stage, "asset": row["file"], "release_tag": source["release_tag"],
                "directory_sha256": directory_sha, "safetensors_file_sha256": fingerprint["safetensors_file_sha256"],
                "canonical_tensor_sha256": fingerprint["canonical_tensor_sha256"],
                "tensor_count": fingerprint["tensor_count"], "all_tensor_bytes_match_native_receipts": True})
    same(len(results), 24, "adapter fingerprint population differs")
    return {"pass": True, "new_published_adapters": 12, "historical_reference_adapters": 12,
        "method": "stdlib safetensors raw-byte hashes with frozen canonical JSON framing; no model loading", "adapters": results}


def independent_metrics(repo, context, sprocket_rows, training):
    results = repo / REL / "results"
    judgments, mapping, summary = read(results / "judgments.json"), read(results / "private-mapping.json"), read(results / "summary.json")
    ratings = judgments["ratings"]
    same({kind: len(rows) for kind, rows in ratings.items()}, {"primary": 72, "context": 156, "sprocket": 36}, "judgment cardinalities differ")
    by_member = defaultdict(list)
    for ident, rating in ratings["primary"].items():
        item = mapping["primary"][ident]
        side = "A" if item["A"] == "D_CORRECTED" else "B"
        other = "B" if side == "A" else "A"
        by_member[item["member_id"]].append({key: rating[side][key] - rating[other][key] for key in context.judge.WORKER_SCORES})
    same(set(by_member), set(context.worker.PANEL), "paired metric panel differs")
    means = {}
    reported = {row["member_id"]: row for row in summary["worker"]["per_member"]}
    for member, rows in by_member.items():
        same(len(rows), 12, "paired metric case population differs")
        means[member] = {key: statistics.mean(row[key] for row in rows) for key in context.judge.WORKER_SCORES}
        same(reported[member]["mean_delta"], means[member], "member effect arithmetic differs")
        same(reported[member]["training_execution"]["paired_execution_provenance"],
            context.judge.paired_execution_provenance(member, training), "final report differs from native training execution provenance")
    population = {key: sum(row[key] for rows in by_member.values() for row in rows) / 72
        for key in context.judge.WORKER_SCORES}
    gains = [means[member]["task_relevance"] for member in context.worker.PANEL]
    threshold = context.wcfg["primary_analysis"]
    criteria = {"mean_gain": population["task_relevance"] >= threshold["working_effect_threshold"],
        "members_positive": sum(value > 0 for value in gains) >= threshold["minimum_members_with_positive_mean_gain"],
        "maximum_member_regression": min(gains) >= -threshold["maximum_member_mean_relevance_regression"]}
    worker = summary["worker"]
    same(worker["mean_delta"], population, "population effect arithmetic differs")
    same(worker["working_effect_criteria"], criteria, "predeclared working-effect criteria differ")
    same(worker["working_effect_signal"], all(criteria.values()), "working-effect conjunction differs")
    same(worker["positive_member_relevance_gains"], sum(value > 0 for value in gains), "positive-member count differs")
    same(worker["worst_member_relevance_delta"], min(gains), "worst-member metric differs")
    same(worker["automatic_rollout"], False, "unexpected automatic rollout")
    same(worker["paired_execution_classifications"], dict(Counter(
        row["training_execution"]["paired_execution_provenance"]["classification"] for row in reported.values())),
        "paired execution classification counts differ")
    classification = "working_effect_criteria_met" if all(criteria.values()) else (
        "member_harm_exceeds_limit" if not criteria["maximum_member_regression"] else "working_effect_criteria_not_met")
    same(worker["classification"], classification, "worker classification differs")
    for stage in ("BASE", "C", "D_ORIGINAL"):
        selected = [rating for ident, rating in ratings["context"].items() if mapping["context"][ident]["stage"] == stage]
        expected = {"responses": len(selected), "means": {key: statistics.mean(row[key] for row in selected) for key in context.judge.WORKER_SCORES}}
        same(worker["context_summary"][stage], expected, "context metric arithmetic differs")
    native = {row["id"]: row for row in sprocket_rows}
    arms, pairs = defaultdict(list), defaultdict(dict)
    for ident, rating in ratings["sprocket"].items():
        item, source = mapping["sprocket"][ident], native[ident]
        grounded = rating["response_grounding"] >= 1 and rating["scenario_grounding"] >= 1 and not rating["unsupported_claims"]
        row = {**rating, "legacy_schema_valid": source["legacy_schema_valid"],
            "strict_json_schema_valid": source["strict_json_schema_valid"], "empty_response": source["empty_response"],
            "grounded": grounded, "strict_and_grounded": source["strict_json_schema_valid"] and grounded,
            "strict_grounded_and_actionable": source["strict_json_schema_valid"] and grounded and rating["corrective_usefulness"] >= 1}
        arms[item["arm"]].append(row)
        pair = pairs[(item["case_id"], item["worker"])]
        require(item["arm"] not in pair, "duplicate matched Sprocket arm")
        pair[item["arm"]] = row
    same(set(arms), {"original", "compact_schema_first"}, "Sprocket arm set differs")
    same(len(pairs), 18, "Sprocket case/seed pair count differs")
    for pair in pairs.values():
        same(set(pair), set(arms), "incomplete matched Sprocket pair")
    for arm, rows in arms.items():
        same(len(rows), 18, "Sprocket arm count differs")
        expected = {"responses": len(rows), **{key: sum(row[key] for row in rows) for key in
            ("legacy_schema_valid", "strict_json_schema_valid", "empty_response", "unsupported_claims", "grounded", "strict_and_grounded", "strict_grounded_and_actionable")},
            "means": {key: statistics.mean(row[key] for row in rows) for key in context.judge.SPROCKET_SCORES}}
        same(summary["sprocket"]["arms"][arm], expected, "Sprocket arm metric arithmetic differs")
    syntax = sum(row["strict_json_schema_valid"] for row in arms["compact_schema_first"]) - sum(row["strict_json_schema_valid"] for row in arms["original"])
    useful = sum(row["strict_grounded_and_actionable"] for row in arms["compact_schema_first"]) - sum(row["strict_grounded_and_actionable"] for row in arms["original"])
    same(summary["sprocket"]["strict_syntax_count_gain"], syntax, "Sprocket syntax gain differs")
    same(summary["sprocket"]["strict_grounded_actionable_count_gain"], useful, "Sprocket grounded/useful gain differs")
    sp_classification = "syntax_gain_without_grounded_actionable_gain" if syntax > 0 and useful <= 0 else (
        "grounded_actionable_count_gain" if useful > 0 else "no_grounded_actionable_count_gain")
    same(summary["sprocket"]["classification"], sp_classification, "Sprocket classification differs")
    return {"pass": True, "worker_mean_delta": population, "working_effect_criteria": criteria,
        "worker_classification": classification, "sprocket_syntax_count_gain": syntax,
        "sprocket_grounded_actionable_count_gain": useful, "sprocket_classification": sp_classification,
        "positive_effect_required_for_evidence_acceptance": False}


def replay_final(repo, scratch, context):
    source, replay = repo / REL / "results", scratch / "judge-offline-replay"
    original = inventory(source)
    before_receipts = {name for name in original if re.fullmatch(r"execution-[0-9]{3}\.json", name)}
    old_argv = sys.argv
    try:
        sys.argv = [context.judge.__file__, "--config", str(repo / REL / "judge-config.json"),
            "--worker-artifacts", str(repo / REL / "inputs/worker"),
            "--training-artifacts", str(repo / REL / "inputs/training"),
            "--sprocket-artifacts", str(repo / REL / "inputs/sprocket"),
            "--resume", str(source), "--out", str(replay), "--offline"]
        with quiet():
            context.judge.main()
    finally:
        sys.argv = old_argv
    after = inventory(replay)
    after_receipts = {name for name in after if re.fullmatch(r"execution-[0-9]{3}\.json", name)}
    same(len(after_receipts - before_receipts), 1, "offline replay did not create exactly one invocation receipt")
    for name in before_receipts:
        same(after[name], original[name], "offline replay changed an original invocation receipt")
    new_receipt = read(replay / next(iter(after_receipts - before_receipts)))
    same((new_receipt["offline"], new_receipt["new_provider_requests"], new_receipt["logical_request_limit"]),
         (True, 0, 114), "offline judge replay records spending or a changed limit")
    same({key: value for key, value in after.items() if key not in after_receipts},
         {key: value for key, value in original.items() if key not in before_receipts},
         "offline provider/metrics/report reconstruction changed immutable bytes")
    declared = read(source / "hashes.json")
    same(declared["schema_version"], 1, "wrong final hash inventory schema")
    same(declared["files"], {key: value for key, value in original.items() if key != "hashes.json" and key not in before_receipts},
         "final evidence hash inventory differs")
    requests = sorted((source / "provider").glob("*-request.json"))
    groups = defaultdict(list)
    attempts = 0
    for path in requests:
        label = path.name.removesuffix("-request.json")
        logical = re.sub(r"-transport-resume-[0-9]{2}$", "", label)
        groups[logical].append(label)
        trace = path.with_name(label + "-attempts.json")
        if trace.exists():
            rows = read(trace)
            require(isinstance(rows, list) and len(rows) <= context.campaign["nvidia"]["generation"]["attempts"], "provider HTTP-attempt ceiling exceeded")
            attempts += len(rows)
    for logical, labels in groups.items():
        allowed = [logical, logical + "-transport-resume-01", logical + "-transport-resume-02"]
        same(sorted(labels), sorted(allowed[:len(labels)]), "provider transport-launch population is not a bounded prefix")
    counts = {"worker": sum(label.startswith("f-worker-") for label in groups),
              "sprocket": sum(label.startswith("f-sprocket-") for label in groups)}
    require(sum(counts.values()) == len(groups) and counts["worker"] <= 88 and counts["sprocket"] <= 26 and len(groups) <= 114,
            "provider logical population/limits differ")
    same(read(source / "summary.json")["provider"]["logical_requests"], len(groups), "summary logical request count differs")
    return {"pass": True, "offline_reconstruction": "byte-identical", "new_provider_requests": 0,
        "logical_requests": len(groups), "logical_requests_by_family": counts,
        "transport_launches_archived": len(requests), "http_attempt_records_archived": attempts,
        "logical_limits": {"worker": 88, "sprocket": 26, "total": 114}, "transport_launches_per_logical_limit": 3,
        "http_attempts_per_launch_limit": context.campaign["nvidia"]["generation"]["attempts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--mode", choices=("source", "inputs", "final"), default="source")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-release-dir", type=Path)
    parser.add_argument("--input-release-json", type=Path)
    parser.add_argument("--input-target")
    parser.add_argument("--final-release-dir", type=Path)
    parser.add_argument("--final-release-json", type=Path)
    parser.add_argument("--final-target")
    parser.add_argument("--reference-assets-dir", type=Path)
    parser.add_argument("--history-api-dir", type=Path)
    parser.add_argument("--require-main", action="store_true", help="Require HEAD to equal the already fetched origin/main")
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    require(repo != out and repo not in out.parents, "audit report must remain outside the repository")
    require(not out.exists() and not args.out.is_symlink(), "audit output must be a new file")
    same(git(repo, "rev-parse", "HEAD"), args.expected_head, "checkout is not at the requested exact head")
    same(git(repo, "status", "--porcelain", "--untracked-files=no"), "", "audit checkout has tracked modifications")
    if args.require_main:
        same(git(repo, "rev-parse", "origin/main"), args.expected_head, "audited HEAD is not fetched origin/main")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "village-001f-independent-acceptance-v1", "mode": args.mode,
        "audit_head": args.expected_head, "audit_script_sha256": sha(Path(__file__)),
        "source_freeze_commit": SOURCE_COMMIT, "source_acceptance": "not_evaluated",
        "input_acceptance": "not_evaluated", "final_acceptance": "not_evaluated", "closure_ready": False,
        "new_model_generations": 0, "new_optimizer_steps": 0, "new_model_forward_batches": 0,
        "new_provider_calls": 0, "new_tokenizer_calls": 0}
    failure = None
    try:
        with tempfile.TemporaryDirectory(prefix="independent-f-", dir=out.parent) as temporary:
            scratch = Path(temporary)
            context, source = source_preflight(repo, scratch)
            report.update(source_acceptance="pass", source=source)
            print("Verified frozen F source, contracts and native plans; no execution claimed", flush=True)
            if args.mode != "source":
                for name in ("input_release_dir", "input_release_json", "input_target", "reference_assets_dir", "history_api_dir"):
                    require(getattr(args, name) is not None, "actual audit requires --" + name.replace("_", "-"))
                current_request = context.op.request()
                custody = context.op.verify_inputs()
                input_request, request_sha = request_at(repo, custody["source_head"])
                same(request_sha, custody["request_sha256"], "input request does not match custody receipt")
                verify_phase_authority(repo, input_request, "generate", custody["source_head"], context)
                for key in ("stage_e_main_commit", "stage_e_summary_sha256", "contract_sha256", "input_release_tag", "final_release_tag"):
                    same(input_request[key], current_request[key], "input/final authority differs: " + key)
                input_recovery_path = repo / REL / "inputs/recovery/recovery-manifest.json"
                input_recovery = recovery_manifest(input_recovery_path, input_request)
                baseline = {INPUT_TAR} | {f"village001f-{member}-{arm}.tar.gz" for member in context.worker.PANEL for arm in context.judge.ARM_STAGE}
                input_receipt, release = release_readback(repo, args.input_release_dir, args.input_release_json,
                    args.input_target, INPUT_TAG, baseline, input_recovery, "input-release-assets.json", INPUT_TAR)
                release["validated_worktree_files_bound_to_tag"] = bind_worktree_to_tag(repo, args.input_target,
                    [REL / "inputs", REL / "input-custody.json", REL / "input-release-assets.json"])
                same((input_receipt["run_id"], input_receipt["source_head"]), (custody["run_id"], custody["source_head"]),
                     "input release and native custody provenance differ")
                report["input_release"] = release
                report["input_launch_history"] = verify_actions_history(repo, args.history_api_dir, input_receipt, input_request, context)
                report["input_recovery"] = verify_recovery(repo, scratch, context, input_recovery,
                    input_recovery_path.parent, args.input_release_dir, input_request, input_receipt["source_head"])
                training, workers, sp_rows, native = validate_native_inputs(repo, scratch, context)
                report["native_inputs"] = native
                report["adapter_bytes"] = validate_adapter_bytes(repo, scratch, context, training, workers,
                    args.input_release_dir, args.reference_assets_dir)
                report["input_acceptance"] = "pass"
                print("Verified actual F inputs, 24 adapter fingerprints and complete input release readback", flush=True)
                if args.mode == "final":
                    for name in ("final_release_dir", "final_release_json", "final_target"):
                        require(getattr(args, name) is not None, "final audit requires --" + name.replace("_", "-"))
                    final_receipt = read(repo / REL / "final-release-assets.json")
                    final_request, _ = request_at(repo, final_receipt["source_head"])
                    same(final_request, current_request, "final request differs from publication source")
                    verify_phase_authority(repo, final_request, "judge", final_receipt["source_head"], context)
                    ancestor(repo, args.input_target, final_receipt["source_head"])
                    final_recovery_path = repo / REL / "recovery/judge/recovery-manifest.json"
                    final_recovery = recovery_manifest(final_recovery_path, final_request)
                    _, release = release_readback(repo, args.final_release_dir, args.final_release_json,
                        args.final_target, FINAL_TAG, FINAL_JSON | {FINAL_TAR}, final_recovery,
                        "final-release-assets.json", FINAL_TAR)
                    release["validated_worktree_files_bound_to_tag"] = bind_worktree_to_tag(repo, args.final_target,
                        [REL / "results", REL / "recovery", REL / "final-release-assets.json", REL / "request.json"])
                    report["final_release"] = release
                    report["final_launch_history"] = verify_actions_history(repo, args.history_api_dir, final_receipt, final_request, context)
                    report["final_recovery"] = verify_recovery(repo, scratch, context, final_recovery,
                        final_recovery_path.parent, args.final_release_dir, final_request, final_receipt["source_head"])
                    report["provider_replay"] = replay_final(repo, scratch, context)
                    report["independent_metrics"] = independent_metrics(repo, context, sp_rows, training)
                    report["final_acceptance"] = "pass"
                    report["closure_ready"] = args.require_main
                    print("Verified final provider replay, metrics, recovery accounting and release readback", flush=True)
        same(git(repo, "rev-parse", "HEAD"), args.expected_head, "repository HEAD moved during audit")
        same(git(repo, "status", "--porcelain", "--untracked-files=no"), "", "audit modified tracked repository content")
    except Exception as exc:
        failure = exc
        report.update(failed=True, error=type(exc).__name__ + ": " + str(exc), closure_ready=False)
    out.write_bytes(canonical(report))
    if failure:
        raise failure
    print(json.dumps({key: report[key] for key in ("mode", "source_acceptance", "input_acceptance", "final_acceptance", "closure_ready")}, indent=2))


if __name__ == "__main__":
    main()
