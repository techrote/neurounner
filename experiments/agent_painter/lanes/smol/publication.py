#!/usr/bin/env python3
"""Hosted GitHub custody collection and immutable recovery; never execute a model or native runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[4]
REPOSITORY = "techrote/cybersand"
SHA = re.compile(r"[a-f0-9]{40}\Z")
DIGEST = re.compile(r"sha256:([a-f0-9]{64})\Z")
MAX_BYTES = 2 * 1024**3
MAX_MEMBERS = 20000
MAX_WAIT_SECONDS = 3600
MAX_DOWNLOAD_SECONDS = 1200
NATIVE_JOB = "Current-source native replay and immutable evidence"
NATIVE_STEP = "Rebuild unchanged native authority and verify actual raw model drawings"
PACKAGING_STEP = "Verify checkpoint bytes and package transparent research evidence"
RUNTIME_PATH = "godot/addons/cybersand_native/bin/libcybersand_native.linux.x86_64.so"
RECOVERY_REQUEST = {
    "schema_version": 1, "kind": "smol-publication-recovery",
    "producer_source": "fef6654ecd0828f765af1a13d2d10469013a3ea6",
    "producer_run_id": 43522896533, "producer_run_attempt": 1, "sequence": 2,
    "reason": "release-api-forbidden", "delivery": "github-actions-artifact",
}
SEALED_CASE_PREFIXES = ("greebles-staggered_cassettes-", "pipework-double_loop_bridge-",
                        "architecture-three_bay_arcade-")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(_value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def read_json(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Exclusive creation preserves earlier records, including partial failures."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True, timeout=60).strip()


def hosted_identity(operation, directory):
    env = os.environ
    require(env.get("GITHUB_ACTIONS") == "true" and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
            and env.get("RUNNER_OS") == "Linux", "requires an actual Linux GitHub-hosted Actions runner")
    require(bool(env.get("GH_TOKEN")), "GH_TOKEN is required for the read-only GitHub API")
    require(env.get("GITHUB_REPOSITORY") == REPOSITORY, "publisher repository mismatch")
    source = env.get("GITHUB_SHA", "")
    require(SHA.fullmatch(source) and git("rev-parse", "HEAD") == source, "publisher source/checkout mismatch")
    run_id, attempt = env.get("GITHUB_RUN_ID", ""), env.get("GITHUB_RUN_ATTEMPT", "")
    require(re.fullmatch(r"[1-9][0-9]*", run_id) and re.fullmatch(r"[1-9][0-9]*", attempt),
            "actual publisher run and attempt are required")
    for name in ("GITHUB_JOB", "RUNNER_NAME", "RUNNER_ARCH"):
        require(bool(env.get(name)), "missing publisher runner identity: " + name)
    runtime_path = ROOT / "experiments/agent_painter/native/runtime.py"
    spec = importlib.util.spec_from_file_location("smol_publication_source_identity", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    source_record = runtime.source_identity(ROOT, source)
    changed = set(git("diff", "--name-only", "HEAD").splitlines())
    if operation == "capture":
        require(changed <= {RUNTIME_PATH}, "capture checkout changed outside the recorded native binary")
        if RUNTIME_PATH in changed:
            build = read_json(directory / "native-build.json")
            binary = ROOT / RUNTIME_PATH
            require(build.get("kind") == "agent-painter-source-build"
                    and build.get("source_sha") == source and build.get("source_tree") == source_record["source_tree"]
                    and build.get("native_path") == RUNTIME_PATH
                    and build.get("native_sha256") == sha256(binary)
                    and build.get("native_size") == binary.stat().st_size,
                    "changed native binary is not bound to this source build")
    else:
        require(not changed, "recovery publisher requires a clean tracked checkout")
    return {"schema_version": 1, "repository": REPOSITORY, "operation": operation,
            "source_sha": source, "source_tree": source_record["source_tree"],
            "run_id": int(run_id), "run_attempt": int(attempt), "job_id": env["GITHUB_JOB"],
            "runner_name": env["RUNNER_NAME"], "hostname": socket.gethostname(),
            "runner_environment": env["RUNNER_ENVIRONMENT"], "os": env["RUNNER_OS"], "arch": env["RUNNER_ARCH"]}


def recovery_request(path):
    request = read_json(path)
    require(type(request) is dict and set(request) == set(RECOVERY_REQUEST)
            and all(type(request[key]) is type(value) and request[key] == value
                    for key, value in RECOVERY_REQUEST.items()), "recovery request differs from the fixed producer")
    return request


def api_json(route, timeout=60):
    """gh reads GH_TOKEN itself. Never print credentials, headers or subprocess stderr."""
    result = subprocess.run(["gh", "api", "--method", "GET", route], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    require(result.returncode == 0, "GitHub API request failed: " + route)
    require(len(result.stdout) <= 16 * 1024**2, "GitHub metadata response exceeded its bound")
    value = strict_json(result.stdout.decode("utf-8"))
    require(type(value) is dict, "GitHub API object required")
    return value


def verify_run(run, source, run_id, attempt):
    require(run.get("id") == run_id and run.get("head_sha") == source and run.get("run_attempt") == attempt
            and run.get("repository", {}).get("full_name") == REPOSITORY,
            "producer run source/repository/attempt mismatch")


def producer_run(source, run_id, attempt, recovery):
    deadline = time.monotonic() + (MAX_WAIT_SECONDS if recovery else 60)
    route = f"repos/{REPOSITORY}/actions/runs/{run_id}"
    while True:
        remaining = deadline - time.monotonic()
        require(remaining > 0, "producer did not finish within the bounded 3600-second wait")
        run = api_json(route, timeout=max(1, min(60, remaining)))
        verify_run(run, source, run_id, attempt)
        if not recovery or run.get("status") == "completed":
            return run
        print(json.dumps({"event": "waiting-for-producer", "run_id": run_id, "attempt": attempt,
                          "status": run.get("status"), "remaining_seconds": int(deadline - time.monotonic())}), flush=True)
        time.sleep(max(0, min(15, deadline - time.monotonic())))


def verify_jobs(response, source, run_id, attempt, recovery):
    jobs = response.get("jobs")
    require(isinstance(jobs, list) and 0 < len(jobs) <= 100 and response.get("total_count") == len(jobs)
            and len({row.get("id") for row in jobs}) == len(jobs), "incomplete or duplicated producer job response")
    selected = {}
    for component in ("baseline", "candidate", "native"):
        name = NATIVE_JOB if component == "native" else component + " on independent CPU VM"
        matches = [job for job in jobs if job.get("name") == name]
        require(len(matches) == 1, "missing/duplicate producer job: " + component)
        job = matches[0]
        require(job.get("run_id") == run_id and job.get("run_attempt") == attempt
                and job.get("head_sha") == source and type(job.get("id")) is int and job["id"] > 0
                and type(job.get("runner_id")) is int and job["runner_id"] > 0
                and job.get("runner_group_name") == "GitHub Actions"
                and "ubuntu-24.04" in job.get("labels", []), "wrong source or hosted runner assignment")
        if component != "native":
            require(job.get("status") == "completed" and job.get("conclusion") == "success",
                    component + " model job must have completed successfully")
        else:
            steps = job.get("steps", [])
            built = [step for step in steps if step.get("name") == NATIVE_STEP]
            require(len(built) == 1 and built[0].get("status") == "completed"
                    and built[0].get("conclusion") == "success", "producer native rebuild/replay step did not succeed")
            failed = [step for step in steps if step.get("conclusion") in ("failure", "cancelled", "timed_out")]
            require(all(step.get("name") == PACKAGING_STEP and step.get("conclusion") == "failure" for step in failed),
                    "producer failed outside the preserved packaging step")
            if recovery:
                require(job.get("status") == "completed" and job.get("conclusion") in ("success", "failure")
                        and (job["conclusion"] != "failure" or bool(failed)), "producer native job has unexplained failure")
            else:
                require(job.get("status") in ("in_progress", "completed") and job.get("conclusion") in (None, "success"),
                        "capture requires the active successful native publication job")
        selected[component] = job
    require(selected["baseline"]["id"] != selected["candidate"]["id"]
            and selected["baseline"]["runner_id"] != selected["candidate"]["runner_id"],
            "model jobs do not have separate runner registrations")
    return selected


def artifact_metadata(response, component, source, run_id, attempt):
    artifacts = response.get("artifacts")
    require(isinstance(artifacts, list) and len(artifacts) <= 100 and response.get("total_count") == len(artifacts),
            "artifact response is incomplete; refusing truncated pagination")
    name = f"smol-{component}-{source}-{run_id}-{attempt}"
    matches = [item for item in artifacts if item.get("name") == name]
    require(len(matches) == 1, "missing/duplicate immutable producer artifact: " + name)
    item = matches[0]
    workflow = item.get("workflow_run", {})
    require(type(item.get("id")) is int and item["id"] > 0 and item.get("expired") is False
            and type(item.get("size_in_bytes")) is int and 0 < item["size_in_bytes"] <= MAX_BYTES
            and isinstance(item.get("digest"), str) and DIGEST.fullmatch(item["digest"])
            and workflow.get("id") == run_id and workflow.get("head_sha") == source,
            "artifact source, size, digest or expiry metadata is invalid")
    return item


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name
            and not any(ord(character) < 32 for character in name), "unsafe ZIP name")
    stripped = name[:-1] if name.endswith("/") else name
    parts = stripped.split("/")
    require(parts and all(part not in ("", ".", "..") for part in parts)
            and not PurePosixPath(stripped).is_absolute() and ":" not in parts[0], "ZIP path escapes its component")
    require(not any(part.lower() in ("final", "final.jsonl")
                    or part.lower().startswith(SEALED_CASE_PREFIXES) for part in parts), "FINAL targets/evidence are forbidden")
    return "/".join(parts)


def inspect_archive(path, component):
    """Bound expansion and verify CRC before creating any extracted evidence."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= MAX_MEMBERS, "ZIP member count exceeds its bound")
        names, files = {}, []
        total = 0
        for item in entries:
            require(item.orig_filename == item.filename, "ZIP filename contains an ambiguous NUL suffix")
            name = safe_name(item.filename)
            require(name not in names, "duplicate ZIP member or file/directory alias")
            mode = stat.S_IFMT(item.external_attr >> 16)
            require(mode in (0, stat.S_IFREG, stat.S_IFDIR) and not stat.S_ISLNK(item.external_attr >> 16),
                    "ZIP symlink or special file is forbidden")
            require(not item.flag_bits & 1 and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "encrypted or unsupported ZIP member")
            require(item.file_size >= 0 and (not item.is_dir() or item.file_size == 0), "invalid ZIP member size")
            names[name] = item.is_dir()
            if not item.is_dir():
                total += item.file_size
                require(total <= MAX_BYTES, "ZIP expansion exceeds 2 GiB")
                files.append(item)
        require(files, "empty producer ZIP")
        for name in names:
            for parent in PurePosixPath(name).parents:
                if str(parent) != ".":
                    require(names.get(str(parent), True), "ZIP file is also used as a parent directory")
        has_prefix = [item.filename.startswith("evidence/") for item in files]
        prefix = ""
        if component == "build" and any(has_prefix):
            require(all(has_prefix), "build ZIP has mixed evidence/ and other roots; no partial remapping")
            prefix = "evidence/"
        members = []
        for item in files:
            name = safe_name(item.filename[len(prefix):])
            require(not (component == "build" and name.split("/")[0] in ("publication", "producer-downloads")),
                    "build ZIP cannot supply publication custody records")
            members.append((item.filename, name, item.file_size))
        require(archive.testzip() is None, "producer ZIP CRC check failed")
    return {"member_prefix": prefix, "expanded_bytes": total, "members": members}


def safe_destination(root, relative):
    root = Path(root).absolute()
    # Check every existing ancestor, including the supplied evidence directory.
    for path in (root, *root.parents):
        require(not path.is_symlink(), "evidence root contains a symlink")
    path = root
    for part in safe_name(relative).split("/"):
        path = path / part
        require(not path.is_symlink(), "evidence destination contains a symlink")
    return path


def transfer_archive(archive_path, destination, plan, extract):
    """Capture compares original bytes; recovery creates each new file exclusively."""
    with zipfile.ZipFile(archive_path) as archive:
        for member, relative, size in plan["members"]:
            target = safe_destination(destination, relative)
            if extract:
                require(not target.exists(), "refusing to overwrite original evidence: " + str(target))
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                require(target.is_file() and target.stat().st_size == size, "archive/existing evidence size mismatch: " + relative)
            with archive.open(member) as source, target.open("xb" if extract else "rb") as output:
                count = 0
                while block := source.read(1024 * 1024):
                    count += len(block)
                    require(count <= size, "ZIP member expanded beyond its declared size")
                    if extract:
                        output.write(block)
                    else:
                        require(output.read(len(block)) == block, "archive/existing evidence bytes differ: " + relative)
                require(count == size and (extract or output.read(1) == b""), "ZIP/evidence member size changed")


def download_artifact(item, path, deadline):
    remaining = deadline - time.monotonic()
    require(remaining > 0, "artifact download time budget exhausted")
    route = f"repos/{REPOSITORY}/actions/artifacts/{item['id']}/zip"
    with path.open("xb") as stream:
        result = subprocess.run(["gh", "api", "--method", "GET", route], cwd=ROOT, stdout=stream,
                                stderr=subprocess.PIPE, timeout=max(1, min(600, remaining)), check=False)
    require(result.returncode == 0, "producer artifact download failed: " + str(item["id"]))
    actual_size, actual_hash = path.stat().st_size, sha256(path)
    require(actual_size == item["size_in_bytes"] and actual_hash == DIGEST.fullmatch(item["digest"]).group(1),
            "downloaded ZIP differs from GitHub API size/digest")
    return actual_size, actual_hash


def run(args):
    directory = args.directory.absolute()
    safe_destination(directory, "publication")
    publisher = hosted_identity(args.operation, directory)
    if args.operation == "recover":
        request = recovery_request(args.request)
        source, run_id, attempt = (request[key] for key in ("producer_source", "producer_run_id", "producer_run_attempt"))
        components = ("baseline", "candidate", "native", "build")
    else:
        source, run_id, attempt = (publisher[key] for key in ("source_sha", "run_id", "run_attempt"))
        components = ("baseline", "candidate")
    publication = directory / "publication"
    publication.mkdir(parents=True, exist_ok=False)
    args._publication_owned = True
    downloads = directory / "producer-downloads"
    downloads.mkdir(exist_ok=False)
    write_json(publication / "publisher-receipt.json", publisher)
    recovered = args.operation == "recover"
    run_record = producer_run(source, run_id, attempt, recovered)
    write_json(publication / "producer-run.json", run_record)
    jobs_path = f"/repos/{REPOSITORY}/actions/runs/{run_id}/attempts/{attempt}/jobs"
    response = api_json(jobs_path.lstrip("/") + "?per_page=100")
    identity = {"schema_version": 1, "repository": REPOSITORY, "source_sha": source,
                "run_id": run_id, "run_attempt": attempt}
    write_json(publication / "job-assignments.json", {**identity, "api_path": jobs_path, "response": response})
    jobs = verify_jobs(response, source, run_id, attempt, recovered)
    artifacts = api_json(f"repos/{REPOSITORY}/actions/runs/{run_id}/artifacts?per_page=100")
    records, plans = [], []
    total_expanded = 0
    deadline = time.monotonic() + MAX_DOWNLOAD_SECONDS
    destinations = {"baseline": directory / "baseline", "candidate": directory / "candidate",
                    "native": directory / "smol-native", "build": directory}
    for component in components:
        item = artifact_metadata(artifacts, component, source, run_id, attempt)
        archive = downloads / (component + ".zip")
        size, digest = download_artifact(item, archive, deadline)
        plan = inspect_archive(archive, component)
        total_expanded += plan["expanded_bytes"]
        require(total_expanded <= MAX_BYTES, "combined producer expansion exceeds 2 GiB")
        plans.append((archive, destinations[component], plan))
        records.append({"component": component, "artifact": item,
                        "archive_path": "producer-downloads/" + component + ".zip",
                        "archive_sha256": digest, "archive_bytes": size, "member_prefix": plan["member_prefix"]})
        print(json.dumps({"event": "producer-archive-verified", "component": component,
                          "artifact_id": item["id"], "bytes": size, "sha256": digest,
                          "member_prefix": plan["member_prefix"]}), flush=True)
    if recovered:
        targets = set()
        for _archive, destination, plan in plans:
            for _member, relative, _size in plan["members"]:
                target = safe_destination(destination, relative)
                require(target not in targets and not target.exists(), "recovery would overwrite original evidence")
                targets.add(target)
    for archive, destination, plan in plans:
        transfer_archive(archive, destination, plan, extract=recovered)
    for arm in ("baseline", "candidate"):
        environment = read_json(directory / arm / "environment/run-receipt.json")
        require(environment.get("arm") == arm and environment.get("source_sha") == source
                and environment.get("run_id") == run_id and environment.get("run_attempt") == attempt
                and environment.get("runner_name") == jobs[arm]["runner_name"]
                and environment.get("runner_environment") == "github-hosted", "artifact environment differs from its producer job")
    write_json(publication / "producer-artifacts.json", {**identity, "artifacts": records})
    print(json.dumps({"event": "publication-custody-complete", "operation": args.operation,
                      "producer_source": source, "producer_run_id": run_id, "producer_run_attempt": attempt,
                      "producer_conclusion": run_record.get("conclusion"), "publisher_source": publisher["source_sha"],
                      "components": list(components), "model_execution": False, "native_execution": False}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("capture", "recover"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "recover":
            command.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        if getattr(args, "_publication_owned", False):
            failure = args.directory / "publication/failure.json"
            if not failure.exists():
                write_json(failure, {"status": "FAILED", "type": type(error).__name__, "error": str(error),
                                    "model_execution": False, "native_execution": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
