#!/usr/bin/env python3
"""Recover the completed VILLAGE-001D population without running any models.

This deliberately accepts only the original failed run whose sole incomplete
lane was the V-G12-B before/after comparison. Original ZIPs remain immutable;
extracted working copies are separate from their retained custody records.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPOSITORY = "techrote/neurounner"
REPOSITORY_ID = 1432134981
SOURCE_RUN = 40613393744
SOURCE_HEAD = "388ad349d48102f6266c0c0c837efec0275e38c9"
WORKFLOW_PATH = ".github/workflows/village-001d-preference.yml"
FAILED_MEMBER = "V-G12-B"
FAILED_JOB = 122582069277
MEMBERS = tuple(f"V-G{i:02d}-{s}" for i in range(1, 13) for s in "ABC")
STAGES = {"feedback": "feedback", "pairs": "pairs", "preference": "preference", "compare": "comparison"}
MAX_ZIP_BYTES = 128 * 1024 * 1024
MAX_EXTRACTED_BYTES = 256 * 1024 * 1024


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_source_run(run: dict) -> None:
    for key, value in {
        "id": SOURCE_RUN, "head_sha": SOURCE_HEAD, "head_branch": "main",
        "path": WORKFLOW_PATH, "status": "completed", "conclusion": "failure",
        "event": "workflow_dispatch", "run_attempt": 1,
    }.items():
        require(run.get(key) == value, f"source run {key} mismatch")
    for key in ("repository", "head_repository"):
        require(run.get(key, {}).get("id") == REPOSITORY_ID, f"source {key} ID mismatch")
        require(run[key].get("full_name") == REPOSITORY, f"source {key} name mismatch")
    require(run.get("head_commit", {}).get("id") == SOURCE_HEAD, "source head commit mismatch")


def validate_jobs(jobs: list[dict]) -> None:
    expected = {"plan": "success", "assemble": "skipped"}
    for stage in ("feedback", "compile_pairs", "preference_train", "compare"):
        expected.update({f"{stage} ({m})": "success" for m in MEMBERS})
    expected[f"compare ({FAILED_MEMBER})"] = "failure"
    require(len(jobs) == len(expected), "source job count mismatch")
    require(len({j.get("id") for j in jobs}) == len(jobs), "duplicate source job ID")
    require(len({j.get("name") for j in jobs}) == len(jobs), "duplicate source job name")
    require({j.get("name") for j in jobs} == set(expected), "source job names mismatch")
    for job in jobs:
        require(job.get("run_id") == SOURCE_RUN and job.get("head_sha") == SOURCE_HEAD,
                "source job provenance mismatch")
        require(job.get("run_attempt") == 1 and job.get("status") == "completed",
                "source job attempt/status mismatch")
        require(job.get("conclusion") == expected[job["name"]],
                f"source job conclusion mismatch: {job['name']}")
        if job["name"] == f"compare ({FAILED_MEMBER})":
            require(job.get("id") == FAILED_JOB, "failed comparison job ID mismatch")


def validate_inventory(artifacts: list[dict], now: datetime | None = None, *,
                       require_unexpired: bool = True) -> list[dict]:
    expected = {f"village001d-{stage}-{member}" for stage in STAGES for member in MEMBERS}
    require(isinstance(artifacts, list) and len(artifacts) == 144, "expected 144 original artifacts")
    names = [a.get("name") for a in artifacts]
    require(len(set(names)) == 144 and set(names) == expected, "missing, extra or duplicate artifact name")
    require(len({a.get("id") for a in artifacts}) == 144, "duplicate artifact ID")
    now = now or datetime.now(timezone.utc)
    for artifact in artifacts:
        aid = artifact.get("id")
        require(type(aid) is int and aid > 0, "invalid artifact ID")
        require(type(artifact.get("expired")) is bool, "invalid artifact expired flag")
        try:
            expiry = datetime.fromisoformat(artifact["expires_at"].replace("Z", "+00:00"))
            require(expiry.tzinfo is not None, "artifact expiry must be timezone-aware")
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid artifact expiry") from exc
        # Retention expiry prevents a new GitHub download, not verification of
        # an already preserved, digest-identical original ZIP.
        if require_unexpired:
            require(artifact["expired"] is False and expiry > now,
                    f"expired artifact: {artifact['name']}")
        require(re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest", "")) is not None,
                "missing/invalid original artifact digest")
        require(type(artifact.get("size_in_bytes")) is int and 0 < artifact["size_in_bytes"] <= MAX_ZIP_BYTES,
                "invalid artifact size")
        require(artifact.get("archive_download_url") ==
                f"https://api.github.com/repos/{REPOSITORY}/actions/artifacts/{aid}/zip",
                "artifact download URL mismatch")
        source = artifact.get("workflow_run", {})
        for key, value in {"id": SOURCE_RUN, "repository_id": REPOSITORY_ID,
                           "head_repository_id": REPOSITORY_ID, "head_branch": "main",
                           "head_sha": SOURCE_HEAD}.items():
            require(source.get(key) == value, f"artifact source {key} mismatch")
    return sorted(artifacts, key=lambda a: a["name"])


def verify_zip(path: Path, artifact: dict) -> bytes:
    require(path.is_file() and not path.is_symlink(), f"missing original ZIP: {path.name}")
    require(path.stat().st_size == artifact["size_in_bytes"], f"ZIP byte size mismatch: {path.name}")
    data = path.read_bytes()
    require(sha256(data) == artifact["digest"].removeprefix("sha256:"),
            f"ZIP SHA-256 mismatch: {path.name}")
    return data


def safe_extract(path: Path, destination: Path) -> list[dict]:
    """Preflight every member before extracting anything; never follow links."""
    require(not destination.exists() and not destination.is_symlink(), "extraction destination already exists")
    records = []
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= 10000, "invalid ZIP member count")
        require(sum(e.file_size for e in entries) <= MAX_EXTRACTED_BYTES, "ZIP expands beyond bound")
        seen: dict[str, bool] = {}
        for entry in entries:
            name = entry.filename
            parts = name.rstrip("/").split("/")
            require(name and "\\" not in name and "\x00" not in name and ":" not in name,
                    "unsafe ZIP member name")
            require(not PurePosixPath(name).is_absolute() and all(p not in ("", ".", "..") for p in parts),
                    "ZIP path traversal")
            normalized = "/".join(parts)
            require(normalized not in seen, "duplicate ZIP member path")
            mode = stat.S_IFMT(entry.external_attr >> 16)
            require(mode in (0, stat.S_IFREG, stat.S_IFDIR), "ZIP symlink or special file refused")
            require(not (entry.flag_bits & 1), "encrypted ZIP member refused")
            require(mode != stat.S_IFDIR or entry.is_dir(), "ZIP directory mode/name mismatch")
            seen[normalized] = entry.is_dir()
        for name in seen:
            parts = name.split("/")
            for length in range(1, len(parts)):
                parent = "/".join(parts[:length])
                require(parent not in seen or seen[parent], "ZIP file/directory collision")
        destination.mkdir(parents=True)
        for entry in entries:
            target = destination.joinpath(*entry.filename.rstrip("/").split("/"))
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(entry)  # Also verifies the container's CRC.
            require(len(data) == entry.file_size, "ZIP member byte count mismatch")
            with target.open("xb") as stream:
                stream.write(data)
            records.append({"path": entry.filename, "bytes": len(data), "sha256": sha256(data)})
    return sorted(records, key=lambda r: r["path"])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get_bytes_once(url: str, token: str, limit: int) -> bytes:
    require(url.startswith(f"https://api.github.com/repos/{REPOSITORY}/actions/"), "unexpected API URL")
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28",
               "User-Agent": "neurounner-village-evidence-recovery"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, headers=headers), timeout=90)
    except urllib.error.HTTPError as exc:
        if exc.code not in (301, 302, 303, 307, 308):
            raise
        target = exc.headers.get("Location", "")
        parsed = urllib.parse.urlsplit(target)
        require(parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password,
                "unsafe artifact redirect")
        # Signed artifact storage receives no GitHub token or API headers.
        response = urllib.request.urlopen(target, timeout=90)
    with response:
        data = response.read(limit + 1)
    require(len(data) <= limit, "download exceeds size bound")
    return data


def get_bytes(url: str, token: str, limit: int = MAX_ZIP_BYTES) -> bytes:
    """Retry bounded transient GET failures without changing custody checks."""
    for attempt in range(1, 5):
        try:
            return _get_bytes_once(url, token, limit)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f"artifact API/storage GET failed with HTTP {error.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
            if attempt == 4:
                raise RuntimeError("artifact API/storage transport failed after four attempts") from None
        time.sleep(2 ** (attempt - 1))
    raise RuntimeError("unreachable download retry state")


def fetch_collection(kind: str, token: str) -> list[dict]:
    rows = []
    for page in range(1, 4):
        url = f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{SOURCE_RUN}/{kind}?per_page=100&page={page}"
        result = json.loads(get_bytes(url, token))
        rows.extend(result[kind])
        if len(rows) >= result["total_count"]:
            require(len(rows) == result["total_count"], "GitHub collection count mismatch")
            return rows
    raise ValueError("unexpected source pagination count")


def recover(out: Path, run: dict, artifacts: list[dict], jobs: list[dict], *,
            existing_zips: Path | None = None, token: str = "") -> dict:
    validate_source_run(run)
    validate_jobs(jobs)
    artifacts = validate_inventory(artifacts, require_unexpired=existing_zips is None)
    require(not out.exists() and not out.is_symlink(), "recovery destination must be fresh")
    custody = out / "custody"
    (custody / "zips").mkdir(parents=True)
    metadata = {"source-run.json": run, "source-artifacts.json": artifacts, "source-jobs.json": jobs}
    for name, value in metadata.items():
        dump(custody / name, value)
    records = []
    for artifact in artifacts:
        name = artifact["name"]
        zip_path = custody / "zips" / f"{name}.zip"
        if existing_zips is not None:
            source = existing_zips / zip_path.name
            verify_zip(source, artifact)
            shutil.copyfile(source, zip_path)
        else:
            data = get_bytes(artifact["archive_download_url"], token, artifact["size_in_bytes"])
            zip_path.write_bytes(data)
        verify_zip(zip_path, artifact)
        stage, member = name.removeprefix("village001d-").split("-", 1)
        destination = out / STAGES[stage] / member
        files = safe_extract(zip_path, destination)
        required = {"feedback": ("manifest.json", "feedback.jsonl"),
                    "pairs": ("manifest.json", "pairs.jsonl"),
                    "preference": ("result.json",), "compare": ("evaluation.json",)}[stage]
        if stage == "compare" and member == FAILED_MEMBER:
            require(not (destination / "evaluation.json").exists(), "failed comparison unexpectedly terminal")
            require(any(f["path"].startswith("provider/") for f in files), "failed comparison lacks raw provider evidence")
        else:
            require(all((destination / p).is_file() for p in required), f"missing completed {stage} evidence: {member}")
        records.append({"id": artifact["id"], "name": name, "original_zip": f"zips/{zip_path.name}",
                        "bytes": artifact["size_in_bytes"], "sha256": artifact["digest"][7:],
                        "working_copy": f"{STAGES[stage]}/{member}", "files": files})
    manifest = {"schema": "village-001d-completed-recovery-v1", "source_repository": REPOSITORY,
                "source_run_id": SOURCE_RUN, "source_head": SOURCE_HEAD,
                "failed_comparison_member": FAILED_MEMBER, "original_artifacts": 144,
                "custody_mode": "verified_existing_zips" if existing_zips is not None else "github_download",
                "new_optimizer_steps": 0, "new_model_calls": 0, "new_provider_calls": 0,
                "recovery_script_sha256": sha256(Path(__file__).read_bytes()),
                "metadata": {name: {"bytes": (custody / name).stat().st_size,
                                     "sha256": sha256((custody / name).read_bytes())} for name in metadata},
                "artifacts": records}
    dump(custody / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--existing-zips", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-artifacts", type=Path)
    parser.add_argument("--source-jobs", type=Path)
    args = parser.parse_args()
    supplied = (args.source_run, args.source_artifacts, args.source_jobs)
    require(all(supplied) or not any(supplied), "supply all three source metadata files or none")
    token = os.environ.get("GH_TOKEN", os.environ.get("GITHUB_TOKEN", ""))
    if all(supplied):
        run, artifacts, jobs = (json.loads(p.read_bytes()) for p in supplied)
    else:
        run = json.loads(get_bytes(f"https://api.github.com/repos/{REPOSITORY}/actions/runs/{SOURCE_RUN}", token))
        artifacts, jobs = fetch_collection("artifacts", token), fetch_collection("jobs", token)
    result = recover(args.out, run, artifacts, jobs, existing_zips=args.existing_zips, token=token)
    print(json.dumps({"original_artifacts": result["original_artifacts"],
                      "new_optimizer_steps": 0, "new_provider_calls": 0,
                      "manifest": str(args.out / "custody" / "manifest.json")}))


if __name__ == "__main__":
    main()
