#!/usr/bin/env python3
"""Recover the frozen Stage E provider campaign and retain original ZIP custody."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

from village_001d_recover_completed import get_bytes, safe_extract, verify_zip
import village_001e_isolated as isolated
import village_001e_recover_provider_budget as recovery
import village_001e_reallocate_provider_budget as reallocation

ROOT = Path(__file__).resolve().parents[1]
REL = Path("experiments/village-001/qualification-20261119")
WORK = ROOT / REL
CUSTODY = WORK / "provider-recovery"
WORKFLOW = ".github/workflows/village-001e-recovery.yml"
REPO = "techrote/neurounner"
REPO_ID = 1432134981
BRANCH = "work/village-001e-qualification-20261119"
ORIGINAL_RUN = 44439359038
ORIGINAL_HEAD = "7308271015bf8cc0ff98e0d35ad68314e65db721"
PROVIDER_STEP = "Complete only missing provider fields within the frozen global ceiling"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def gh(resource):
    return json.loads(get_bytes(f"https://api.github.com/repos/{REPO}/" + resource, os.environ.get("GH_TOKEN", "")))


def request():
    r = read(WORK / "recovery-request.json")
    expected = {"schema", "publish_branch", "execution_revision", "original_run_id", "original_source_head", "manifest_sha256", "resume_recovery_run_id", "new_training", "new_local_generations", "release_tag"}
    require(set(r) in (expected, expected | {"allocation_manifest_sha256"}), "unexpected recovery request fields")
    require(r["schema"] == "village-001e-bounded-provider-recovery-request-v1" and r["publish_branch"] == BRANCH, "wrong recovery request")
    require(r["original_run_id"] == ORIGINAL_RUN and r["original_source_head"] == ORIGINAL_HEAD, "original campaign changed")
    require(type(r["execution_revision"]) is int and r["execution_revision"] > 0, "invalid recovery revision")
    require(r["resume_recovery_run_id"] is None or (type(r["resume_recovery_run_id"]) is int and r["resume_recovery_run_id"] > 0), "invalid recovery resume run")
    require(r["new_training"] is False and r["new_local_generations"] == 0, "recovery cannot run local models")
    require(r["release_tag"] == isolated.FIXED["release_tag"], "qualification release changed")
    manifest_path = CUSTODY / "manifest.json"
    require(recovery.sha256_file(manifest_path) == r["manifest_sha256"], "frozen recovery manifest changed")
    manifest, config = recovery.validate_manifest(manifest_path)
    run = read(CUSTODY / "original-run.json")
    inventory = read(CUSTODY / "original-artifacts.json")
    require(manifest["original_run"] == run and run["id"] == ORIGINAL_RUN and run["head_sha"] == ORIGINAL_HEAD, "original run metadata changed")
    require(recovery.sha256_file(CUSTODY / "original-run.json") == manifest["run_record_sha256"], "run receipt hash changed")
    require(recovery.sha256_file(CUSTODY / "original-artifacts.json") == manifest["artifact_inventory_sha256"], "artifact receipt hash changed")
    require({a["name"] for a in inventory["artifacts"]} == {row["artifact_name"] for row in manifest["lanes"] if row.get("artifact") is not None}, "uploaded original artifact set differs from manifest")
    require(recovery.sha256_file(CUSTODY / "original-jobs.json") == manifest["jobs_record_sha256"], "original job-step proof changed")
    allocation_context(r, manifest, config)
    isolated.request()
    isolated.verify_generation()
    return r, manifest, config


def allocation_context(r, manifest=None, config=None):
    """An explicit extension augments the original manifest; it never replaces it."""
    if "allocation_manifest_sha256" not in r:
        return None
    digest = r["allocation_manifest_sha256"]
    require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "invalid allocation manifest SHA")
    require(r["execution_revision"] >= 2 and r["resume_recovery_run_id"] is not None, "allocation extension requires a preserved prior execution")
    path = CUSTODY / "allocation-extension.json"
    require(recovery.sha256_file(path) == digest, "allocation manifest bytes changed")
    allocation, original, allocation_config = reallocation.load_allocation(path)
    require(allocation["original_manifest"]["sha256"] == r["manifest_sha256"], "allocation extension changed original budget authority")
    if manifest is not None:
        require(original == manifest, "allocation extension loaded another original manifest")
    if config is not None:
        require(allocation_config.resolve() == config.resolve(), "allocation extension changed qualification config")
    baseline = allocation["prior_recovery_run"]
    check_run(baseline, run_id=baseline["id"], workflow=WORKFLOW)
    return path, allocation, original, allocation_config


def verify():
    r, manifest, _ = request()
    require(os.environ.get("GITHUB_REPOSITORY") == REPO and os.environ.get("GITHUB_REF_NAME") == BRANCH, "wrong repository or publication branch")
    require(os.environ.get("GITHUB_EVENT_NAME") == "push" and os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "live recovery requires a new explicit owned-branch request")
    remote = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True).split()[0]
    require(remote == os.environ["GITHUB_SHA"], "owned branch moved before recovery")
    subprocess.run(["git", "merge-base", "--is-ancestor", ORIGINAL_HEAD, "HEAD"], check=True)
    verify_run_history(r)
    tag = subprocess.check_output(["git", "ls-remote", "--tags", "origin", "refs/tags/" + r["release_tag"]], text=True)
    require(not tag.strip(), "immutable qualification tag already exists")
    release = subprocess.run(["gh", "api", f"repos/{REPO}/releases/tags/{r['release_tag']}"], capture_output=True, text=True)
    require(release.returncode != 0 and "HTTP 404" in release.stderr, "qualification release exists or absence is uncertain")
    active = allocation_context(r, manifest)
    plan = active[1] if active else manifest
    status_key = "baseline_status" if active else "status"
    print(json.dumps({"original_run": ORIGINAL_RUN, "pending_lanes": [x["lane"] for x in plan["lanes"] if x[status_key] == "pending"], "budget": plan["budget"]}, indent=2))


def paged(resource, key):
    rows = []
    for page in range(1, 11):
        data = gh(resource + ("&" if "?" in resource else "?") + f"per_page=100&page={page}")
        rows.extend(data[key])
        if len(rows) >= data["total_count"]:
            break
    require(len(rows) == data["total_count"] and len({row["id"] for row in rows}) == len(rows), "incomplete/duplicate GitHub history")
    return rows


def verify_run_history(r):
    """Never reset supplemental spending by pushing another fresh request."""
    current_id = int(os.environ["GITHUB_RUN_ID"])
    runs = paged(f"actions/workflows/{Path(WORKFLOW).name}/runs?branch={quote(BRANCH, safe='')}&event=push", "workflow_runs")
    current = [x for x in runs if x["id"] == current_id]
    require(len(current) == 1 and current[0]["head_sha"] == os.environ["GITHUB_SHA"], "current run absent from authoritative recovery history")
    older = sorted((x for x in runs if x["id"] != current_id), key=lambda x: x["run_number"])
    require(all(x["run_number"] < current[0]["run_number"] and x["status"] == "completed" for x in older), "another recovery execution is pending or newer")
    spent_runs = []
    allocated_spending = False
    for run in older:
        check_run(run, run_id=run["id"], workflow=WORKFLOW)
        subprocess.run(["git", "merge-base", "--is-ancestor", run["head_sha"], "HEAD"], check=True)
        previous = json.loads(subprocess.check_output(["git", "show", run["head_sha"] + ":" + str(REL / "recovery-request.json")], text=True))
        require(previous.get("manifest_sha256") == r["manifest_sha256"] and previous.get("original_run_id") == ORIGINAL_RUN, "prior recovery used another budget manifest")
        if previous.get("allocation_manifest_sha256") is not None:
            require(previous["allocation_manifest_sha256"] == r.get("allocation_manifest_sha256"), "active allocation extension cannot be dropped or replaced")
        jobs = paged(f"actions/runs/{run['id']}/jobs", "jobs")
        recovery_jobs = [job for job in jobs if job["name"] == "recover"]
        require(len(recovery_jobs) <= 1, "ambiguous prior recovery job")
        if recovery_jobs and recovery_jobs[0]["conclusion"] != "skipped":
            steps = [s for s in recovery_jobs[0].get("steps", []) if s["name"] == PROVIDER_STEP]
            require(len(steps) == 1, "prior provider execution cannot be established")
            if steps[0].get("conclusion") != "skipped":
                spent_runs.append(run)
                allocated_spending = allocated_spending or previous.get("allocation_manifest_sha256") is not None
    if older:
        previous = json.loads(subprocess.check_output(["git", "show", older[-1]["head_sha"] + ":" + str(REL / "recovery-request.json")], text=True))
        require(r["execution_revision"] == previous["execution_revision"] + 1, "recovery revision must advance exactly once from latest request")
    else:
        require(r["execution_revision"] == 1, "first recovery revision must be one")
    expected_resume = spent_runs[-1]["id"] if spent_runs else None
    require(r["resume_recovery_run_id"] == expected_resume, "must resume the latest earlier provider execution; fresh recovery would reset campaign spending")
    active = allocation_context(r)
    if active and not allocated_spending:
        require(active[1]["prior_recovery_run"]["id"] == expected_resume, "first allocation execution must resume its exact frozen baseline run")
    print(json.dumps({"prior_recovery_runs": len(older), "prior_provider_executions": len(spent_runs), "required_resume_run_id": expected_resume}))


def check_run(run, *, run_id, workflow, head=None):
    require(run.get("id") == run_id and run.get("path") == workflow and run.get("head_branch") == BRANCH and run.get("status") == "completed", "resume run identity/status mismatch")
    require(run.get("run_attempt") == 1, "resume run must have one unambiguous attempt")
    require(all(run.get(key, {}).get("id") == REPO_ID and run[key].get("full_name") == REPO for key in ("repository", "head_repository")), "resume run repository mismatch")
    require(re.fullmatch(r"[0-9a-f]{40}", run.get("head_sha", "")), "invalid resume source SHA")
    if head is not None:
        require(run["head_sha"] == head, "resume source head mismatch")


def artifact_inventory(run_id):
    rows = []
    for page in range(1, 5):
        data = gh(f"actions/runs/{run_id}/artifacts?per_page=100&page={page}")
        rows.extend(data["artifacts"])
        if len(rows) >= data["total_count"]:
            break
    require(len(rows) == data["total_count"] and len({a["id"] for a in rows}) == len(rows), "incomplete/duplicate remote artifact inventory")
    return rows


def download_artifact(artifact, *, run_id, head, archive, out):
    require(not artifact.get("expired") and 0 < artifact["size_in_bytes"] < 128 * 1024 * 1024, "expired or oversized recovery artifact")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest", "")), "artifact SHA unavailable")
    require(artifact["workflow_run"]["id"] == run_id and artifact["workflow_run"]["head_sha"] == head, "artifact source differs from frozen run")
    url = f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact['id']}/zip"
    require(artifact["archive_download_url"] == url, "artifact endpoint mismatch")
    require(not archive.exists() and not out.exists(), "recovery destination must be new")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(get_bytes(url, os.environ.get("GH_TOKEN", ""), artifact["size_in_bytes"]))
    verify_zip(archive, artifact)
    return safe_extract(archive, out)


def download_original(out):
    _, manifest, config = request()
    require(not out.exists(), "original download directory must be new")
    run = gh(f"actions/runs/{ORIGINAL_RUN}")
    check_run(run, run_id=ORIGINAL_RUN, workflow=recovery.ORIGINAL_WORKFLOW, head=ORIGINAL_HEAD)
    current = {a["name"]: a for a in artifact_inventory(ORIGINAL_RUN)}
    recorded = read(CUSTODY / "original-artifacts.json")
    for artifact in recorded["artifacts"]:
        require(artifact["name"] in current, "frozen original artifact disappeared")
        remote = current[artifact["name"]]
        for key in ("id", "name", "digest", "size_in_bytes", "archive_download_url", "workflow_run"):
            require(remote[key] == artifact[key], "frozen artifact changed: " + key)
        require(not remote["expired"], "frozen artifact has expired")
    for artifact in recorded["artifacts"]:
        name = artifact["name"]
        download_artifact(artifact, run_id=ORIGINAL_RUN, head=ORIGINAL_HEAD, archive=out / "archives" / (name + ".zip"), out=out / "artifacts" / name)
        print("Verified original artifact " + name, flush=True)
    args = argparse.Namespace(config=config, artifacts=out / "artifacts", archives=out / "archives", run_record=CUSTODY / "original-run.json", artifact_inventory=CUSTODY / "original-artifacts.json", jobs_record=CUSTODY / "original-jobs.json")
    require(recovery.make_manifest(args) == manifest, "downloaded original campaign does not reconstruct the frozen recovery manifest")


def download_prior(out):
    r, _, _ = request()
    run_id = r["resume_recovery_run_id"]
    if run_id is None:
        print("No previous bounded recovery requested")
        return
    run = gh(f"actions/runs/{run_id}")
    check_run(run, run_id=run_id, workflow=WORKFLOW)
    subprocess.run(["git", "fetch", "--no-tags", "origin", run["head_sha"]], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", run["head_sha"], "HEAD"], check=True)
    expected_name = f"village001e-budget-recovery-{run_id}"
    matches = [a for a in artifact_inventory(run_id) if a["name"] == expected_name]
    require(len(matches) == 1, "previous recovery artifact missing or ambiguous")
    artifact = matches[0]
    files = download_artifact(artifact, run_id=run_id, head=run["head_sha"], archive=out.parent / "prior-recovery.zip", out=out)
    recovery.write_once(CUSTODY / f"resume-recovery-{run_id}.json", {"schema": "village-001e-recovery-artifact-custody-v1", "run": run, "artifact": artifact, "files": files})
    if (out / "custody").exists():
        recovery.tree(out / "custody")
        for receipt in (out / "custody").glob("resume-recovery-*.json"):
            recovery.write_once(CUSTODY / receipt.name, read(receipt))


def run_all(original, out, prior=None):
    r, manifest, config = request()
    require(not out.exists(), "recovery output must be new")
    manifest_path = CUSTODY / "manifest.json"
    active = allocation_context(r, manifest, config)
    rows = active[1]["lanes"] if active else manifest["lanes"]
    original_rows = {row["artifact_name"]: row for row in manifest["lanes"]}
    sources = {}
    require((r["resume_recovery_run_id"] is None) == (prior is None), "declared prior recovery must be supplied, and unsolicited prior recovery is refused")
    if prior is not None:
        require(prior.is_dir() and not prior.is_symlink(), "prior recovery root is missing or nonregular")
        receipt_path = CUSTODY / f"resume-recovery-{r['resume_recovery_run_id']}.json"
        require(receipt_path.is_file(), "prior recovery custody receipt missing")
        receipt = read(receipt_path)
        check_run(receipt["run"], run_id=r["resume_recovery_run_id"], workflow=WORKFLOW)
        require(receipt["artifact"]["workflow_run"]["id"] == r["resume_recovery_run_id"], "prior artifact receipt belongs to another run")
        recorded = {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in receipt["files"]}
        require(recovery.tree(prior) == recorded, "prior recovery bytes do not match verified ZIP receipt")
        allowed = {"inherited", "lanes", "custody", "recovery-failures.json"}
        require({p.name for p in prior.iterdir()} <= allowed, "unexpected prior recovery evidence")
        expected = {row["artifact_name"] for row in rows}
        require({p.name for p in (prior / "inherited").iterdir()} == expected, "prior inherited forest must retain all 48 lanes")
        if (prior / "lanes").exists():
            require({p.name for p in (prior / "lanes").iterdir()} <= expected, "unexpected prior output lane")
        if active:
            baseline = active[1]["prior_recovery_run"]
            if r["resume_recovery_run_id"] == baseline["id"]:
                baseline_receipt = receipt
                baseline_zip = prior.parent / "prior-recovery.zip"
                reallocation.validate_baseline_bundle(prior, active[1])
            else:
                baseline_receipt = read(prior / "custody" / f"resume-recovery-{baseline['id']}.json")
                baseline_zip = prior / "custody" / f"recovery-run-{baseline['id']}.zip"
            check_run(baseline_receipt["run"], run_id=baseline["id"], workflow=WORKFLOW, head=baseline["head_sha"])
            frozen_artifact = active[1]["prior_recovery_artifact"]
            for key in ("id", "name", "digest", "size_in_bytes", "archive_download_url", "workflow_run"):
                require(baseline_receipt["artifact"][key] == frozen_artifact[key], "allocation baseline artifact custody changed: " + key)
            verify_zip(baseline_zip, frozen_artifact)
    # Preflight every source before spending any provider request.
    for row in rows:
        path = original / row["artifact_name"]
        if prior is not None:
            path = prior / "lanes" / row["artifact_name"]
            if path.exists():
                inherited = recovery.tree(prior / "inherited" / row["artifact_name"])
                current = recovery.tree(path)
                require(all(current.get(name) == value for name, value in inherited.items()), "prior output lost immutable inherited history; refusing budget reset")
            else:
                path = prior / "inherited" / row["artifact_name"]
        if active:
            reallocation.validate_prior_source(path, active[1], row, original=manifest)
        else:
            recovery.preserve_original(path, row)
        existing = path / "budget-recovery-identity.json"
        if existing.exists():
            require(read(existing) == recovery.recovery_identity(manifest_path, manifest, original_rows[row["artifact_name"]]), "prior recovery belongs to another manifest")
        sources[row["artifact_name"]] = path
    # A cancellation halfway through this invocation must retain source lanes
    # that were completed in older invocations but not yet visited in this one.
    # Copy the entire inherited forest before the first new provider request.
    for row in rows:
        name = row["artifact_name"]
        shutil.copytree(sources[name], out / "inherited" / name)
    if prior is not None:
        if (prior / "custody").exists():
            shutil.copytree(prior / "custody", out / "custody")
        (out / "custody").mkdir(exist_ok=True)
        prior_zip = prior.parent / "prior-recovery.zip"
        receipt_path = CUSTODY / f"resume-recovery-{r['resume_recovery_run_id']}.json"
        verify_zip(prior_zip, read(receipt_path)["artifact"])
        shutil.copy2(prior_zip, out / "custody" / f"recovery-run-{r['resume_recovery_run_id']}.zip")
        shutil.copy2(receipt_path, out / "custody" / receipt_path.name)
    failures = []
    for row in rows:
        name = row["artifact_name"]
        try:
            if active:
                reallocation.run_lane(*active, row, out / "inherited" / name, out / "lanes" / name, offline=row["baseline_status"] == "completed")
            else:
                recovery.run_lane(manifest_path, manifest, config, row, out / "inherited" / name, out / "lanes" / name, offline=row["status"] == "completed")
        except Exception as exc:
            failures.append({"lane": row["lane"], "error": type(exc).__name__ + ": " + str(exc)})
            print(json.dumps(failures[-1]), flush=True)
            # Validate the formatting continuation on already-started lanes
            # before permitting untouched tournament lanes to spend requests.
            if row.get("kind") == "tournament" and row["baseline_status" if active else "status"] == "pending":
                break
    if failures:
        recovery.write_once(out / "recovery-failures.json", failures)
        raise RuntimeError("bounded recovery has incomplete lanes; original and new raw files are retained")
    result = reallocation.verify_complete(*active, out / "lanes") if active else recovery.verify_complete(manifest_path, manifest, config, out / "lanes")
    recovery.write_once(CUSTODY / "accounting.json", result)
    print(json.dumps({k: result[k] for k in ("actual_logical_requests", "actual_supplemental_logical_requests", "all_48_lanes_reconstructed_offline")}, indent=2))


def layout(source, destination):
    r, manifest, config = request()
    require(not destination.exists(), "assembly layout must be new")
    active = allocation_context(r, manifest, config)
    if active:
        reallocation.verify_complete(*active, source)
    else:
        recovery.verify_complete(CUSTODY / "manifest.json", manifest, config, source)
    for row in manifest["lanes"]:
        group = "tournament" if row["kind"] == "tournament" else "ecology-judgments"
        shutil.copytree(source / row["artifact_name"], destination / group / row["artifact_name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    for command in ("download-original", "download-prior"):
        p = sub.add_parser(command)
        p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("run-all")
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--prior", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("layout")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    elif args.command == "download-original":
        download_original(args.out)
    elif args.command == "download-prior":
        download_prior(args.out)
    elif args.command == "run-all":
        run_all(args.original, args.out, args.prior)
    else:
        layout(args.source, args.out)


if __name__ == "__main__":
    main()
