#!/usr/bin/env python3
"""Bounded operator for post-E causal diagnostics; never changes D/E evidence."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from village_001d_recover_completed import get_bytes, safe_extract, verify_zip
from village_001f_worker_diagnostic import checked_adapter, load_contract
import village_001f_worker_diagnostic as worker

ROOT = Path(__file__).resolve().parents[1]
REL = Path("experiments/village-001/diagnostic-20261119")
WORK = ROOT / REL
BRANCH = "work/village-001f-causal-diagnostics-20261119"
REPO = "techrote/neurounner"
WORKFLOW = ".github/workflows/village-001f-diagnostics.yml"
INPUT_TAG = "village-001f-pilot-inputs-20261119"
FINAL_TAG = "village-001f-mask-sprocket-20261119"
MEMBERS = ["V-G01-A", "V-G03-B", "V-G05-C", "V-G07-A", "V-G09-B", "V-G11-C"]
ARMS = ["historical_token_id_mask", "length_derived_mask"]
E_SUMMARY = "experiments/village-001/final-evidence/summary.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".ckpt")
CONTAINER_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".npy", ".npz")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def files(root: Path) -> dict:
    require(root.is_dir() and not root.is_symlink(), "missing or nonregular custody root")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "nonregular custody entry")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    return result


def evidence_files(root: Path) -> dict:
    inventory = files(root)
    for relative in inventory:
        path = Path(relative)
        require(not relative.lower().endswith(WEIGHT_SUFFIXES + CONTAINER_SUFFIXES), "model weights/checkpoints and binary containers must remain outside Git evidence")
        require(not any(part.lower() in {"adapter", "adapters", "checkpoint", "checkpoints", "models"} for part in path.parts[:-1]), "model directory must remain outside Git evidence")
    return inventory


def request() -> dict:
    r = json.loads((WORK / "request.json").read_text())
    require(r.get("schema") == "village-001f-post-e-request-v1" and r.get("publish_branch") == BRANCH, "wrong diagnostic request")
    require(r.get("phase") in ("generate", "judge"), "wrong diagnostic phase")
    require(type(r.get("execution_revision")) is int and r["execution_revision"] > 0, "invalid revision")
    require(r.get("input_release_tag") == INPUT_TAG and r.get("final_release_tag") == FINAL_TAG, "release target drift")
    require(r.get("maximum_training_lanes") == 12 and r.get("maximum_optimizer_steps") == 1440, "training budget drift")
    require(r.get("new_worker_generations") == 300 and r.get("new_sprocket_generations") == 36, "generation budget drift")
    require(r.get("maximum_provider_logical_requests") == 114, "judge budget drift")
    require(r.get("resume_run_id") is None or (type(r["resume_run_id"]) is int and r["resume_run_id"] > 0), "invalid recovery run")
    require(re.fullmatch(r"[0-9a-f]{40}", r.get("stage_e_main_commit", "")), "missing accepted E commit")
    require(sha(ROOT / E_SUMMARY) == r.get("stage_e_summary_sha256"), "accepted E evidence changed")
    for name, expected in r.get("contract_sha256", {}).items():
        require(name in {"worker-config.json", "worker-eval.json", "sprocket-config.json", "judge-config.json"}, "unexpected contract file")
        require(sha(WORK / name) == expected, "frozen diagnostic contract changed: " + name)
    require(set(r.get("contract_sha256", {})) == {"worker-config.json", "worker-eval.json", "sprocket-config.json", "judge-config.json"}, "contract inventory incomplete")
    cfg, _, _, _ = load_contract(WORK / "worker-config.json")
    require(cfg["members"] == MEMBERS and cfg["arms"] == ARMS, "panel or arms changed")
    return r


def gh(resource: str):
    result = subprocess.run(["gh", "api", f"repos/{REPO}/" + resource], capture_output=True, text=True)
    require(result.returncode == 0, "GitHub read failed: " + resource)
    return json.loads(result.stdout)


def absent_tag(tag: str) -> None:
    require(not subprocess.check_output(["git", "ls-remote", "--tags", "origin", "refs/tags/" + tag], text=True).strip(), "immutable tag already exists: " + tag)
    result = subprocess.run(["gh", "api", f"repos/{REPO}/releases/tags/{tag}"], capture_output=True, text=True)
    require(result.returncode != 0 and "HTTP 404" in result.stderr, "release exists or absence unavailable")


def verify_execution_history(r: dict) -> None:
    """Never reset a phase's model budget through a new push or Actions rerun."""
    require(os.environ.get("GITHUB_RUN_ATTEMPT") == "1", "Actions reruns are disabled; resume from a new owned-branch request")
    run_id = int(os.environ["GITHUB_RUN_ID"])
    inventory = gh("actions/workflows/village-001f-diagnostics.yml/runs?branch=" + BRANCH + "&event=push&per_page=100")
    runs = inventory.get("workflow_runs", [])
    require(type(inventory.get("total_count")) is int and inventory["total_count"] == len(runs) <= 100, "complete bounded diagnostic run history required")
    spending, required_artifacts = [], set()
    step_names = ({"Train one fixed 120-step matched arm", "Generate fixed Q BASE control",
        "Compare four fixed model stages on the same Q reserve", "Run 36 bounded native prompt comparisons"}
        if r["phase"] == "generate" else {"Blindly measure causal effects and feedback grounding"})
    job_names = ("train", "base", "infer", "sprocket") if r["phase"] == "generate" else ("judge",)
    for run in runs:
        require(type(run.get("id")) is int and run["id"] > 0, "invalid prior diagnostic run ID")
        if run["id"] == run_id:
            continue
        require(run.get("path") == WORKFLOW and run.get("head_branch") == BRANCH and run.get("event") == "push", "diagnostic run history source mismatch")
        # The default API filter returns only the latest attempt. A refused
        # Actions rerun must not hide model work from an earlier attempt.
        jobs_document = gh(f"actions/runs/{run['id']}/jobs?filter=all&per_page=100")
        jobs = jobs_document.get("jobs", [])
        require(type(jobs_document.get("total_count")) is int and jobs_document["total_count"] == len(jobs) <= 100, "complete bounded diagnostic job history required")
        spent = False
        for job in jobs:
            if not any(job.get("name") == name or job.get("name", "").startswith(name + " (") for name in job_names):
                continue
            if job.get("conclusion") == "skipped" or job.get("status") == "queued":
                continue
            steps = [step for step in job.get("steps", []) if step.get("name") in step_names]
            require(steps or not job.get("started_at"), "started diagnostic job lacks its bounded execution step")
            job_spent = any(step.get("status") in ("in_progress", "completed") and step.get("conclusion") != "skipped" for step in steps)
            if job_spent:
                name = job["name"]
                if name == "judge":
                    artifact_name = "village001f-judgments"
                elif name == "base":
                    artifact_name = "village001f-infer-BASE"
                elif name == "sprocket":
                    artifact_name = "village001f-sprocket"
                elif name.startswith("train ("):
                    match = re.fullmatch(r"train \((V-G[0-9]{2}-[ABC]), (historical_token_id_mask|length_derived_mask)\)", name)
                    require(match and match[1] in MEMBERS, "unknown spending training lane")
                    artifact_name = f"village001f-train-{match[1]}-{match[2]}"
                else:
                    match = re.fullmatch(r"infer \((V-G[0-9]{2}-[ABC])\)", name)
                    require(match and match[1] in MEMBERS, "unknown spending inference lane")
                    artifact_name = f"village001f-infer-{match[1]}"
                required_artifacts.add(artifact_name)
                spent = True
        if spent:
            require(run.get("status") == "completed", "prior diagnostic spending run is still active")
            spending.append(run)
    if not spending:
        require(r["resume_run_id"] is None, "no prior spending run exists for this diagnostic phase")
        return
    source = max(spending, key=lambda run: run["id"])
    require(r["resume_run_id"] == source["id"], "phase must resume the latest prior spending run")
    head = source.get("head_sha", "")
    require(re.fullmatch(r"[0-9a-f]{40}", head), "invalid prior diagnostic source head")
    subprocess.run(["git", "fetch", "--no-tags", "origin", head], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"], check=True)
    prior_request = json.loads(subprocess.check_output(["git", "show", head + ":" + (REL / "request.json").as_posix()]))
    for key in ("phase", "contract_sha256", "stage_e_main_commit", "stage_e_summary_sha256", "input_release_tag", "final_release_tag"):
        require(prior_request.get(key) == r[key], "prior spending run has different frozen authority: " + key)
    artifact_document = gh(f"actions/runs/{source['id']}/artifacts?per_page=100")
    artifacts = artifact_document.get("artifacts", [])
    require(type(artifact_document.get("total_count")) is int and artifact_document["total_count"] == len(artifacts) <= 100, "complete recovery artifact inventory required")
    relevant = [a for a in artifacts if (a.get("name") == "village001f-judgments" if r["phase"] == "judge" else
        a.get("name", "").startswith(("village001f-train-", "village001f-infer-")) or a.get("name") == "village001f-sprocket")]
    require(relevant and len({a["name"] for a in relevant}) == len(relevant), "prior model work lacks unique preserved recovery artifacts")
    require(required_artifacts <= {a["name"] for a in relevant}, "latest recovery run lost a previously spending lane; custody repair required")
    for artifact in relevant:
        validate_recovery_artifact(artifact, source["id"], head)


def verify() -> None:
    r = request()
    require(os.environ.get("GITHUB_REPOSITORY") == REPO and os.environ.get("GITHUB_REF_NAME") == BRANCH and os.environ.get("GITHUB_EVENT_NAME") == "push", "wrong live execution surface")
    head = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/" + BRANCH], text=True).split()[0]
    require(head == os.environ["GITHUB_SHA"], "owned branch moved before execution")
    issue = gh("issues/14")
    require(issue["state"] == "closed" and issue.get("state_reason") == "completed", "Stage E must be accepted before this pilot executes")
    main_output = subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], text=True).split()
    require(len(main_output) == 2 and re.fullmatch(r"[0-9a-f]{40}", main_output[0]), "main authority unavailable")
    subprocess.run(["git", "fetch", "--no-tags", "origin", main_output[0]], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", r["stage_e_main_commit"], main_output[0]], check=True)
    subprocess.run(["git", "merge-base", "--is-ancestor", r["stage_e_main_commit"], "HEAD"], check=True)
    accepted_summary = subprocess.check_output(["git", "show", r["stage_e_main_commit"] + ":" + E_SUMMARY])
    require(hashlib.sha256(accepted_summary).hexdigest() == r["stage_e_summary_sha256"], "E summary does not match the accepted main commit")
    verify_execution_history(r)
    if r["phase"] == "generate":
        require(not (WORK / "inputs").exists(), "pilot inputs already frozen")
        absent_tag(INPUT_TAG)
    else:
        custody = verify_inputs()
        subprocess.run(["git", "merge-base", "--is-ancestor", custody["source_head"], "HEAD"], check=True)
        verify_release(WORK / "input-release-assets.json", INPUT_TAG)
    absent_tag(FINAL_TAG)
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in {"phase": r["phase"], "members": MEMBERS, "arms": ARMS,
                "resume_requested": "true" if r["resume_run_id"] is not None else "false"}.items():
            stream.write(key + "=" + (value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))) + "\n")


def assets(member: str, source: str) -> None:
    require(member in MEMBERS and source in ("C", "D"), "unknown diagnostic asset")
    path = ROOT / ("experiments/village-001/sft-evidence/adapter-release.json" if source == "C" else "experiments/village-001/preference-evidence/adapter-release.json")
    release = json.loads(path.read_text())
    row = next(x for x in release["assets"] if x["member_id"] == member)
    require(re.fullmatch(r"[A-Za-z0-9_.-]+", row["file"]), "unsafe asset name")
    print("\t".join([release["release_tag"], row["file"], row["sha256"]]))


def validate_recovery_run(run: dict, run_id: int) -> str:
    require(type(run_id) is int and run_id > 0 and run.get("id") == run_id, "wrong recovery run")
    require(run.get("status") == "completed" and run.get("path") == WORKFLOW and run.get("head_branch") == BRANCH, "wrong recovery workflow/branch/status")
    require(all(run.get(k, {}).get("id") == 1432134981 and run[k]["full_name"] == REPO for k in ("repository", "head_repository")), "wrong recovery repository")
    head = run.get("head_sha", "")
    require(re.fullmatch(r"[0-9a-f]{40}", head), "invalid source head")
    return head


def validate_recovery_artifact(artifact: dict, run_id: int, head: str) -> str:
    require(type(artifact.get("id")) is int and artifact["id"] > 0, "invalid recovery artifact ID")
    require(not artifact["expired"] and artifact["workflow_run"]["id"] == run_id and artifact["workflow_run"]["head_sha"] == head, "artifact source/expiry mismatch")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest", "")) and type(artifact["size_in_bytes"]) is int and 0 < artifact["size_in_bytes"] < 128 * 1024 * 1024, "artifact size/digest invalid")
    url = f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact['id']}/zip"
    require(artifact["archive_download_url"] == url, "artifact URL mismatch")
    return url


def recovery_sources(root: Path, phase: str, *, ignore_extracted_lane: bool = False) -> list[dict]:
    """Verify preserved native ZIPs, including inherited recovery hops, offline."""
    require(not root.is_symlink(), "recovery source root cannot be a symlink")
    if not root.exists():
        return []
    inventory = files(root)
    if ignore_extracted_lane:
        require(not (root / "lane").exists() or (root / "lane").is_dir(), "extracted recovery lane is not a directory")
        inventory = {key: value for key, value in inventory.items() if Path(key).parts[0] != "lane"}
    receipts = [root / name for name in inventory if name.endswith("-custody.json")]
    require(len(receipts) <= 256, "recovery history exceeds bounded inventory")
    expected_names = ({"village001f-judgments"} if phase == "judge" else
        {f"village001f-train-{member}-{arm}" for member in MEMBERS for arm in ARMS}
        | {f"village001f-infer-{member}" for member in [*MEMBERS, "BASE"]} | {"village001f-sprocket"})
    seen_files, found = set(), {}
    for path in receipts:
        archive = path.with_name(path.name.removesuffix("-custody.json") + ".zip")
        receipt = json.loads(path.read_text())
        run, artifact = receipt["run"], receipt["artifact"]
        head = validate_recovery_run(run, run.get("id"))
        validate_recovery_artifact(artifact, run["id"], head)
        require(artifact["name"] in expected_names, "recovery artifact is outside this diagnostic phase")
        verify_zip(archive, artifact)
        with tempfile.TemporaryDirectory(prefix="village001f-recovery-verify-") as temporary:
            extracted = safe_extract(archive, Path(temporary) / "native")
        require(extracted == receipt["files"], "recovery extracted-file custody mismatch")
        for item in (path, archive):
            seen_files.add(item.relative_to(root).as_posix())
        key = artifact["id"]
        if key in found:
            require(sha(path) == found[key]["receipt_sha256"] and sha(archive) == found[key]["archive_sha256"], "conflicting recovery source for one artifact")
            continue
        digest = artifact["digest"].removeprefix("sha256:")
        name = f"village001f-recovery-{key}-{digest[:16]}"
        found[key] = {"receipt_path": path, "archive_path": archive, "receipt_sha256": sha(path),
            "archive_sha256": digest, "archive_file": name + ".zip", "receipt_file": name + "-custody.json",
            "artifact_id": key, "artifact_name": artifact["name"], "run_id": run["id"], "source_head": head}
    require(seen_files == set(inventory), "recovery history contains unbound files or missing receipt pairs")
    return [found[key] for key in sorted(found)]


def copy_recovery_sources(sources: list[dict], evidence: Path, release_dir: Path, source_run_id: int | None) -> None:
    require(not evidence.exists() and not evidence.is_symlink(), "recovery evidence must be new")
    require(release_dir.is_dir() and not release_dir.is_symlink(), "recovery release directory must exist")
    require(release_dir.resolve() != ROOT.resolve() and ROOT.resolve() not in release_dir.resolve().parents, "recovery ZIPs must remain outside repository")
    for source in sources:
        require(not (release_dir / source["archive_file"]).exists(), "immutable recovery release asset already exists")
    evidence.mkdir(parents=True)
    rows = []
    for source in sources:
        shutil.copy2(source["archive_path"], release_dir / source["archive_file"])
        shutil.copy2(source["receipt_path"], evidence / source["receipt_file"])
        require(sha(release_dir / source["archive_file"]) == source["archive_sha256"] and sha(evidence / source["receipt_file"]) == source["receipt_sha256"], "recovery permanent-copy mismatch")
        rows.append({key: value for key, value in source.items() if not key.endswith("_path")})
    dump(evidence / "recovery-manifest.json", {"schema": "village-001f-permanent-recovery-custody-v1",
        "requested_source_run_id": source_run_id, "original_zip_files_in_release_only": True, "sources": rows})


def archive_recovery(artifacts: Path, evidence: Path, release_dir: Path) -> None:
    r = request()
    sources = recovery_sources(artifacts, r["phase"], ignore_extracted_lane=True)
    copy_recovery_sources(sources, evidence, release_dir, r["resume_run_id"])


def recover(name: str, out: Path) -> None:
    r = request()
    if r["resume_run_id"] is None:
        return
    require(re.fullmatch(r"village001f-[A-Za-z0-9_-]+", name), "unsafe artifact name")
    run_id, token = r["resume_run_id"], os.environ.get("GH_TOKEN", "")
    api = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}"
    run = json.loads(get_bytes(api, token))
    head = validate_recovery_run(run, run_id)
    subprocess.run(["git", "fetch", "--no-tags", "origin", head], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"], check=True)
    inventory = json.loads(get_bytes(api + "/artifacts?per_page=100", token))
    require(type(inventory.get("total_count")) is int and inventory["total_count"] == len(inventory.get("artifacts", [])) <= 100, "complete bounded recovery artifact inventory required")
    matches = [x for x in inventory["artifacts"] if x["name"] == name]
    require(len(matches) <= 1, "duplicate recovery artifact")
    if not matches:
        return
    a = matches[0]
    url = validate_recovery_artifact(a, run_id, head)
    archive = out.parent / (out.name + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    require(not archive.exists() and not archive.is_symlink(), "recovery ZIP already exists")
    archive.write_bytes(get_bytes(url, token, a["size_in_bytes"]))
    verify_zip(archive, a)
    rows = safe_extract(archive, out)
    dump(out.parent / (out.name + "-custody.json"), {"run": run, "artifact": a, "files": rows})
    # Retain prior native recovery ZIPs as well. They are sidecars, never files
    # inside the model/judge checkpoint consumed by an immutable resume.
    history_name = "village001f-recovery-" + name.removeprefix("village001f-")
    histories = [x for x in inventory["artifacts"] if x["name"] == history_name]
    require(len(histories) <= 1, "duplicate recovery history artifact")
    if histories:
        historical = histories[0]
        history_url = validate_recovery_artifact(historical, run_id, head)
        with tempfile.TemporaryDirectory(prefix="village001f-inherited-custody-") as temporary:
            source = Path(temporary)
            zipped = source / "history.zip"
            zipped.write_bytes(get_bytes(history_url, token, historical["size_in_bytes"]))
            verify_zip(zipped, historical)
            inherited = source / "history"
            safe_extract(zipped, inherited)
            recovery_sources(inherited, r["phase"])
            destination = out.parent / "history"
            require(not destination.exists() and not destination.is_symlink(), "recovery history output must be new")
            shutil.copytree(inherited, destination)


def validate_forward_scores(root: Path, cfg: dict, campaign: dict, audited: dict) -> None:
    paths = list(root.rglob("score.json"))
    require(len(paths) == 18, "all 18 fixed-weight diagnostic score lanes required")
    seen = set()
    for path in paths:
        doc = json.loads(path.read_text())
        identity = doc.get("identity", {})
        member, stage = identity.get("member_id"), identity.get("stage")
        key = (member, stage)
        require(member in MEMBERS and stage in ("BASE", "C", "D_ORIGINAL") and key not in seen, "unexpected or duplicate fixed-weight diagnostic lane")
        seen.add(key)
        adapter_sha = None if stage == "BASE" else worker.source_result(member, "C" if stage == "C" else "D")["adapter"]["sha256"]
        expected = worker.run_identity(WORK / "worker-config.json", cfg, kind="forward-only-paired-mask-scores", member=member, stage=stage, adapter_sha=adapter_sha)
        require(identity == expected, "fixed-weight diagnostic source identity drift")
        require(doc.get("pair_dataset_sha256") == worker.sha(worker.pair_path(member)), "fixed-weight diagnostic pair source drift")
        require(doc.get("model_forward_batches") == 16 and doc.get("optimizer_steps") == doc.get("new_generations") == 0, "fixed-weight diagnostic workload changed")
        pairs = worker.selected_pairs(member, campaign)
        rows = doc.get("rows", [])
        require(len(rows) == len(pairs) == 8 and [x.get("pair_id") for x in rows] == [x["pair_id"] for x in pairs], "fixed-weight diagnostic pair selection drift")
        for row, pair in zip(rows, pairs):
            require(row.get("scenario_id") == pair["scenario_id"] and row.get("training_diagnostic") is True, "fixed-weight diagnostic case marker drift")
            sequences = [audited[(member, pair["pair_id"], choice)] for choice in ("chosen", "rejected")]
            lengths = [len(x["input_token_ids"]) for x in sequences]
            width = max(lengths)
            ids = [x["input_token_ids"] + [2] * (width - length) for x, length in zip(sequences, lengths)]
            labels = [x["labels"] + [-100] * (width - length) for x, length in zip(sequences, lengths)]
            require(row.get("input_ids") == ids and row.get("labels") == labels, "fixed-weight diagnostic token/label custody mismatch")
            require(row.get("historical_attention_mask") == [[int(token != 2) for token in seq] for seq in ids] and row.get("length_attention_mask") == worker.length_mask_values(lengths, width), "fixed-weight diagnostic mask drift")
            require(row.get("real_positions_wrongly_masked") == [x["real_tokens_masked_by_historical_rule"] for x in sequences], "fixed-weight diagnostic real masked positions drift")
            scores = row.get("scores", {})
            require(set(scores) == set(ARMS), "fixed-weight diagnostic mask scores incomplete")
            for value in scores.values():
                require(set(value) == {"chosen_mean_logp", "rejected_mean_logp", "margin"} and all(type(v) in (int, float) and math.isfinite(v) for v in value.values()), "fixed-weight diagnostic nonfinite score")
                require(value["chosen_mean_logp"] <= 1e-6 and value["rejected_mean_logp"] <= 1e-6 and math.isclose(value["margin"], value["chosen_mean_logp"] - value["rejected_mean_logp"], rel_tol=1e-5, abs_tol=1e-6), "fixed-weight diagnostic log-probability arithmetic drift")
    require(seen == {(member, stage) for member in MEMBERS for stage in ("BASE", "C", "D_ORIGINAL")}, "fixed-weight diagnostic population incomplete")


def validate_input_sources(artifacts: Path) -> None:
    """Qualify all raw inputs before creating permanent evidence or weights."""
    import village_001f_judge as judge
    for name in ("worker", "sprocket", "token-audit"):
        evidence_files(artifacts / name)
    files(artifacts / "training")  # Includes weights, but never permits symlinks.
    jcfg, wcfg, suite, campaign, pref = judge.load_contract(WORK / "judge-config.json")
    training, _ = judge.load_training(artifacts / "training", jcfg, wcfg, campaign, pref)
    workers, _ = judge.load_workers(artifacts / "worker", training, jcfg, wcfg, suite)
    sprocket, _, _ = judge.load_sprocket(artifacts / "sprocket", jcfg)
    require(len(training) == 12 and len(workers) == 25 and sum(len(lane["rows"]) for lane in workers.values()) == 300 and len(sprocket) == 36, "complete native diagnostic population required")
    audit_paths = list((artifacts / "token-audit").rglob("token-mask-audit.json"))
    require(len(audit_paths) == 1, "one native tokenizer audit required")
    audit = json.loads(audit_paths[0].read_text())
    expected_identity = worker.run_identity(WORK / "worker-config.json", wcfg, kind="tokenizer-mask-audit")
    require(audit.get("identity") == expected_identity, "tokenizer audit source identity drift")
    require(audit.get("optimizer_steps") == 0 and audit.get("new_model_generations") == 0 and audit.get("tokenizer_eos_id") == audit.get("tokenizer_pad_id") == 2, "tokenizer audit contract drift")
    require(audit.get("sequences") == 720 and len(audit.get("records", [])) == 720, "native tokenizer sequence audit incomplete")
    expected_rows = {(member, row["pair_id"], choice): row for member in MEMBERS for row in worker.jsonl(worker.pair_path(member)) for choice in ("chosen", "rejected")}
    found = {}
    for row in audit["records"]:
        key = (row.get("member_id"), row.get("pair_id"), row.get("choice"))
        require(key in expected_rows and key not in found, "duplicate or unexpected tokenizer audit row")
        found[key] = row
        ids = row.get("input_token_ids")
        require(isinstance(ids, list) and ids and all(type(x) is int and x >= 0 for x in ids), "invalid native tokenizer IDs")
        require(row.get("scenario_id") == expected_rows[key]["scenario_id"], "tokenizer case source mismatch")
        require(row.get("sequence_tokens") == len(ids) <= 512, "native tokenizer sequence length drift")
        prompt_ids = row.get("prompt_token_ids")
        require(isinstance(prompt_ids, list) and all(type(x) is int and x >= 0 for x in prompt_ids), "invalid native tokenizer prompt IDs")
        common = 0
        for left, right in zip(ids, prompt_ids):
            if left != right:
                break
            common += 1
        require(common < len(ids) and row.get("prompt_common_prefix_tokens") == common and row.get("labels") == [-100] * common + ids[common:], "native assistant label boundary mismatch")
        require(row.get("historical_attention_mask") == [int(x != 2) for x in ids] and row.get("length_attention_mask") == [1] * len(ids), "native historical/corrected mask mismatch")
        require(row.get("real_tokens_masked_by_historical_rule") == [i for i, x in enumerate(ids) if x == 2], "native incorrectly masked position mismatch")
    require(set(found) == set(expected_rows), "native tokenizer pair population incomplete")
    validate_forward_scores(artifacts / "worker", wcfg, campaign, found)


def archive_inputs(artifacts: Path, release_dir: Path) -> None:
    r = request()
    cfg, _, _, _ = load_contract(WORK / "worker-config.json")
    destination = WORK / "inputs"
    require(not destination.exists(), "diagnostic inputs already frozen")
    require(not release_dir.exists() and not release_dir.is_symlink(), "release directory must be new")
    require(release_dir.resolve() != ROOT.resolve() and ROOT.resolve() not in release_dir.resolve().parents, "adapter release directory must remain outside repository")
    validate_input_sources(artifacts)
    recovery = recovery_sources(artifacts / "recovery", r["phase"])
    # Validate every exact training directory/adapter before copying any bytes.
    for member in MEMBERS:
        require((artifacts / "worker" / f"village001f-infer-{member}").is_dir(), "missing worker inference")
        for arm in ARMS:
            lane = artifacts / "training" / f"village001f-train-{member}-{arm}"
            files(lane)
            checked_adapter(cfg, member, "D_REPRODUCED" if arm == ARMS[0] else "D_CORRECTED", lane / "adapter", lane)
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="village001f-input-staging-", dir=release_dir.parent) as temporary:
        staged = Path(temporary) / "inputs"
        for member in MEMBERS:
            for arm in ARMS:
                lane = artifacts / "training" / f"village001f-train-{member}-{arm}"
                target = staged / "training" / member / arm
                target.mkdir(parents=True, exist_ok=True)
                for name in ("training.json", "training-identity.json", "result.json"):
                    shutil.copy2(lane / name, target / name)
                archive = release_dir / f"village001f-{member}-{arm}.tar.gz"
                with archive.open("xb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped, tarfile.open(fileobj=zipped, mode="w") as tar:
                    for name in sorted(files(lane / "adapter")):
                        path = lane / "adapter" / name
                        info = tar.gettarinfo(str(path), arcname=name)
                        info.uid = info.gid = info.mtime = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
        for name in ("worker", "sprocket", "token-audit"):
            shutil.copytree(artifacts / name, staged / name)
        copy_recovery_sources(recovery, staged / "recovery", release_dir, r["resume_run_id"])
        staged_inventory = evidence_files(staged)
        require(json.loads((staged / "sprocket/summary.json").read_text())["local_generations"] == 36, "Sprocket count mismatch")
        shutil.copytree(staged, destination)
        require(evidence_files(destination) == staged_inventory, "permanent native input copy mismatch")
    dump(WORK / "input-custody.json", {"schema": "village-001f-input-custody-v1", "run_id": int(os.environ["GITHUB_RUN_ID"]), "source_head": os.environ["GITHUB_SHA"],
        "stage_e_main_commit": r["stage_e_main_commit"], "stage_e_summary_sha256": r["stage_e_summary_sha256"],
        "contract_sha256": r["contract_sha256"], "input_release_tag": r["input_release_tag"], "final_release_tag": r["final_release_tag"],
        "request_sha256": sha(WORK / "request.json"), "worker_generations": 300, "sprocket_generations": 36, "optimizer_steps": 1440, "files": staged_inventory})


def verify_inputs() -> dict:
    r = request()
    record = json.loads((WORK / "input-custody.json").read_text())
    require(record.get("schema") == "village-001f-input-custody-v1", "wrong input custody schema")
    for key in ("stage_e_main_commit", "stage_e_summary_sha256", "contract_sha256", "input_release_tag", "final_release_tag"):
        require(record.get(key) == r[key], "frozen diagnostic input authority changed: " + key)
    require(re.fullmatch(r"[0-9a-f]{40}", record.get("source_head", "")), "input source head missing")
    require(record["files"] == evidence_files(WORK / "inputs"), "frozen diagnostic input custody changed")
    require(record["worker_generations"] == 300 and record["sprocket_generations"] == 36 and record["optimizer_steps"] == 1440, "diagnostic workload mismatch")
    return record


def release_inventory(directory: Path, target: Path, tag: str) -> None:
    require(tag in (INPUT_TAG, FINAL_TAG), "diagnostic release target is not isolated")
    inventory = files(directory)
    require(inventory and all(Path(name).name == name for name in inventory), "release assets must be nonempty regular top-level files")
    dump(target, {"release_tag": tag, "source_head": os.environ["GITHUB_SHA"], "run_id": int(os.environ["GITHUB_RUN_ID"]), "assets": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(directory.iterdir()) if p.is_file()]})


def verify_release(manifest: Path, tag: str) -> None:
    require(tag in (INPUT_TAG, FINAL_TAG), "diagnostic release target is not isolated")
    expected = json.loads(manifest.read_text())
    release = gh("releases/tags/" + tag)
    found = {x["name"]: x for x in release["assets"]}
    require(release["prerelease"] and not release.get("draft", False) and release.get("tag_name") == tag and expected["release_tag"] == tag and set(found) == {x["file"] for x in expected["assets"]}, "release set mismatch")
    require(len(found) == len(release["assets"]) == len(expected["assets"]) and bool(found), "duplicate or empty release asset inventory")
    for row in expected["assets"]:
        a = found[row["file"]]
        require(a["state"] == "uploaded" and a["size"] == row["bytes"] and a["digest"] == "sha256:" + row["sha256"], "release asset mismatch")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("verify-inputs")
    p = sub.add_parser("assets"); p.add_argument("--member", required=True); p.add_argument("--source", choices=("C", "D"), required=True)
    p = sub.add_parser("recover"); p.add_argument("--name", required=True); p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("archive-inputs"); p.add_argument("--artifacts", type=Path, required=True); p.add_argument("--release-dir", type=Path, required=True)
    p = sub.add_parser("archive-recovery"); p.add_argument("--artifacts", type=Path, required=True); p.add_argument("--evidence", type=Path, required=True); p.add_argument("--release-dir", type=Path, required=True)
    p = sub.add_parser("release-inventory"); p.add_argument("--directory", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--tag", required=True)
    p = sub.add_parser("verify-release"); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.command == "verify": verify()
    elif args.command == "verify-inputs": verify_inputs()
    elif args.command == "assets": assets(args.member, args.source)
    elif args.command == "recover": recover(args.name, args.out)
    elif args.command == "archive-inputs": archive_inputs(args.artifacts, args.release_dir)
    elif args.command == "archive-recovery": archive_recovery(args.artifacts, args.evidence, args.release_dir)
    elif args.command == "release-inventory": release_inventory(args.directory, args.manifest, args.tag)
    else: verify_release(args.manifest, args.tag)


if __name__ == "__main__":
    main()
