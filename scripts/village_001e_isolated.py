#!/usr/bin/env python3
"""Source-bound operator for the isolated VILLAGE-001E qualification campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from village_001d_recover_completed import get_bytes, safe_extract, verify_zip
from village_001e_ecology_plan import build_sessions

ROOT = Path(__file__).resolve().parents[1]
REL = Path("experiments/village-001/qualification-20261119")
CAMPAIGN = ROOT / REL
BRANCH = "work/village-001e-qualification-20261119"
REPO = "techrote/neurounner"
WORKFLOW = ".github/workflows/village-001e-isolated.yml"
MEMBERS = [f"V-G{i:02d}-{s}" for i in range(1, 13) for s in "ABC"]
FIXED = {
    "schema": "village-001e-isolated-execution-v1",
    "publish_branch": BRANCH,
    "stage_d_evidence_head": "2e61085c3edd623ffe15d54b1898d9bcedbbfa42",
    "stage_d_summary_sha256": "67bfdea373cd025c016c621c7eccc696da2b2b02bbe62242cada753ebba60302",
    "stage_d_release_sha256": "6348d07c8809ebf91c69c3beeb67b0a0136f618d360eb109195308fb890d2e23",
    "final_suite_sha256": "b404422c5a32ccdc53abf78f849a92a82938c0e88c2585d9d4ac90f3519631da",
    "release_tag": "village-001e-final-20261119",
    "new_training": False,
}


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tree_files(root: Path) -> dict:
    require(root.is_dir() and not root.is_symlink(), "missing/nonregular evidence root")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "nonregular evidence path")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return result


def request() -> tuple[dict, dict]:
    req = json.loads((CAMPAIGN / "request.json").read_text())
    require(set(req) == set(FIXED) | {"phase", "execution_revision", "resume_run_id"}, "unexpected request fields")
    require(all(req.get(k) == v for k, v in FIXED.items()), "isolated request identity drift")
    require(req["phase"] in ("generate", "judge"), "unknown phase")
    require(type(req["execution_revision"]) is int and req["execution_revision"] > 0, "invalid revision")
    require(req["resume_run_id"] is None or (type(req["resume_run_id"]) is int and req["resume_run_id"] > 0), "invalid resume run")
    require(sha(ROOT / "experiments/village-001/preference-evidence/summary.json") == FIXED["stage_d_summary_sha256"], "D summary changed")
    require(sha(ROOT / "experiments/village-001/preference-evidence/adapter-release.json") == FIXED["stage_d_release_sha256"], "D release changed")
    require(sha(CAMPAIGN / "final-eval.json") == FIXED["final_suite_sha256"], "final suite changed")
    cfg = json.loads((CAMPAIGN / "config.json").read_text())
    expected = json.loads((ROOT / "experiments/village-001/tournament-config.json").read_text())
    expected.update({"final_eval_config": str(REL / "final-eval.json"), "final_release_tag": "village-001d-recovery-40613393744-20261118", "qualification_release_tag": FIXED["release_tag"]})
    require(cfg == expected, "qualification recipe or gate drift")
    return req, cfg


def validate_live_request() -> None:
    req, cfg = request()
    require(os.environ.get("GITHUB_REPOSITORY") == REPO, "wrong repository")
    require(os.environ.get("GITHUB_REF_NAME") == BRANCH, "wrong publication branch")
    require(os.environ.get("GITHUB_EVENT_NAME") == "push", "live execution requires owned-branch push")
    remote = subprocess.check_output(["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"], text=True).split()[0]
    require(remote == os.environ["GITHUB_SHA"], "owned branch moved before execution")
    tag = subprocess.check_output(["git", "ls-remote", "--tags", "origin", "refs/tags/" + FIXED["release_tag"]], text=True)
    require(not tag.strip(), "immutable qualification tag already exists")
    release = subprocess.run(["gh", "api", f"repos/{REPO}/releases/tags/{FIXED['release_tag']}"], capture_output=True, text=True)
    require(release.returncode != 0 and "HTTP 404" in release.stderr, "release exists or absence cannot be established")
    if req["phase"] == "judge":
        verify_generation()
        replay_root = Path(os.environ["RUNNER_TEMP"]) / "generation-source-replay"
        for member in MEMBERS + ["BASE"]:
            args = ["python3", "scripts/village_001e_generate_final_responses.py", "--config", str(REL / "config.json"), "--resume", str(REL / "generation/responses" / member), "--out", str(replay_root / "responses" / member), "--offline"]
            args += ["--base-only"] if member == "BASE" else ["--member", member]
            subprocess.run(args, check=True)
        for session in build_sessions(cfg):
            ident = session["session_id"]
            subprocess.run(["python3", "scripts/village_001e_ecology_run.py", "--config", str(REL / "config.json"), "--session", ident, "--resume", str(REL / "generation/ecology" / ident), "--out", str(replay_root / "ecology" / ident), "--offline"], check=True)
    else:
        require(not (CAMPAIGN / "generation").exists(), "generation already frozen; use judge phase")
    sessions = build_sessions(cfg)
    outputs = {
        "phase": req["phase"],
        "members": MEMBERS + ["BASE"],
        "ecology": [s["session_id"] for s in sessions],
        "providers": {"include": ([{"kind": "ecology", "id": s["session_id"]} for s in sessions] + [{"kind": "tournament", "id": f"F{i:02d}"} for i in range(1, 25)])},
    }
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in outputs.items():
            stream.write(key + "=" + (value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))) + "\n")
    print(json.dumps({"phase": req["phase"], "revision": req["execution_revision"], "suite_sha256": FIXED["final_suite_sha256"], "new_training": False}))


def archive_generation(artifacts: Path) -> None:
    req, cfg = request()
    require(req["phase"] == "generate", "generation archive requires generate phase")
    out = CAMPAIGN / "generation"
    require(not out.exists(), "generation archive already frozen")
    expected = [("responses", m, "result.json", f"village001e-responses-{m}") for m in MEMBERS + ["BASE"]]
    expected += [("ecology", s["session_id"], "session.json", f"village001e-ecology-{s['session_id']}") for s in build_sessions(cfg)]
    for kind, ident, terminal, artifact in expected:
        source = artifacts / artifact
        require((source / terminal).is_file(), f"missing terminal generation: {ident}")
        tree_files(source)
        doc = json.loads((source / terminal).read_text())
        require(doc.get("member_id" if kind == "responses" else "session_id") == ident, "generation identity mismatch")
    for kind, ident, _, artifact in expected:
        shutil.copytree(artifacts / artifact, out / kind / ident)
    manifest = {"schema": "village-001e-generation-custody-v1", "run_id": int(os.environ["GITHUB_RUN_ID"]), "source_head": os.environ["GITHUB_SHA"], "request_sha256": sha(CAMPAIGN / "request.json"), "config_sha256": sha(CAMPAIGN / "config.json"), "final_suite_sha256": FIXED["final_suite_sha256"], "response_members": 37, "ecology_sessions": 24, "local_generations": 1068, "new_optimizer_steps": 0, "files": tree_files(out)}
    dump(CAMPAIGN / "generation-custody.json", manifest)
    verify_generation()


def verify_generation() -> None:
    manifest = json.loads((CAMPAIGN / "generation-custody.json").read_text())
    require(manifest.get("config_sha256") == sha(CAMPAIGN / "config.json"), "generation config drift")
    require(manifest.get("final_suite_sha256") == FIXED["final_suite_sha256"], "generation suite drift")
    require(manifest.get("files") == tree_files(CAMPAIGN / "generation"), "generation custody drift")
    require(len(list((CAMPAIGN / "generation/responses").glob("*/result.json"))) == 37, "response population incomplete")
    require(len(list((CAMPAIGN / "generation/ecology").glob("*/session.json"))) == 24, "ecology population incomplete")


def recover_lane(kind: str, ident: str, out: Path) -> None:
    req, cfg = request()
    run_id = req["resume_run_id"]
    if run_id is None:
        print("No prior lane requested")
        return
    allowed = {"responses": set(MEMBERS + ["BASE"]), "ecology": {s["session_id"] for s in build_sessions(cfg)}, "judge-ecology": {s["session_id"] for s in build_sessions(cfg)}, "judge-tournament": {f"F{i:02d}" for i in range(1, 25)}}
    require(ident in allowed.get(kind, set()), "unexpected lane")
    token = os.environ.get("GH_TOKEN", "")
    api = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}"
    run = json.loads(get_bytes(api, token))
    require(run.get("path") == WORKFLOW and run.get("head_branch") == BRANCH and run.get("status") == "completed", "resume workflow/branch/status mismatch")
    require(all(run.get(k, {}).get("full_name") == REPO and run[k].get("id") == 1432134981 for k in ("repository", "head_repository")), "resume repository mismatch")
    source_head = run["head_sha"]
    require(re.fullmatch(r"[0-9a-f]{40}", source_head) is not None, "invalid source SHA")
    subprocess.run(["git", "fetch", "--no-tags", "origin", source_head], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["git", "merge-base", "--is-ancestor", source_head, "HEAD"], check=True)
    name = f"village001e-{kind}-{ident}"
    matches = []
    for page in range(1, 5):
        data = json.loads(get_bytes(api + f"/artifacts?per_page=100&page={page}", token))
        matches.extend(a for a in data["artifacts"] if a["name"] == name)
        if page * 100 >= data["total_count"]:
            break
    require(len(matches) <= 1, "duplicate resume artifact")
    if not matches:
        print("Prior run has no artifact for " + name)
        return
    artifact = matches[0]
    require(not artifact["expired"], "resume artifact expired")
    require(artifact["workflow_run"]["id"] == run_id and artifact["workflow_run"]["head_sha"] == source_head, "artifact provenance mismatch")
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", artifact.get("digest", "")) is not None, "artifact digest unavailable")
    require(0 < artifact["size_in_bytes"] < 128 * 1024 * 1024, "artifact outside size bound")
    expected_url = f"https://api.github.com/repos/{REPO}/actions/artifacts/{artifact['id']}/zip"
    require(artifact["archive_download_url"] == expected_url, "artifact download URL mismatch")
    zip_path = out.parent / (out.name + ".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    require(not zip_path.exists(), "resume ZIP already exists")
    zip_path.write_bytes(get_bytes(expected_url, token, artifact["size_in_bytes"]))
    verify_zip(zip_path, artifact)
    files = safe_extract(zip_path, out)
    dump(out.parent / (out.name + "-custody.json"), {"run": run, "artifact": artifact, "files": files})
    print(json.dumps({"recovered": name, "files": len(files), "sha256": artifact["digest"]}))


def asset_list(kind: str, ident: str) -> None:
    _, cfg = request()
    release = json.loads((ROOT / cfg["preference_evidence_root"] / "adapter-release.json").read_text())
    by_member = {x["member_id"]: x for x in release["assets"]}
    if kind == "responses":
        require(ident in MEMBERS + ["BASE"], "unknown response member")
        members = [] if ident == "BASE" else [ident]
    else:
        sessions = {s["session_id"]: s for s in build_sessions(cfg)}
        require(ident in sessions, "unknown ecology session")
        members = sessions[ident]["members"]
    for member in members:
        row = by_member[member]
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", row["file"]) is not None, "unsafe asset name")
        print("\t".join((member, release["release_tag"], row["file"], row["sha256"])))


def attach_custody(receipt: Path, out: Path) -> None:
    if not receipt.exists() or not out.exists():
        return
    require(receipt.is_file() and not receipt.is_symlink() and out.is_dir() and not out.is_symlink(), "invalid custody attachment paths")
    target = out / f"resume-artifact-custody-{sha(receipt)}.json"
    if target.exists():
        require(target.read_bytes() == receipt.read_bytes(), "custody receipt collision")
    else:
        with target.open("xb") as stream:
            stream.write(receipt.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("verify-generation")
    p = sub.add_parser("archive-generation")
    p.add_argument("--artifacts", type=Path, required=True)
    p = sub.add_parser("recover-lane")
    p.add_argument("--kind", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("attach-custody")
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("asset-list")
    p.add_argument("--kind", choices=("responses", "ecology"), required=True)
    p.add_argument("--id", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        validate_live_request()
    elif args.command == "verify-generation":
        verify_generation()
    elif args.command == "archive-generation":
        archive_generation(args.artifacts)
    elif args.command == "asset-list":
        asset_list(args.kind, args.id)
    elif args.command == "attach-custody":
        attach_custody(args.receipt, args.out)
    else:
        recover_lane(args.kind, args.id, args.out)


if __name__ == "__main__":
    main()
