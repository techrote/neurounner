#!/usr/bin/env python3
"""Read-only Stage E acceptance audit. All model/provider access is disabled.

The repository is never modified. Replays and release decoding use temporary
directories. --source-only validates the already frozen source/generation inputs
without claiming completed Stage E acceptance. A negative usable-village verdict
does not fail evidence acceptance. Main/docs requirements are separate switches.
"""
from __future__ import annotations

import argparse
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
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
Q = Path("experiments/village-001/qualification-20261119")
E = Path("experiments/village-001/final-evidence")
D = Path("datasets/village-001")
TAG = "village-001e-final-20261119"
MODEL_TAG = "village-001d-recovery-40613393744-20261118"
MANIFEST_SHA = "805a0fc80513c291c2906e0a12115278d2b7f87c30544725c08dcfc4d75d96d6"
JSON_ASSETS = {name + ".json" for name in ("summary", "confusion-matrices", "ecology-summary", "model-registry", "dataset-provenance", "per-member-verdicts", "final-eval")}
TAR_ASSET = "village-001e-complete-evidence.tar"
BASELINE_RUN = 44443194861
BASELINE_ZIP_SHA = "4c5f263a836e8a38fc7e8cdeb0143103072745eb8a628d569ebc2eda2727d70d"
ALLOCATION_SHA = "f513e0cb59e9a4322b770e9bac950dd8aa49480ec795c3a32bdbd1747a73e440"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def same(actual, expected, message):
    require(actual == expected, message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def no_network(*args, **kwargs):
    raise RuntimeError("network/provider access is forbidden in the independent acceptance audit")


@contextlib.contextmanager
def quiet_main(module, argv):
    old = sys.argv
    try:
        sys.argv = [module.__file__, *map(str, argv)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        sys.argv = old


def replay_assembler(repo, scratch, asm, isolated, cfg, provenance):
    """Reuse every strict identity/chain/scoring validator without touching Git."""
    mirror = scratch / "assembler-replay"
    for rel in ("scripts", "experiments/village-001", "experiments/persona-001", str(D)):
        shutil.copytree(repo / rel, mirror / rel, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    old_root = asm.ROOT
    old_env = {key: os.environ.get(key) for key in ("GITHUB_RUN_ID", "GITHUB_SHA")}
    try:
        asm.ROOT = mirror
        os.environ["GITHUB_RUN_ID"] = str(provenance["workflow_run_id"])
        os.environ["GITHUB_SHA"] = provenance["source_commit"]
        args = ["--response-artifacts", repo / Q / "generation/responses",
                "--tournament-artifacts", repo / E / "tournament-scenarios",
                "--ecology-artifacts", repo / Q / "generation/ecology",
                "--ecology-judgment-artifacts", repo / E / "ecology/judgments",
                "--config", repo / Q / "config.json", "--out", mirror / E,
                "--release-tag", TAG]
        with quiet_main(asm, args):
            asm.main()
    finally:
        asm.ROOT = old_root
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    same(isolated.tree_files(mirror / E), isolated.tree_files(repo / E), "strict reassembly differs from committed final evidence")
    same(isolated.tree_files(mirror / D), isolated.tree_files(repo / D), "strict reassembly differs from committed final dataset")


def independently_recompute_gates(repo, cfg, summary):
    members = [f"V-G{i:02d}-{s}" for i in range(1, 13) for s in "ABC"]
    records = {member: [] for member in members}
    controls = []
    for index in range(1, 25):
        doc = read(repo / E / "tournament-scenarios" / f"F{index:02d}" / "scenario.json")
        same(sorted(row["member_id"] for row in doc["records"]), members, "tournament population incomplete")
        controls.append(doc["untuned_base_control"])
        for row in doc["records"]:
            records[row["member_id"]].append(row)
    declared = cfg["tournament"]
    admitted = set(read(repo / "experiments/village-001/preference-evidence/summary.json")["qualified_for_population_tournament"])
    published = {row["member_id"]: row for row in summary["members_table"]}
    same(set(published), set(members), "member verdicts omit a phenotype")
    total_hits = {f"{mode}_{task}": 0 for mode in ("raw", "behavior") for task in ("lineage", "sibling", "member")}
    member_gates, means, passed = {}, [], []
    for member, rows in records.items():
        same(len(rows), 24, "member does not have exactly 24 judgments")
        lineage, sibling = member.split("-")[1:]
        hits = {}
        for mode in ("raw", "behavior"):
            hits[mode + "_lineage"] = sum(r[mode + "_prediction"]["predicted_lineage"] == lineage for r in rows)
            hits[mode + "_sibling"] = sum(r[mode + "_sibling_prediction"]["predicted_sibling"] == sibling for r in rows)
            hits[mode + "_member"] = sum(r[mode + "_prediction"]["predicted_lineage"] == lineage and r[mode + "_sibling_prediction"]["predicted_sibling"] == sibling for r in rows)
            for task in ("lineage", "sibling", "member"):
                total_hits[mode + "_" + task] += hits[mode + "_" + task]
                field = mode + ("_sibling_accuracy_given_correct_lineage" if task == "sibling" else "_" + task + "_accuracy")
                same(published[member][field], hits[mode + "_" + task] / 24, member + " per-member accuracy drift")
        relevance = statistics.mean(float(r["raw_prediction"]["task_relevance"]) for r in rows)
        genericness = statistics.mean(float(r["raw_prediction"]["genericness"]) for r in rows)
        gates = {"maturation": member in admitted,
                 "behavior_lineage_hits": hits["behavior_lineage"] >= declared["individual_min_behavior_lineage_hits"],
                 "behavior_sibling_hits_given_correct_lineage": hits["behavior_sibling"] >= declared["individual_min_behavior_sibling_hits"],
                 "raw_task_relevance": relevance >= declared["individual_min_task_relevance"],
                 "raw_genericness": genericness <= declared["individual_max_genericness"]}
        same(published[member]["individual_gates"], gates, member + " individual gate drift")
        same(published[member]["individual_quality_pass"], all(gates.values()), member + " individual verdict drift")
        same(published[member]["failed_individual_gates"], [key for key, value in gates.items() if not value], member + " failures hidden")
        same(published[member]["raw_task_relevance"], relevance, member + " relevance mean drift")
        same(published[member]["raw_genericness"], genericness, member + " genericness mean drift")
        member_gates[member] = gates
        means.append(genericness)
        if all(gates.values()):
            passed.append(member)
    for mode in ("raw", "behavior"):
        target = summary["raw" if mode == "raw" else "behavior_only"]
        for task in ("lineage", "sibling", "member"):
            field = "sibling_accuracy_given_correct_lineage" if task == "sibling" else task + "_accuracy"
            same(target[field], total_hits[mode + "_" + task] / 864, "population accuracy drift")
    base_genericness = statistics.mean(float(row["raw"]["genericness"]) for row in controls)
    values = {"behavior_lineage_accuracy": total_hits["behavior_lineage"] / 864,
              "behavior_sibling_accuracy_given_correct_lineage": total_hits["behavior_sibling"] / 864,
              "mean_genericness_below_base": statistics.mean(means), "individual_quality_members": len(passed)}
    thresholds = {"behavior_lineage_accuracy": declared["lineage_gate"],
                  "behavior_sibling_accuracy_given_correct_lineage": declared["sibling_gate_given_correct_lineage"],
                  "mean_genericness_below_base": base_genericness, "individual_quality_members": declared["individual_quality_required"]}
    same(set(summary["population_gates"]), set(values), "population gate set changed")
    checks = {}
    for key, value in values.items():
        check = value < thresholds[key] if key == "mean_genericness_below_base" else value >= thresholds[key]
        same(summary["population_gates"][key]["value"], value, key + " value drift")
        same(summary["population_gates"][key]["threshold"], thresholds[key], key + " threshold drift")
        same(summary["population_gates"][key]["pass"], check, key + " decision drift")
        checks[key] = check
    same(summary["individual_quality_pass"], passed, "qualified member list drift")
    same(summary["individual_quality_fail"], [m for m in members if m not in passed], "failed member list drift")
    same(summary["individual_quality_pass_count"], len(passed), "member pass count drift")
    same(summary["usable_village"], all(checks.values()), "usable-village result drift")
    same(summary["final_quality"]["fresh_base_raw_genericness"], base_genericness, "BASE comparator drift")
    return {"raw_record_counts": {"villager": 864, "BASE": 24}, "accuracy_hit_counts": total_hits,
            "population_gates": summary["population_gates"], "individual_quality_pass": passed,
            "individual_quality_fail": summary["individual_quality_fail"], "individual_gates": member_gates,
            "usable_village": summary["usable_village"], "completion_requires_usable_village": False,
            "effective_lineage_minimum_hits": math.ceil(declared["lineage_gate"] * 864),
            "effective_sibling_minimum_hits": math.ceil(declared["sibling_gate_given_correct_lineage"] * 864)}


def audit_prior_chain(repo, scratch, recovery, latest_zip, latest_run, allocation_context=None):
    """Verify every inherited ZIP/receipt; count raw requests, never receipt status."""
    pending = [(latest_run, latest_zip)]
    verified = {}
    snapshot = None
    while pending:
        run_id, archive = pending.pop()
        receipt = read(repo / Q / "provider-recovery" / f"resume-recovery-{run_id}.json")
        same(receipt["run"]["id"], run_id, "prior custody run identity changed")
        same(receipt["artifact"]["workflow_run"]["id"], run_id, "prior artifact/run identity changed")
        same(receipt["artifact"]["workflow_run"]["head_sha"], receipt["run"]["head_sha"], "prior artifact/source identity changed")
        recovery.verify_zip(archive, receipt["artifact"])
        if run_id in verified:
            same(verified[run_id]["zip_sha256"], sha(archive), "duplicate prior run has different ZIP bytes")
            continue
        decoded = scratch / f"prior-tree-{run_id}"
        recovery.safe_extract(archive, decoded)
        recorded = {row["path"]: {key: row[key] for key in ("bytes", "sha256")} for row in receipt["files"]}
        same(recovery.tree(decoded), recorded, "prior recovery ZIP custody differs")
        native_gaps = {}
        if allocation_context is not None:
            path, allocation, original, config, extension = allocation_context
            for row in allocation["lanes"]:
                source = decoded / "lanes" / row["artifact_name"]
                if not source.exists():
                    source = decoded / "inherited" / row["artifact_name"]
                extension.validate_prior_source(source, allocation, row, original=original)
                actual, logical = recovery.labels(source)
                receipts = [read(p) for p in source.glob("allocation-extension-execution-*.json")]
                gap = {"logical": len(logical) - row["baseline_logical_count"] - sum(r["new_logical_requests"] for r in receipts),
                       "transports": len(actual) - len(row["baseline_actual_labels"]) - sum(r["new_transport_launches"] for r in receipts)}
                require(min(gap.values()) >= 0, "prior invocation receipts overstate archived raw requests")
                native_gaps[row["lane"]] = gap
            if run_id == BASELINE_RUN:
                same(sha(archive), BASELINE_ZIP_SHA, "first recovery ZIP differs from independent readback")
                extension.validate_baseline_bundle(decoded, allocation)
                metadata = allocation["metadata"]
                inputs = {}
                for key, record in metadata.items():
                    if record["path"] is not None:
                        inputs[key] = repo / record["path"]
                    else:
                        raise RuntimeError("final allocation metadata must have a durable repository source path")
                rebuilt = extension.make_manifest(argparse.Namespace(manifest=repo / allocation["original_manifest"]["path"],
                    prior_root=decoded, prior_zip=archive, run_record=inputs["run"], jobs_record=inputs["jobs"],
                    artifact_inventory=inputs["artifacts"]))
                same(rebuilt, allocation, "allocation does not reconstruct from the original preserved snapshot")
                snapshot = {"run_id": run_id, "zip_sha256": sha(archive), "files": len(recorded),
                            "logical_requests": 673, "transport_launches": 674, "completed_lanes": 41,
                            "completed_lane_logical_requests": 630,
                            "completed_trees": {r["lane"]: {"tree_sha256": r["baseline_source_tree_sha256"],
                                "terminal_sha256": r["baseline_terminal_sha256"]} for r in allocation["lanes"] if r["baseline_status"] == "completed"},
                            "missing_native_v1_receipt": {"lane": "F19", "logical_requests": 3, "transport_launches": 3,
                                "basis": "interrupted execution; immutable request files plus exact GitHub ZIP/run custody"}}
        verified[run_id] = {"zip_sha256": sha(archive), "files": len(recorded), "extension_receipt_gaps": native_gaps}
        for older in (decoded / "custody").glob("recovery-run-*.zip"):
            match = re.fullmatch(r"recovery-run-(\d+)\.zip", older.name)
            require(match is not None, "unsafe inherited recovery ZIP name")
            pending.append((int(match.group(1)), older))
    if allocation_context is not None:
        require(snapshot is not None, "final release lost the exact first recovery ZIP")
    return {"verified_prior_runs": verified, "first_recovery_snapshot": snapshot}


def audit_release(args, repo, scratch, recovery, manifest, config, receipt, allocation_context=None):
    require(args.release_dir and args.published_release and args.release_target,
            "release audit requires --release-dir, --published-release and --release-target together")
    target = args.release_target
    same(git(repo, "rev-parse", f"refs/tags/{TAG}^{{commit}}"), target, "fetched release tag does not point to the expected bot commit")
    subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", target, "HEAD"], check=True)
    published = read(args.published_release)
    same(published["tag_name"], TAG, "published release tag changed")
    require(published["prerelease"] is True and published["draft"] is False, "qualification is not a published prerelease")
    assets = {row["name"]: row for row in published["assets"]}
    expected = {row["file"]: row for row in receipt["assets"]}
    same(len(assets), len(published["assets"]), "duplicate remote release asset")
    same(set(expected), JSON_ASSETS | {TAR_ASSET}, "release receipt does not contain exactly eight declared assets")
    same(set(assets), set(expected), "remote release asset set changed")
    for name, row in expected.items():
        path = args.release_dir / name
        same(path.stat().st_size, row["bytes"], "downloaded asset byte count differs: " + name)
        same(sha(path), row["sha256"], "downloaded asset digest differs: " + name)
        remote = assets[name]
        same((remote["state"], remote["size"], remote["digest"]), ("uploaded", row["bytes"], "sha256:" + row["sha256"]), "remote release asset differs: " + name)
        if name in JSON_ASSETS:
            same(path.read_bytes(), (repo / E / name).read_bytes(), "release JSON differs from Git evidence: " + name)
    # Compare TAR files directly to Git blob identities at the publication commit.
    prefixes = [str(E), str(Q), str(D / "evaluation"), str(D / "final-manifest.json"), str(D / "final-hashes.sha256"), str(D / "README.md")]
    raw = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "-z", target, "--", *prefixes])
    blobs = {}
    for entry in raw.split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            mode, kind, oid = meta.decode().split()
            require(mode in ("100644", "100755") and kind == "blob", "nonregular Git entry in release scope")
            blobs[path.decode()] = oid
    blobs.pop(str(Q / "release-assets.json"), None)  # Receipt is created after TAR construction.
    original_archives, original_trees = scratch / "original-archives", scratch / "original-trees"
    original_archives.mkdir()
    seen, zip_names = set(), set()
    object_format = git(repo, "rev-parse", "--show-object-format")
    with tarfile.open(args.release_dir / TAR_ASSET, "r:") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            require(not path.is_absolute() and ".." not in path.parts, "unsafe TAR member path")
            if member.isdir():
                continue
            require(member.isfile() and member.name not in seen, "nonregular/duplicate TAR member")
            seen.add(member.name)
            stream = archive.extractfile(member)
            require(stream is not None, "missing TAR file stream")
            if member.name in blobs:
                h = hashlib.new(object_format)
                h.update(f"blob {member.size}\0".encode())
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
                same(h.hexdigest(), blobs[member.name], "TAR file differs from tagged Git blob: " + member.name)
            elif path.parent == PurePosixPath("archives") and path.suffix == ".zip":
                zip_names.add(path.name)
                with (original_archives / path.name).open("wb") as output:
                    shutil.copyfileobj(stream, output)
            elif member.name == "prior-recovery.zip":
                with (scratch / member.name).open("wb") as output:
                    shutil.copyfileobj(stream, output)
            else:
                raise RuntimeError("unexpected TAR file: " + member.name)
    same(set(blobs), seen & set(blobs), "complete-evidence TAR omits committed evidence")
    expected_zips = {row["artifact_name"] + ".zip" for row in manifest["lanes"] if row["artifact"] is not None}
    same(zip_names, expected_zips, "original ZIP population missing from final release")
    for row in manifest["lanes"]:
        if row["artifact"] is not None:
            archive = original_archives / (row["artifact_name"] + ".zip")
            recovery.verify_zip(archive, row["artifact"])
            recovery.safe_extract(archive, original_trees / row["artifact_name"])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        reconstructed = recovery.make_manifest(argparse.Namespace(config=config, artifacts=original_trees,
            archives=original_archives, run_record=repo / Q / "provider-recovery/original-run.json",
            artifact_inventory=repo / Q / "provider-recovery/original-artifacts.json",
            jobs_record=repo / Q / "provider-recovery/original-jobs.json"))
    same(reconstructed, manifest, "release originals do not reconstruct the frozen recovery manifest")
    request = read(repo / Q / "recovery-request.json")
    prior_id = request["resume_recovery_run_id"]
    prior_custody = None
    if prior_id is not None:
        prior_custody = audit_prior_chain(repo, scratch, recovery, scratch / "prior-recovery.zip", prior_id, allocation_context)
    else:
        require("prior-recovery.zip" not in seen, "unsolicited prior recovery ZIP")
    return {"pass": True, "assets": len(expected), "tag_commit": target, "release_id": published["id"],
            "downloaded_sha256_verified": {name: row["sha256"] for name, row in expected.items()},
            "tagged_git_files_in_tar": len(blobs), "original_provider_zips_reconstructed": len(expected_zips),
            "zero_call_identities_rederived": sum(row["status"] == "unstarted" for row in manifest["lanes"]),
            "prior_recovery_custody": prior_custody}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--published-release", type=Path)
    parser.add_argument("--release-target", help="Exact expected bot commit; fetched local tag must resolve to it")
    parser.add_argument("--require-main", action="store_true", help="Require HEAD to equal the independently fetched origin/main")
    parser.add_argument("--require-docs", action="store_true", help="Require final RAG and dated measured journal signposts")
    args = parser.parse_args()
    repo = args.repo.resolve()
    require(repo not in args.out.resolve().parents, "audit output must remain outside the tracked repository")
    same(git(repo, "rev-parse", "HEAD"), args.expected_head, "audit worktree is not at the requested commit")
    same(git(repo, "status", "--porcelain", "--untracked-files=no"), "", "audit worktree has tracked modifications")
    # Offline is enforced in addition to all frozen validators' own offline flags.
    socket.create_connection = no_network
    socket.socket.connect = no_network
    socket.socket.connect_ex = no_network
    os.environ.pop("NVIDIA_API_KEY", None)
    sys.path.insert(0, str(repo / "scripts"))
    provider_api = importlib.import_module("village_001b_generate_member")
    provider_api.call_api = no_network
    isolated = importlib.import_module("village_001e_isolated")
    operator = importlib.import_module("village_001e_recovery_operator")
    recovery = importlib.import_module("village_001e_recover_provider_budget")
    asm = importlib.import_module("village_001e_assemble")
    request, manifest, config = operator.request()
    same(sha(repo / Q / "provider-recovery/manifest.json"), MANIFEST_SHA, "prelaunch recovery manifest differs from independently frozen hash")
    cfg = read(config)
    active_allocation = None
    if "allocation_manifest_sha256" in request:
        extension = importlib.import_module("village_001e_reallocate_provider_budget")
        allocation_path = repo / Q / "provider-recovery/allocation-extension.json"
        same(request["allocation_manifest_sha256"], ALLOCATION_SHA, "allocation differs from independently reviewed final freeze")
        same(sha(allocation_path), request["allocation_manifest_sha256"], "active allocation request binding changed")
        allocation, original, allocation_config = extension.load_allocation(allocation_path)
        same(original, manifest, "extension replaced the original recovery manifest")
        same(allocation_config, config, "extension replaced the qualification config")
        budget = allocation["budget"]
        same(tuple(budget[key] for key in ("baseline_logical_requests_archived", "baseline_transport_launches_archived",
             "completed_lanes", "completed_lane_logical_requests", "maximum_total_logical_requests", "unallocated_logical_slots")),
             (673, 674, 41, 630, 1059, 45), "extension differs from independently audited snapshot/allocation")
        same(allocation["prior_recovery_artifact"]["digest"], "sha256:" + BASELINE_ZIP_SHA, "extension baseline ZIP changed")
        active_allocation = (allocation_path, allocation, original, config, extension)
    generation = read(repo / Q / "generation-custody.json")
    report = {"schema": "village-001e-independent-acceptance-v1", "issue": 14,
              "audit_head": args.expected_head, "audit_script_sha256": sha(Path(__file__)),
              "source_freeze_verified": True, "generation_inventory_files": len(generation["files"]),
              "recovery_manifest_sha256": MANIFEST_SHA, "evidence_acceptance": "not_evaluated",
              "release_readback": {"pass": False, "reason": "not supplied"}, "new_provider_calls": 0,
              "new_optimizer_steps": 0, "new_local_generations": 0, "closure_ready": False}
    if active_allocation:
        report["allocation_extension"] = {"sha256": sha(allocation_path), "budget": budget,
            "missing_native_v1_receipt": {"lane": "F19", "logical_requests": 3, "transport_launches": 3,
                "accounted_by": "frozen raw baseline inventory; exact snapshot ZIP is verified during release readback"}}
    if not args.source_only:
        summary = read(repo / E / "summary.json")
        provenance = read(repo / E / "provenance.json")
        receipt = read(repo / Q / "release-assets.json")
        same((receipt["schema"], receipt["release_tag"], receipt["new_optimizer_steps"]), ("village-001e-isolated-release-v1", TAG, 0), "release receipt identity changed")
        same((receipt["run_id"], receipt["source_head"]), (provenance["workflow_run_id"], provenance["source_commit"]), "execution provenance differs across release/evidence")
        same(receipt["recovery_manifest_sha256"], MANIFEST_SHA, "release recovery identity drift")
        if active_allocation:
            same(receipt.get("allocation_manifest_sha256"), sha(allocation_path), "release allocation identity drift")
        else:
            require("allocation_manifest_sha256" not in receipt, "release contains an undeclared allocation extension")
        same(receipt["global_accounting_sha256"], sha(repo / Q / "provider-recovery/accounting.json"), "release accounting binding drift")
        same(provenance["ecology_used_as_hard_population_gate"], False, "ecology was made an undeclared gate")
        same((repo / E / "hashes.sha256").read_text(), asm.hash_inventory(repo / E), "final evidence hash inventory differs")
        same((repo / D / "final-hashes.sha256").read_text(), asm.dataset_hash_inventory(repo / D), "whole-dataset hash inventory differs")
        same(isolated.tree_files(repo / E), isolated.tree_files(repo / D / "evaluation"), "promoted evaluation differs from final evidence")
        same(isolated.tree_files(repo / Q / "generation/responses"), isolated.tree_files(repo / E / "final-responses"), "final generation source copy differs")
        same(isolated.tree_files(repo / Q / "generation/ecology"), isolated.tree_files(repo / E / "ecology/sessions"), "final ecology generation source copy differs")
        with tempfile.TemporaryDirectory(prefix="independent-stage-e-") as temporary:
            scratch = Path(temporary)
            lanes = scratch / "provider-lanes"
            invocations = 0
            extension_invocations = 0
            extension_receipt_gaps = {}
            for row in manifest["lanes"]:
                source = repo / E / ("tournament-scenarios" if row["kind"] == "tournament" else "ecology/judgments") / row["lane"]
                shutil.copytree(source, lanes / row["artifact_name"])
                receipts = [read(path) for path in sorted(source.glob("budget-recovery-execution-*.json"))]
                if active_allocation:
                    current = extension.lane_record(allocation, row["kind"], row["lane"])
                    baseline_receipt_names = {name for name in current["baseline_files"] if re.fullmatch(r"budget-recovery-execution-\d+\.json", name)}
                    same({p.name for p in source.glob("budget-recovery-execution-*.json")}, baseline_receipt_names,
                         "native v1 invocation receipts were added or lost after the frozen extension boundary")
                else:
                    require(receipts, "missing native recovery invocation receipt")
                for native in receipts:
                    same(native["manifest_sha256"], MANIFEST_SHA, "native receipt manifest drift")
                    same(native["original_files_preserved"], True, "native receipt records lost originals")
                    same(native["effective_logical_limit"], row["effective_logical_limit"], "native receipt budget drift")
                    if native["offline"]:
                        same((native["new_transport_launches"], native["new_logical_requests"], native["new_provider_calls"]), (0, 0, 0), "offline receipt records spending")
                actual, logical = recovery.labels(source)
                if active_allocation:
                    expected_gap = 3 if row["lane"] == "F19" else 0
                    same(current["baseline_logical_count"] - row["original_logical_count"] - sum(r["new_logical_requests"] for r in receipts),
                         expected_gap, "v1 logical receipt gap differs from the frozen cancellation snapshot")
                    same(len(current["baseline_actual_labels"]) - len(row["original_request_labels"]) - sum(r["new_transport_launches"] for r in receipts),
                         expected_gap, "v1 transport receipt gap differs from the frozen cancellation snapshot")
                    newer = [read(p) for p in sorted(source.glob("allocation-extension-execution-*.json"))]
                    require(newer, "final lane lacks an allocation invocation receipt")
                    for native in newer:
                        same(native["allocation_manifest_sha256"], sha(allocation_path), "allocation receipt source identity drift")
                        same(native["effective_logical_limit"], current["effective_logical_limit"], "allocation receipt cap drift")
                        same(native["baseline_logical_requests_archived"], current["baseline_logical_count"], "allocation receipt reset its baseline")
                        same(native["previous_source_files_preserved"], True, "allocation receipt records lost source bytes")
                        require(native["status"] in ("completed", "failed", "interrupted"), "unknown allocation receipt status")
                        if native["status"] == "completed":
                            same(native["terminal_verified"], True, "completed allocation receipt did not verify its terminal")
                        if native["offline"]:
                            same((native["new_logical_requests"], native["new_transport_launches"], native["new_provider_calls_attempted"]),
                                 (0, 0, 0), "offline allocation receipt records spending")
                    gap = {"logical": len(logical) - current["baseline_logical_count"] - sum(r["new_logical_requests"] for r in newer),
                           "transports": len(actual) - len(current["baseline_actual_labels"]) - sum(r["new_transport_launches"] for r in newer)}
                    require(min(gap.values()) >= 0, "allocation receipts overstate final archived raw requests")
                    extension_receipt_gaps[row["lane"]] = gap
                    extension_invocations += len(newer)
                else:
                    same(sum(r["new_logical_requests"] for r in receipts), len(logical) - row["original_logical_count"], "recovery receipt logical deltas do not reconcile")
                    same(sum(r["new_transport_launches"] for r in receipts), len(actual) - len(row["original_request_labels"]), "recovery receipt transport deltas do not reconcile")
                invocations += len(receipts)
            accounting = (extension.verify_complete(allocation_path, allocation, manifest, config, lanes) if active_allocation else
                          recovery.verify_complete(repo / Q / "provider-recovery/manifest.json", manifest, config, lanes))
            same(accounting, read(repo / Q / "provider-recovery/accounting.json"), "independent provider replay/accounting differs")
            print("Verified 48 offline provider lanes and native recovery accounting", flush=True)
            replay_assembler(repo, scratch, asm, isolated, cfg, provenance)
            print("Verified strict reassembly, all 61 native generation chains, and whole dataset bytes", flush=True)
            report.update(evidence_acceptance="pass", strict_assembly_replay="byte-identical", generation_lanes=61,
                          generation_rows=1068, provider_lanes=48, recovery_invocation_receipts=invocations,
                          allocation_invocation_receipts=extension_invocations,
                          actual_logical_requests=accounting["actual_logical_requests"],
                          actual_transport_launches=accounting["actual_transport_launches"],
                          actual_supplemental_logical_requests=accounting["actual_supplemental_logical_requests"],
                          final_evidence_files=len(isolated.tree_files(repo / E)),
                          dataset_inventory_records=len((repo / D / "final-hashes.sha256").read_text().splitlines()),
                          summary_sha256=sha(repo / E / "summary.json"), model_registry_sha256=sha(repo / E / "model-registry.json"),
                          independent_metrics=independently_recompute_gates(repo, cfg, summary))
            if args.release_dir or args.published_release or args.release_target:
                report["release_readback"] = audit_release(args, repo, scratch, recovery, manifest, config, receipt, active_allocation)
            if active_allocation:
                report["allocation_extension"]["actual_post_allocation_logical_requests"] = accounting["actual_post_allocation_logical_requests"]
                report["allocation_extension"]["native_receipt_gaps"] = extension_receipt_gaps
                for lane, gap in extension_receipt_gaps.items():
                    if any(gap.values()):
                        require(report["release_readback"]["pass"], "interrupted allocation receipt gap requires verified release custody")
                        inherited = report["release_readback"]["prior_recovery_custody"]["verified_prior_runs"]
                        require(any(run["extension_receipt_gaps"].get(lane) == gap for run in inherited.values()),
                                "final allocation receipt gap lacks an identical hash-bound interrupted prior snapshot")
        registry = read(repo / E / "model-registry.json")
        same(registry["release_tag"], MODEL_TAG, "registry substituted a different model release")
        same(len(registry["models"]), 36, "registry omitted failed models")
        report["model_registry"] = {"models": 36, "release_tag": MODEL_TAG, "unchanged_D_assets_verified_by_assembler": True}
        report["ecology"] = {"sessions": summary["ecology"]["sessions"], "turns": 180, "lineage_audits": 12, "hard_population_gate": False}
    main_ok = git(repo, "rev-parse", "refs/remotes/origin/main") == args.expected_head if args.require_main else False
    if args.require_main:
        require(main_ok, "HEAD differs from the independently fetched origin/main")
    rag, journal = (repo / "RAG.md").read_text(), (repo / "docs/MODEL_JOURNAL.md").read_text()
    docs_ok = str(E / "REPORT.md") in rag and "once #14 executes" not in rag and TAG in journal and "2026-11-19" in journal
    if args.require_docs:
        require(docs_ok, "final RAG/measured model-journal signposts are incomplete")
    report["main_readback"] = {"required": args.require_main, "pass": main_ok, "basis": "caller-refreshed local origin/main"}
    report["documentation_signposts"] = {"required": args.require_docs, "pass": docs_ok, "human_review_still_required": True}
    report["closure_ready"] = report["evidence_acceptance"] == "pass" and report["release_readback"]["pass"] and main_ok and docs_ok
    same(git(repo, "status", "--porcelain", "--untracked-files=no"), "", "audit modified tracked repository files")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("audit_head", "evidence_acceptance", "release_readback", "closure_ready")}, indent=2))


if __name__ == "__main__":
    main()
