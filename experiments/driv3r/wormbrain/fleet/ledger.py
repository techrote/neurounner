#!/usr/bin/env python3
"""Opt-in DRIV3R four-inference-worm lease planner and original-byte auditor.
NO model loading, model training, API dispatch, native mutation, or provider change.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

REQUEST = "driv3r.wormfleet.request/1"
EPISODE = "driv3r.wormfleet.episode/1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
IDENT = re.compile(r"[a-zA-Z0-9_.-]{1,100}\Z")
KINDS = {"native_model", "model_scored_w0"}


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(x):
    return isinstance(x, str) and SHA.fullmatch(x) is not None


def safe(x):
    return isinstance(x, str) and IDENT.fullmatch(x) is not None


def canonical(x):
    return (json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest(x):
    return hashlib.sha256(x).hexdigest()


def validate_request(r):
    require(isinstance(r, dict) and r.get("schema") == REQUEST, "request schema")
    require(safe(r.get("campaign")) and safe(r.get("wrangler")), "campaign/wrangler ID")
    require(sha(r.get("source_sha256")) and sha(r.get("host_source_sha256")), "pinned source SHA256")
    require(r.get("training") is False and r.get("paid_api") is False and
            r.get("native_permissions_remain_host_owned") is True and
            r.get("automatically_expand") is False, "unsafe permissions")
    require(r.get("max_runners") == 2 and r.get("models_per_runner") == 2,
            "initial ceiling exactly two runners x two inference models")
    es = r.get("episodes")
    require(isinstance(es, list) and len(es) == 4, "exactly four episodes")
    names, seeds, slots = set(), set(), Counter()
    for e in es:
        require(isinstance(e, dict), "bad episode")
        require(safe(e.get("id")) and e["id"] not in names, "duplicate/unsafe episode id")
        names.add(e["id"])
        require(type(e.get("seed")) is int and e["seed"] >= 0 and
                e["seed"] not in seeds, "nonunique seed")
        seeds.add(e["seed"])
        require(type(e.get("runner_slot")) is int and e["runner_slot"] in (0, 1), "invalid slot")
        slots[e["runner_slot"]] += 1
        require(e.get("evidence_class") in KINDS, "invalid evidence kind")
        require(isinstance(e.get("model_id"), str) and bool(e["model_id"].strip())
                and isinstance(e.get("model_revision"), str) and bool(e["model_revision"].strip())
                and sha(e.get("model_sha256")) and sha(e.get("checkpoint_sha256")),
                "model/revision/weights/checkpoint must be pinned")
        max_ticks = 36 if e["evidence_class"] == "native_model" else 128
        require(type(e.get("decision_budget")) is int and 1 <= e["decision_budget"] <= max_ticks,
                "native 36 / source-faithful synthetic 128 decision bound")
        require(type(e.get("material_budget")) is int and
                0 <= e["material_budget"] <= min(e["decision_budget"], 36 if max_ticks == 36 else 128),
                "material budget")
        require(e.get("allow_model_training") is False, "no optimization permitted")
    require(slots == Counter({0: 2, 1: 2}), "two models per runner required")


def plan(r):
    validate_request(r)
    fingerprint = digest(canonical(r))
    jobs = []
    for slot in (0, 1):
        leases = []
        for e in r["episodes"]:
            if e["runner_slot"] != slot:
                continue
            lease_id = "wf-" + digest(canonical({"request": fingerprint, "id": e["id"]}))[:28]
            leases.append({"lease_id": lease_id, **e})
        jobs.append({"runner_slot": slot, "leases": leases, "source_sha256": r["source_sha256"]})
    return {"schema": "driv3r.wormfleet.plan/1", "campaign": r["campaign"],
            "wrangler": r["wrangler"], "request_sha256": fingerprint,
            "runner_jobs": jobs, "total_leases": 4, "dispatch_permitted": False,
            "note": "This plans immutable leases; it never dispatches runners or opens optimizer gates."}


def check_file(root, desc):
    require(isinstance(desc, dict), "missing original-byte descriptor")
    rel, checksum = desc.get("path"), desc.get("sha256")
    require(isinstance(rel, str) and rel and not Path(rel).is_absolute() and
            all(part not in ("..", ".", "") for part in Path(rel).parts),
            "invalid original artifact path")
    file = root / rel
    require(file.is_file() and sha(checksum), "missing original bytes/hash")
    data = file.read_bytes()
    require(digest(data) == checksum, "original file digest mismatch")
    return data


def audit(r, p, result_dir):
    """Original receipt consistency only; remote PID liveness needs external auditing."""
    validate_request(r)
    expected = plan(r)
    require(p == expected, "plan/request mismatch")
    require(result_dir.is_dir(), "missing original result folder")
    wanted = {e["id"]: (e, x["lease_id"]) for job in expected["runner_jobs"]
              for x in job["leases"] for e in r["episodes"] if e["id"] == x["id"]}
    seen, pid_pairs, jobs, verified = set(), set(), set(), []
    for file in sorted(result_dir.glob("*/episode.json")):
        x = json.loads(file.read_text())
        require(isinstance(x, dict) and x.get("schema") == EPISODE, "episode schema")
        eid = x.get("episode_id")
        require(eid in wanted and eid not in seen, "unplanned or duplicate worm")
        seen.add(eid)
        e, lease = wanted[eid]
        require(x.get("campaign") == r["campaign"] and x.get("wrangler") == r["wrangler"] and
                x.get("request_sha256") == expected["request_sha256"] and
                x.get("lease_id") == lease and
                x.get("source_sha256") == r["source_sha256"] and
                x.get("host_source_sha256") == r["host_source_sha256"], "source/lease mismatch")
        for field in ("model_id", "model_revision", "model_sha256", "checkpoint_sha256",
                      "seed", "runner_slot", "evidence_class", "decision_budget"):
            require(x.get(field) == e[field], "unmatched field: " + field)
        require(x.get("status") == "completed" and x.get("optimizer_steps") == 0,
                "incomplete worm/optimizer disallowed")
        require(safe(x.get("job_id")) and safe(x.get("run_id")) and
                type(x.get("pid")) is int and x["pid"] > 0, "invalid runner job/PID")
        pid_key = (x["job_id"], x["pid"])
        require(pid_key not in pid_pairs, "same PID claimed twice")
        pid_pairs.add(pid_key)
        jobs.add(x["job_id"])
        start, end = x.get("forward_start_ns"), x.get("forward_end_ns")
        require(type(start) is int and type(end) is int and 0 < start < end,
                "invalid original model-forward interval")
        forwards, decisions = x.get("model_forward_count"), x.get("decisions_applied")
        require(type(forwards) is int and type(decisions) is int and
                1 <= forwards <= e["decision_budget"] and 0 <= decisions <= forwards,
                "missing/excess model decisions")
        require(type(x.get("native_material_edits")) is int and
                0 <= x["native_material_edits"] <= e["material_budget"], "material budget exceeded")
        require(type(x.get("recoveries")) is int and 0 <= x["recoveries"] <= 8,
                "recovery budget exceeded")
        require(type(x.get("wall_ms")) is int and x["wall_ms"] > 0 and
                type(x.get("rss_peak_kib")) is int and x["rss_peak_kib"] > 0,
                "wall/RSS missing")
        require(x.get("raw_model_responses_retained") is True and
                x.get("legal_action_receipts_checked") is True, "model/legal receipts absent")
        native = e["evidence_class"] == "native_model"
        require(x.get("native_execution") is native, "synthetic/native mislabeled")
        if native:
            require(x.get("independent_native_replay_verified") is True and
                    x.get("original_material_readbacks_verified") is True,
                    "native original replay/readback not qualified")
        else:
            require(x.get("source_faithful_w0") is True, "W0 fidelity missing")
        originals = x.get("originals")
        require(isinstance(originals, dict) and set(originals) ==
                {"png", "raw_decisions_jsonl", "receipt_jsonl"}, "missing original files")
        for description in originals.values():
            content = check_file(file.parent, description)
            if description is originals["png"]:
                require(content.startswith(b"\x89PNG\r\n\x1a\n"), "not PNG original")
        verified.append(x)
    require(seen == set(wanted), "four independent original episodes incomplete")
    require(len(jobs) == 2 and Counter(x["job_id"] for x in verified) ==
            Counter({job: 2 for job in jobs}), "not 2 jobs / 2 resident PIDs per job")
    require({x["runner_slot"] for x in verified} == {0, 1} and all(
        len({x["runner_slot"] for x in verified if x["job_id"] == job}) == 1 for job in jobs),
        "runner slot misattribution")
    # CLOCK_MONOTONIC epochs from different GitHub VMs are NOT comparable.
    # Only compare monotonic intervals measured by processes on the SAME runner.
    local_intervals = {}
    for job in sorted(jobs):
        pair = [x for x in verified if x["job_id"] == job]
        begin = max(x["forward_start_ns"] for x in pair)
        end = min(x["forward_end_ns"] for x in pair)
        require(end - begin >= 1_000_000_000,
                "no same-runner overlap of two resident inference episode intervals")
        local_intervals[job] = round((end-begin)/1e9, 6)
    return {"schema": "driv3r.wormfleet.audit/1", "campaign": r["campaign"],
            "request_sha256": expected["request_sha256"],
            "original_episodes_checked": 4, "claimed_two_runner_jobs": 2,
            "claimed_four_distinct_pid_job_pairs": 4,
            "claimed_same_runner_two_resident_overlap_seconds_by_job": local_intervals,
            "claimed_four_forward_overlap_seconds": None,
            "cross_runner_wall_clock_overlap_verified": False,
            "independently_verified_current_active_models": 0,
            "reason": "Only within-runner monotonic timestamps can be compared. Cross-VM four-way overlap, GitHub job origin and PID liveness need independent UTC/remote witnesses.",
            "source_classes": sorted({x["evidence_class"] for x in verified}),
            "decision_count": sum(x["decisions_applied"] for x in verified),
            "material_edit_count": sum(x["native_material_edits"] for x in verified),
            "recovery_count": sum(x["recoveries"] for x in verified),
            "episodes": [{k: x[k] for k in ("episode_id", "job_id", "run_id", "pid",
                         "model_id", "seed", "decisions_applied", "native_material_edits",
                         "recoveries", "wall_ms", "rss_peak_kib")} for x in verified]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "audit"))
    p.add_argument("--request", required=True, type=Path)
    p.add_argument("--plan", type=Path)
    p.add_argument("--results", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)
    try:
        request = json.loads(a.request.read_text())
        if a.command == "plan":
            output = plan(request)
        else:
            require(a.plan is not None and a.results is not None,
                    "audit requires immutable --plan and original --results")
            output = audit(request, json.loads(a.plan.read_text()), a.results)
        require(not a.output.exists(), "refuse to overwrite immutable output")
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_bytes(canonical(output))
        print(json.dumps({"status": "written", "file": str(a.output),
                          "schema": output["schema"]}))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print("FLEET REFUSED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
