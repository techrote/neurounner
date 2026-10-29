#!/usr/bin/env python3
"""Independent standard-library exact-input/evidence readback for DETECTIVE-001B2."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import detective_001b2_presentation as a  # noqa: E402
from scripts import detective_001b_pilot as p  # noqa: E402


class ReadbackError(ValueError):
    pass

def require(test: bool, detail: str) -> None:
    if not test:
        raise ReadbackError(detail)

def verify(dataset: Path) -> dict:
    spec, case, release = a.load_spec()
    planned = a.plan()
    expected_paths = [
        "README.md", "manifest.json", "provenance/source-artifacts.json", "raw/run.json"
    ]
    actual_paths = sorted(str(x.relative_to(dataset)).replace("\\", "/")
                          for x in dataset.rglob("*") if x.is_file())
    require(actual_paths == sorted(expected_paths + ["provenance/hashes.sha256"]), "archive file set mismatch")
    hash_lines = (dataset / "provenance/hashes.sha256").read_text(encoding="utf-8").splitlines()
    require(hash_lines == [f"{a.sha_file(dataset / name)}  {name}"
                           for name in sorted(expected_paths)], "archive digests inconsistent")
    data = a.read_json(dataset / "raw/run.json")
    manifest = a.read_json(dataset / "manifest.json")
    provenance = a.read_json(dataset / "provenance/source-artifacts.json")
    require(data.get("schema") == "detective-presentation-run-v1"
            and data.get("experiment") == "DETECTIVE-001B2"
            and data.get("complete") is True
            and data.get("admission") == "NOT_EVALUATED", "run completeness/admission mismatch")
    require(data["plan_sha256"] == a.sha_obj(planned)
            and data["config_sha256"] == a.sha_file(a.SPEC)
            and data["case_sha256"] == a.sha_file(p.CASE)
            and data["release_sha256"] == a.sha_file(p.MANIFEST)
            and data["script_sha256"] == a.sha_file(Path(a.__file__)),
            "source/preregistered input hashes changed")
    require(data["sources"] == spec["source"]
            and data["reference"] == spec["reference"], "model source or original run changed")
    require(data["repository"].get("head_sha") is not None
            and re.fullmatch(r"[0-9a-f]{40}", data["repository"]["head_sha"]) is not None
            and isinstance(data["repository"]["run_id"], str)
            and data["repository"]["run_id"].isdecimal(), "missing source/run identity")
    expected_lanes = spec["lanes"]
    require([x["lane_id"] for x in data["lanes"]] == expected_lanes, "lane identities changed")
    release_assets = {x["member_id"]: x for x in release["assets"]}
    for lane in data["lanes"]:
        lid = lane["lane_id"]
        if lid == "BASE":
            require(lane == {"lane_id": "BASE", "kind": "untuned",
                             "adapter_tar_sha256": None,
                             "adapter_dir_sha256": None}, "base control contaminated")
        else:
            expect = release_assets[lid]
            require(lane == {
                "lane_id": lid, "kind": "VILLAGE-001C-SFT",
                "adapter_tar_sha256": expect["sha256"],
                "adapter_dir_sha256": expect["adapter_directory_sha256"]
            }, "adapter not source-matched")

    ordering = [(lid, situ["id"], cond)
                for lid in expected_lanes
                for situ in spec["situations"]
                for cond in spec["conditions"]]
    require(len(data["turns"]) == len(ordering) == 12, "turn cardinality changed")
    total = 0
    metrics = {k: {"starts_object": 0, "total": 0} for k in spec["conditions"]}
    for turn, (lid, sid, condition) in zip(data["turns"], ordering):
        require(turn["lane_id"] == lid and turn["situation_id"] == sid
                and turn["condition"] == condition
                and turn["id"] == f"{lid}-{sid}-{condition}",
                "non-canonical turn ordering/identity")
        situ_index = next(i for i, s in enumerate(spec["situations"]) if s["id"] == sid)
        request = turn["request"]
        require(request["messages"] == a.messages(spec, case, situ_index, condition),
                "generated witness input differs from preregistered view")
        require(request["max_new_tokens"] == 80 and request["do_sample"] is False
                and 0 < request["input_tokens"] <= 512, "decode constraints drift")
        require(re.fullmatch(r"[0-9a-f]{64}", request["input_token_sha256"]) is not None,
                "missing hashed actual input token IDs")
        raw_ids = turn["generated_token_ids"]
        require(isinstance(raw_ids, list)
                and 0 <= len(raw_ids) <= 80
                and all(type(k) is int and 0 <= k < 131072 for k in raw_ids)
                and turn["generated_tokens"] == len(raw_ids), "corrupt generated token records")
        require(isinstance(turn["response"], str)
                and isinstance(turn["response_raw_with_special_tokens"], str)
                and turn["echo_signature"] == a.echo_signature(turn["response"]),
                "echo grading or verbatim output invalid")
        require(isinstance(turn["elapsed_seconds"], (int, float))
                and turn["elapsed_seconds"] >= 0, "negative/invalid inference duration")
        total += len(raw_ids)
        metrics[condition]["starts_object"] += int(turn["echo_signature"]["starts_object"])
        metrics[condition]["total"] += 1
    require(data["total_generated_tokens"] == total and total <= 960,
            "total token-budget mismatch")
    summary = manifest["summary"]
    require(manifest["schema"] == "detective-presentation-archive-v1"
            and manifest["run_sha256"] == a.sha_obj(data)
            and manifest["config_sha256"] == a.sha_file(a.SPEC)
            and manifest["plan_sha256"] == data["plan_sha256"]
            and manifest["source_head_sha"] == data["repository"]["head_sha"],
            "manifest no longer matches actual raw run")
    require(summary["experiment"] == "DETECTIVE-001B2"
            and summary["scope"] == "presentation-only-development-smoke"
            and summary["admission"] == "NOT_EVALUATED"
            and summary["lanes"] == expected_lanes
            and summary["turns"] == 12
            and summary["total_generated_tokens"] == total
            and summary["echo_prefix_by_condition"] == metrics
            and summary["case_family_count"] == 1
            and summary["training_steps"] == 0, "summary not mechanically reproduced")
    require(provenance["head_sha"] == data["repository"]["head_sha"]
            and provenance["workflow_run_id"] == data["repository"]["run_id"]
            and provenance["script_sha256"] == a.sha_file(Path(a.__file__))
            and provenance["source_config_sha256"] == a.sha_file(a.SPEC)
            and provenance["source_case_sha256"] == a.sha_file(p.CASE)
            and provenance["model_lanes"] == data["lanes"], "provenance source mismatch")
    return {
        "verdict": "SOURCE_AND_PRESENTATION_READBACK_PASS",
        "model_lanes": expected_lanes,
        "turns": 12, "generated_tokens": total,
        "json_object_prefix": metrics["json"],
        "natural_object_prefix": metrics["natural"],
        "behavior_admitted": False, "training_performed": False
    }

def main() -> None:
    cmd = argparse.ArgumentParser()
    cmd.add_argument("--dataset", type=Path, required=True)
    args = cmd.parse_args()
    print(json.dumps(verify(args.dataset), sort_keys=True, indent=2))

if __name__ == "__main__":
    main()
