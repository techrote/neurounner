#!/usr/bin/env python3
"""Independent, offline readback of DETECTIVE-001B source-bound smoke evidence."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.detective_001b_pilot import (
    CASE, CONFIG, MANIFEST, ROOT, canonical, check_config, digest_bytes, file_sha,
    grade_script, messages_first, messages_followup, messages_script, plan
)


class EvidenceError(ValueError):
    pass


def need(ok: bool, message: str) -> None:
    if not ok:
        raise EvidenceError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_dataset(dataset: Path) -> dict:
    cfg, case, release = check_config()
    p = plan()
    names = {
        "README.md", "manifest.json", "raw/run.json",
        "provenance/source-artifacts.json", "provenance/hashes.sha256"
    }
    present = {x.relative_to(dataset).as_posix() for x in dataset.rglob("*") if x.is_file()}
    need(present == names, f"archive files changed: {present ^ names}")
    inventory = (dataset / "provenance/hashes.sha256").read_text(encoding="utf-8").splitlines()
    actual = [
        f"{file_sha(dataset / name)}  {name}"
        for name in sorted(names - {"provenance/hashes.sha256"})
    ]
    need(inventory == actual, "archive hash inventory mismatch")
    run = read_json(dataset / "raw/run.json")
    manifest = read_json(dataset / "manifest.json")
    provenance = read_json(dataset / "provenance/source-artifacts.json")
    need(run.get("schema") == "detective-real-smoke-run-v1", "run schema mismatch")
    need(run.get("experiment") == "DETECTIVE-001B", "run experiment mismatch")
    need(run.get("admission") == "EXPLORATORY_NOT_ADMITTED"
         and run.get("behavioral_truth_quality") == "UNSCORED"
         and run.get("complete") is True, "smoke incorrectly labeled as admitted or incomplete")
    need(run["config_sha256"] == file_sha(CONFIG)
         and run["case_sha256"] == file_sha(CASE)
         and run["release_manifest_sha256"] == file_sha(MANIFEST),
         "fixture/manifest/config ancestry drift")
    need(run["plan_sha256"] == digest_bytes(canonical(p).encode("utf-8")),
         "matched plan identity drift")
    source_script = ROOT / "scripts/detective_001b_pilot.py"
    need(run["runner_script_sha256"] == file_sha(source_script), "runner source drift")
    need(run["sources"]["base"] == cfg["base"]
         and run["sources"]["base_weights_sha256_verified"] == cfg["base"]["weights_sha256"],
         "frozen backbone proof mismatch")

    target_lanes = p["lanes"]
    need(len(run["lanes"]) == 3, "unexpected lane count")
    need([x["lane_id"] for x in run["lanes"]] == target_lanes, "lane IDs/order changed")
    assets = {a["member_id"]: a for a in release["assets"]}
    for lane in run["lanes"]:
        lid = lane["lane_id"]
        need(lane["base_weights_sha256"] == cfg["base"]["weights_sha256"], "base hash changed")
        if lid == "BASE":
            need(lane["kind"] == "untuned_smollm2_control"
                 and lane["adapter_directory_sha256"] is None
                 and lane["adapter_archive_sha256"] is None, "base control is not pristine")
        else:
            expected = assets[lid]
            need(lane["kind"] == "village_001c_sft_lora"
                 and lane["adapter_archive_sha256"] == expected["sha256"]
                 and lane["adapter_directory_sha256"] == expected["adapter_directory_sha256"]
                 and lane["member_contract_sha256"] == release["member_contract_sha256"],
                 "not the predeclared SFT cartridge")

    run_id = run["repository"]["run_id"]
    need(isinstance(run_id, str) and run_id.isdecimal(), "actions run ID absent")
    need(re.fullmatch(r"[0-9a-f]{40}", run["repository"]["head_sha"]) is not None,
         "head SHA missing")
    need(provenance["source_head_sha"] == run["repository"]["head_sha"]
         and provenance["actions_run_id"] == run_id
         and provenance["script_sha256"] == file_sha(source_script),
         "provenance mismatch")

    ordered = []
    count = 0
    valid = 0
    solved = 0
    token_total = 0
    for lid in target_lanes:
        for idx in range(2):
            first = next((x for x in run["turns"] if x.get("id") == f"{lid}-I{idx+1:02d}A"), None)
            second = next((x for x in run["turns"] if x.get("id") == f"{lid}-I{idx+1:02d}B"), None)
            need(first is not None and second is not None, "missing interview pair")
            need(first["request"]["messages"] == messages_first(cfg, case, idx), "unmatched blind prompt")
            need(second["request"]["messages"] ==
                 messages_followup(cfg, case, idx, first["response"]), "followup not tied to raw first")
            need(first["turn"] == 1 and second["turn"] == 2, "turn order invalid")
            ordered.extend([first["id"], second["id"]])
        if lid == "BASE":
            for task in cfg["scripter"]["tasks"]:
                script = next((x for x in run["turns"] if x.get("id") == f"BASE-{task['id']}"), None)
                need(script is not None and script.get("type") == "scripter", "missing scripted control")
                need(script["request"]["messages"] == messages_script(cfg, case, task),
                     "scripter saw an unapproved observation")
                need(script["grade"] == grade_script(case, task, script["response"]),
                     "scripter grade not reproducible")
                valid += int(script["grade"]["schema_valid"])
                solved += int(script["grade"]["task_success"])
                ordered.append(script["id"])
    need([x["id"] for x in run["turns"]] == ordered
         and len(ordered) == 14 and len(set(ordered)) == 14, "missing/duplicate/out-of-order turns")
    for turn in run["turns"]:
        need(turn["generated_tokens"] == len(turn["generated_token_ids"])
             and 0 <= turn["generated_tokens"] <= 80, "generated token count inconsistent")
        need(all(type(tok) is int and 0 <= tok < 131072 for tok in turn["generated_token_ids"]),
             "non-token payload in output IDs")
        need(isinstance(turn["response"], str)
             and isinstance(turn["response_raw_with_special_tokens"], str), "raw output lost")
        need(turn["request"]["max_new_tokens"] == 80
             and turn["request"]["do_sample"] is False
             and 0 < turn["request"]["prompt_token_count"] <= 512, "request limits drift")
        need(re.fullmatch(r"[a-f0-9]{64}", turn["request"]["input_token_sha256"]) is not None,
             "input token digest absent")
        if turn["type"] == "interview":
            need(turn["request"]["messages"][0]["role"] == "system", "untrusted first role")
        token_total += turn["generated_tokens"]
        count += turn["type"] == "interview"
    need(token_total == run["total_generated_tokens"]
         and token_total <= cfg["budget"]["max_total_generated_tokens"], "total token budget breached")
    metrics = manifest["metrics"]
    need(metrics == {"lanes": 3, "dialogue_turns": count, "scripter_turns": 2,
                     "generated_tokens": token_total, "valid_scripts": valid,
                     "successful_scripts": solved}, "manifest metrics not source-derived")
    need(manifest["qualification"] == "EXPLORATORY_NOT_ADMITTED"
         and manifest["suite_sha256"] == run["plan_sha256"]
         and manifest["source_config_sha256"] == file_sha(CONFIG)
         and manifest["run_source_sha"] == run["repository"]["head_sha"], "archive admission mislabeled")
    return {
        "verdict": "OFFLINE_SOURCE_READBACK_PASS",
        "models": target_lanes, "turns": len(ordered),
        "generated_tokens": token_total, "script_json_valid": valid,
        "script_tasks_successful": solved,
        "training_claim": False, "behavioral_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_dataset(args.dataset), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
