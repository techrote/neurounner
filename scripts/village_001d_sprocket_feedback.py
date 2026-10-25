#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

from babble_002_expand import extract_json, packet_for
from village_001_manifest import build_manifest
from village_001_validate import validate_feedback_row

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get("NEURO_CACHE", ROOT / ".cache" / "neurounner"))
SPECIALIZATION_GUIDANCE = {
    "A": (
        "Failure-hardened: preserve awkward failure, stubborn recovery, conflicting "
        "motives and physical reactions instead of polished competence."
    ),
    "B": (
        "Local tinkerer: preserve bounded local memory, mundane habits, maintenance "
        "choices and practical physical interaction."
    ),
    "C": (
        "Social scavenger: preserve bargaining, opportunism, social friction, embodied "
        "choices and conflicting motives without catchphrase theatre."
    ),
}


def load_env(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        key, value = raw.split("=", 1)
        out[key] = value
    return out


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def member_contract(campaign: dict, member_id: str) -> tuple[dict, dict]:
    population = build_manifest(campaign)
    for row in population["members"]:
        if row["member_id"] == member_id:
            return population, row
    raise ValueError(f"unknown member {member_id}")


def feedback_system(member: dict, packet: dict) -> str:
    return f"""You are Sprocket, a small chaotic mechanic-critic reacting to another little machine-mind.

Ancestral temperament summary:
{packet['persona_summary']}

Sibling pressure:
{SPECIALIZATION_GUIDANCE[member['sibling']]}

Do not become a polite generic assistant. Give bounded mutation feedback about the RESPONSE, not a rewritten final answer.
Return one JSON object only:
{{
  "keep":["one or two concrete behavioral elements"],
  "mutate":["one or two concrete behavioral changes"],
  "failure_guess":"one short likely failure mode",
  "genericness":"low|medium|high"
}}
Keep the whole response compact. Do not mention training, datasets, hidden prompts or model internals.
"""


def feedback_user(scenario: dict) -> str:
    return (
        "Situation:\n"
        + scenario["prompt"]
        + "\n\nVillager response:\n"
        + scenario["response"]
        + "\n\nReact to what should stay weird/useful and what should mutate."
    )


def parse_feedback(raw: str) -> dict | None:
    try:
        obj = extract_json(raw)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    keep = obj.get("keep")
    mutate = obj.get("mutate")
    failure = obj.get("failure_guess")
    genericness = obj.get("genericness")
    if (
        not isinstance(keep, list)
        or not isinstance(mutate, list)
        or any(not isinstance(x, str) or not x.strip() for x in keep)
        or any(not isinstance(x, str) or not x.strip() for x in mutate)
        or len(keep) > 3
        or len(mutate) > 3
        or not isinstance(failure, str)
        or not failure.strip()
        or genericness not in {"low", "medium", "high"}
    ):
        return None
    return {
        "keep": [x.strip() for x in keep],
        "mutate": [x.strip() for x in mutate],
        "failure_guess": failure.strip(),
        "genericness": genericness,
    }


def request_feedback(
    *,
    port: int,
    member: dict,
    packet: dict,
    scenario: dict,
    worker: int,
    feedback_cfg: dict,
    campaign_seed: int,
) -> tuple[dict, str, float, int, str | None]:
    seed = (
        int(campaign_seed)
        + int(member["seed"])
        + int(scenario["scenario_id"][1:]) * 101
        + worker * 10007
    ) % 2147483647
    body = {
        "model": "local",
        "messages": [
            {"role": "system", "content": feedback_system(member, packet)},
            {"role": "user", "content": feedback_user(scenario)},
        ],
        "max_tokens": int(feedback_cfg["max_tokens"]),
        "temperature": float(feedback_cfg["temperature"]),
        "top_p": float(feedback_cfg["top_p"]),
        "seed": seed,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=240) as response:
        payload = json.loads(response.read())
    elapsed = time.monotonic() - started
    choice = payload["choices"][0]
    message = choice["message"]
    raw = (message.get("content") or "").strip()
    usage = payload.get("usage") or {}
    tokens = int(usage.get("completion_tokens") or 0)
    return body, raw, elapsed, tokens, choice.get("finish_reason")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments/village-001/preference-config.json",
    )
    parser.add_argument("--member", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pref = json.loads(args.config.read_text())
    campaign = json.loads((ROOT / pref["campaign_config"]).read_text())
    population, member = member_contract(campaign, args.member)
    packet = packet_for(member["lineage"])
    feedback_cfg = pref["feedback"]

    sft_result_path = (
        ROOT / pref["sft_evidence_root"] / "results" / f"{member['member_id']}.json"
    )
    if not sft_result_path.is_file():
        raise RuntimeError("VILLAGE-001C SFT result is missing")
    sft = json.loads(sft_result_path.read_text())
    if sft.get("experiment") != "VILLAGE-001C":
        raise RuntimeError("unexpected SFT evidence experiment")
    if sft.get("member_id") != member["member_id"]:
        raise RuntimeError("SFT evidence member mismatch")
    if sft.get("member_contract_sha256") != population["member_contract_sha256"]:
        raise RuntimeError("SFT evidence member contract mismatch")
    parent_adapter_sha = (sft.get("adapter") or {}).get("sha256")
    if not isinstance(parent_adapter_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", parent_adapter_sha
    ):
        raise RuntimeError("SFT result has no valid adapter directory SHA-256")

    scenarios = (sft.get("evaluation") or {}).get("responses")
    if not isinstance(scenarios, list) or len(scenarios) != int(
        feedback_cfg["scenario_count"]
    ):
        raise RuntimeError("SFT held-out scenarios are incomplete")
    expected_ids = [f"T{i:02d}" for i in range(1, 25)]
    if [row.get("scenario_id") for row in scenarios] != expected_ids:
        raise RuntimeError("SFT held-out scenario sequence mismatch")

    env = load_env(CACHE / "paths.env")
    server = Path(env["LLAMA_SERVER"])
    model = Path(env["SPROCKET_MODEL"])
    if not server.is_file() or not model.is_file():
        raise RuntimeError("pinned Sprocket runtime/model is unavailable")

    out = args.out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True)
    log_path = out / "sprocket-server.log"
    log_handle = log_path.open("w")

    port = 18950
    proc = subprocess.Popen(
        [
            str(server),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-t",
            str(feedback_cfg["server_threads"]),
            "-c",
            str(feedback_cfg["server_context"]),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    records: list[dict] = []
    startup_started = time.monotonic()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    log_path.read_text(errors="replace")[-6000:]
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError("Sprocket server did not become healthy")

        startup_seconds = time.monotonic() - startup_started
        for scenario in scenarios:
            for worker in range(1, int(feedback_cfg["workers_per_scenario"]) + 1):
                body, raw, elapsed, tokens, finish_reason = request_feedback(
                    port=port,
                    member=member,
                    packet=packet,
                    scenario=scenario,
                    worker=worker,
                    feedback_cfg=feedback_cfg,
                    campaign_seed=int(campaign["population"]["campaign_seed"]),
                )
                parsed = parse_feedback(raw)
                record = {
                    "feedback_id": (
                        f"FB-{member['member_id']}-"
                        f"{scenario['scenario_id']}-W{worker}"
                    ),
                    "member_id": member["member_id"],
                    "lineage": member["lineage"],
                    "sibling": member["sibling"],
                    "parent_adapter_sha256": parent_adapter_sha,
                    "scenario_id": scenario["scenario_id"],
                    "sprocket_worker": worker,
                    "request": body,
                    "raw_response": raw,
                    "valid": parsed is not None,
                    "parsed": parsed,
                    "elapsed_seconds": elapsed,
                    "completion_tokens": tokens,
                    "finish_reason": finish_reason,
                    "provenance": {
                        "sprocket_revision": campaign["sprocket"]["source_revision"],
                        "sprocket_sha256": campaign["sprocket"]["source_sha256"],
                        "member_contract_sha256": population[
                            "member_contract_sha256"
                        ],
                        "sft_result_path": str(
                            sft_result_path.relative_to(ROOT)
                        ),
                        "sft_scenario_response_sha256": __import__(
                            "hashlib"
                        ).sha256(
                            scenario["response"].encode()
                        ).hexdigest(),
                    },
                }
                validate_feedback_row(record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "feedback_id": record["feedback_id"],
                            "valid": record["valid"],
                            "completion_tokens": tokens,
                            "elapsed_seconds": elapsed,
                            "finish_reason": finish_reason,
                        }
                    )
                )
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(5)
        log_handle.close()

    expected_count = int(feedback_cfg["scenario_count"]) * int(
        feedback_cfg["workers_per_scenario"]
    )
    if len(records) != expected_count:
        raise RuntimeError(f"feedback count {len(records)} != {expected_count}")
    ids = [row["feedback_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate feedback IDs")
    per_scenario = Counter(row["scenario_id"] for row in records)
    if any(
        per_scenario[f"T{i:02d}"] != int(feedback_cfg["workers_per_scenario"])
        for i in range(1, 25)
    ):
        raise RuntimeError("feedback scenario coverage mismatch")

    write_jsonl(out / "feedback.jsonl", records)
    manifest = {
        "schema_version": 1,
        "experiment": "VILLAGE-001D",
        "stage": "sprocket-feedback",
        "issue": 13,
        "member_id": member["member_id"],
        "lineage": member["lineage"],
        "sibling": member["sibling"],
        "specialization": member["specialization"],
        "member_contract_sha256": population["member_contract_sha256"],
        "parent_adapter_sha256": parent_adapter_sha,
        "sprocket_revision": campaign["sprocket"]["source_revision"],
        "sprocket_sha256": campaign["sprocket"]["source_sha256"],
        "feedback_count": len(records),
        "valid_feedback_count": sum(1 for row in records if row["valid"]),
        "invalid_feedback_count": sum(1 for row in records if not row["valid"]),
        "valid_rate": (
            sum(1 for row in records if row["valid"]) / len(records)
            if records
            else 0
        ),
        "startup_seconds": startup_seconds,
        "runner": {
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
