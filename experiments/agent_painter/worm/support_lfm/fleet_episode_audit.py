#!/usr/bin/env python3
"""Fail-closed four-episode LFM WORM ORIGINAL-INFERENCE evidence gate.

SUPPORT-ONLY: never starts a model, training job, native host or queue.
An Actions lease, simulated controller, self-reported label or queued job is
NOT a resident LFM. Original model receipts remain producer-owned.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
W0_SHA = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
MODEL_REV = "fc6221ca597f3315e4f82fc2df606783267b34ba"
SCHEMA = "cybersand.worm.lfm-fleet-original-inference/1"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def valid_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_verified(root, path, digest):
    require(valid_hash(digest), "missing content digest")
    require(isinstance(path, str) and path and "\\" not in path, "bad evidence path")
    parts = PurePosixPath(path)
    require(not parts.is_absolute() and ".." not in parts.parts and "." not in parts.parts,
            "unsafe evidence path")
    file = root.joinpath(*parts.parts)
    require(file.is_file() and not file.is_symlink() and file.resolve().is_relative_to(root.resolve()),
            "missing or unsafe evidence file")
    content = file.read_bytes()
    require(hashlib.sha256(content).hexdigest() == digest, "evidence digest mismatch: " + path)
    return content


def jsonlines(data):
    require(data and data.endswith(b"\n"), "incomplete model/heartbeat JSONL")
    return [json.loads(x) for x in data.splitlines()]


def inspect(root, item, budget, worm):
    require(item.get("kind") == "trained_lfm", "synthetic/untrained/host is not trained LFM")
    require(item.get("status") in ("completed", "running"), "queued/failed is not active")
    for name in ("episode_id", "lease_id", "runner_id"):
        require(isinstance(item.get(name), str) and item[name], name + " missing")
    for name in ("workflow_run_id", "pid", "pid_start_ticks"):
        require(type(item.get(name)) is int and item[name] > 0, name + " missing")
    require(item["pid"] > 1, "invalid inference PID")
    require(type(item.get("seed")) is int and item["seed"] >= 0, "invalid world seed")
    require(item.get("family") in worm.FAMILIES, "unknown W0 family")
    require(item.get("model_id") == "LiquidAI/LFM2.5-VL-450M" and
            item.get("model_revision") == MODEL_REV, "LFM model pin mismatch")
    require(item.get("w0_source_sha") == W0_SHA, "W0 source pin mismatch")
    require(isinstance(item.get("source_sha"), str) and len(item["source_sha"]) == 40 and
            all(c in "0123456789abcdef" for c in item["source_sha"]), "Git source commit SHA missing")
    for name in ("checkpoint_sha256", "original_producer_zip_sha256"):
        require(valid_hash(item.get(name)), name + " missing")
    require(item.get("native_execution") is False and item.get("optimizer_updates") == 0,
            "this is inference-only and non-native")
    trace = json.loads(read_verified(root, item["trace"], item["trace_sha256"]))
    png = read_verified(root, item["png"], item["png_sha256"])
    raw_log = read_verified(root, item["inference"], item["inference_sha256"])
    inference = jsonlines(raw_log)
    pulses = jsonlines(read_verified(root, item["pulses"], item["pulses_sha256"]))
    require(trace.get("contract") == "cybersand.worm.observation-action/0-experimental" and
            trace.get("engine") == "python-stdlib-baseline-not-native" and
            trace.get("mode") in ("callback", "replay"), "not source-faithful W0")
    require(trace["seed"] == item["seed"] and trace["family"] == item["family"] and
            trace["size"] == 96 and trace["step_budget"] == budget, "different world/budget")
    events = trace["events"]
    require(1 <= len(events) <= budget and len(events) == trace["steps_executed"] and
            len(inference) == len(events), "original model actions are missing")
    raw = [e["raw_choice"] for e in events]
    for tick, (receipt, event) in enumerate(zip(inference, events)):
        require(receipt.get("tick") == tick and receipt.get("model_forward") is True and
                receipt.get("raw_choice") == event["raw_choice"] and
                receipt.get("producer_run_id") == item["workflow_run_id"] and
                valid_hash(receipt.get("model_receipt_sha256")) and
                receipt.get("inference_mode") in ("raw_first_pass", "six_logits_host_sample"),
                "missing or altered ORIGINAL LFM choice")
        latency = receipt.get("latency_ms")
        require(type(latency) in (float, int) and math.isfinite(latency) and latency > 0,
                "invalid inference latency")
        require(event["applied"] in event["observation"]["legal"] and
                event["raw_legal"] == (event["raw_choice"] in event["observation"]["legal"]),
                "altered host legality")
    replay = worm.run(seed=item["seed"], family=item["family"], mode="replay",
                      size=96, steps=len(events), raw_actions=raw)
    for name in ("events", "trajectory", "branch_points", "obstacles", "metrics"):
        require(trace[name] == replay[name], "W0 original state mismatch: " + name)
    require(worm.png_bytes(replay) == png, "W0 generated PNG differs from original")
    if item["status"] == "completed":
        require(len(events) == budget or trace["metrics"]["terminated_by_stop"],
                "incomplete original completion")
    require(len(pulses) == 2 and {p.get("sample_slot") for p in pulses} == {"a", "b"},
            "two live process samples required")
    by_slot = {p["sample_slot"]: p for p in pulses}
    for pulse in pulses:
        require(pulse.get("episode_id") == item["episode_id"] and
                pulse.get("runner_id") == item["runner_id"] and
                pulse.get("pid") == item["pid"] and
                pulse.get("pid_start_ticks") == item["pid_start_ticks"],
                "original inference process identity differs")
        require(pulse.get("attestation") == "linux-proc-pid-observed" and
                pulse.get("proc_alive") is True and pulse.get("state") == "running" and
                pulse.get("vcpus") == 4 and type(pulse.get("rss_kib")) is int and
                pulse["rss_kib"] > 0, "not actual four-vCPU resident process observation")
        n = pulse.get("observed_actions")
        require(type(n) is int and 0 < n <= len(inference) and
                type(pulse.get("observed_at_ms")) is int and pulse["observed_at_ms"] > 0,
                "missing observed action count/time")
        require(valid_hash(pulse.get("observed_log_prefix_sha256")), "unhashed process progress")
        prefix = b"\n".join(raw_log.splitlines()[:n]) + b"\n"
        require(hashlib.sha256(prefix).hexdigest() == pulse["observed_log_prefix_sha256"],
                "live log prefix not contained in original model evidence")
    require(by_slot["a"]["observed_at_ms"] < by_slot["b"]["observed_at_ms"] and
            by_slot["a"]["observed_actions"] < by_slot["b"]["observed_actions"],
            "no inference progress across two live samples")
    repetitions = 0
    for i in range(len(raw)):
        j = i
        while j < len(raw) and raw[j] == raw[i]:
            j += 1
        repetitions = max(repetitions, j - i)
    metrics = dict(trace["metrics"])
    metrics.update({
        "raw_choice_counts": dict(Counter(raw)),
        "longest_raw_action_streak": repetitions,
        "raw_invalid": sum(not e["raw_legal"] for e in events),
        "novelty_per_step": metrics["new_steps"] / len(events),
        "coverage_of_free_grid": metrics["unique_trail_cells"] / (96**2 - len(trace["obstacles"])),
        "recoveries_by_retrace_at_blocked_front": sum(
            e["applied"] == "RETRACE" and "STRAIGHT" not in e["observation"]["legal"]
            for e in events)})
    baselines = {m: worm.run(seed=item["seed"], family=item["family"],
                              mode=m, steps=budget, size=96)["metrics"]
                 for m in ("random", "wall")}
    return {
        "episode_id": item["episode_id"], "seed": item["seed"], "family": item["family"],
        "status": item["status"], "runner": item["runner_id"],
        "source_sha": item["source_sha"], "checkpoint_sha256": item["checkpoint_sha256"],
        "decisions": len(events), "metrics": metrics, "baselines": baselines,
        "original_png_sha256": item["png_sha256"], "pulses": by_slot,
        "median_inference_latency_ms": statistics.median(r["latency_ms"] for r in inference),
        "peak_rss_kib": max(p["rss_kib"] for p in pulses)}


def audit(root, manifest, worm):
    require(manifest.get("schema") == SCHEMA, "unknown fleet evidence schema")
    require(isinstance(manifest.get("campaign_id"), str) and manifest["campaign_id"],
            "campaign missing")
    budget = manifest.get("decision_budget")
    require(type(budget) is int and 1 <= budget <= 512, "decision bound invalid")
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list) and len(episodes) >= 4, "need >=4 episodes")
    for unique, label in (
        (lambda e: e.get("episode_id"), "episode"),
        (lambda e: e.get("lease_id"), "lease"),
        (lambda e: e.get("seed"), "obstacle world"),
        (lambda e: (e.get("runner_id"), e.get("pid"), e.get("pid_start_ticks")), "process"),
        (lambda e: e.get("trace"), "original W0 trace"),
        (lambda e: e.get("png"), "original W0 PNG")):
        require(len({unique(e) for e in episodes}) == len(episodes), "duplicate " + label)
    reports = [inspect(root, ep, budget, worm) for ep in episodes]
    # Four real episode processes, two synchronized actions-progress witnesses;
    # being in the Actions QUEUE is not a running resident model.
    valid_slots = [slot for slot in ("a", "b") if
                   max(r["pulses"][slot]["observed_at_ms"] for r in reports) -
                   min(r["pulses"][slot]["observed_at_ms"] for r in reports) <= 10_000]
    occupancy = Counter(r["runner"] for r in reports)
    return {"schema": "cybersand.worm.lfm-fleet-audit/1",
            "campaign_id": manifest["campaign_id"], "decision_budget": budget,
            "four_concurrent_progress_windows": len(valid_slots) == 2,
            "preflight_pass": len(valid_slots) == 2 and len(reports) >= 4,
            "windows": valid_slots, "resident_processes_by_runner": dict(occupancy),
            "model_weights_independently_executed": False,
            "native_godot": False, "optimizer_updates": 0,
            "episodes": reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    import worm
    manifest = json.loads(args.manifest.read_text())
    result = audit(args.manifest.parent, manifest, worm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, sort_keys=True))
    if not result["preflight_pass"]:
        raise SystemExit("VALID W0 ART, but no proof of four simultaneously active LFM episodes")


if __name__ == "__main__":
    main()
