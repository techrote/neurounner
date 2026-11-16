#!/usr/bin/env python3
"""WORM LFM support: record one real Linux model-worker /proc + log pulse.

Only use on the genuine inference worker PID while its append-only JSONL
model-forward stream advances. Never attribute this witness to a queued
Actions job, a simulated W0 controller, an idle model, or an optimizer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def capture(pid, log_path, episode_id, runner_id, slot):
    if pid <= 1 or pid == os.getpid():
        raise ValueError("must inspect a separate running inference model PID")
    root = Path("/proc") / str(pid)
    stat = (root / "stat").read_text()
    # Process comm may include spaces; fields after final ")" start at #3.
    fields = stat[stat.rfind(")") + 2:].split()
    if fields[0] in ("Z", "D", "X"):
        raise RuntimeError("dead/uninterruptible process is not running inference")
    start = int(fields[19])  # original Linux stat field #22
    status = (root / "status").read_text()
    rss = next((int(line.split()[1]) for line in status.splitlines()
                if line.startswith("VmRSS:")), 0)
    vcpus = len(os.sched_getaffinity(pid))
    if rss <= 0 or vcpus != 4:
        raise RuntimeError("worker is not resident on four allowed vCPUs")
    original = log_path.read_bytes()
    if not original or not original.endswith(b"\n"):
        raise RuntimeError("model log is absent or partially written")
    rows = [json.loads(x) for x in original.splitlines()]
    if any(row.get("model_forward") is not True for row in rows):
        raise RuntimeError("host-only/stub/queued rows do not prove model inference")
    return {
        "attestation": "linux-proc-pid-observed",
        "episode_id": episode_id, "runner_id": runner_id,
        "pid": pid, "pid_start_ticks": start, "sample_slot": slot,
        "state": "running", "proc_alive": True,
        "rss_kib": rss, "vcpus": vcpus,
        "observed_at_ms": time.time_ns() // 1000000,
        "observed_actions": len(rows),
        "observed_log_prefix_sha256": hashlib.sha256(original).hexdigest()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", required=True, type=int)
    p.add_argument("--inference-jsonl", required=True, type=Path)
    p.add_argument("--episode-id", required=True)
    p.add_argument("--runner-id", default=os.environ.get("RUNNER_NAME"))
    p.add_argument("--slot", choices=("a", "b"), required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if not a.runner_id:
        p.error("RUNNER_NAME or --runner-id required")
    row = capture(a.pid, a.inference_jsonl, a.episode_id, a.runner_id, a.slot)
    if a.output.exists() and a.output.read_bytes().strip():
        previous = [json.loads(x) for x in a.output.read_text().splitlines()]
        if any(x["sample_slot"] == a.slot for x in previous):
            p.error("pulse slot already exists; refusing replacement")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({k: row[k] for k in (
        "episode_id", "sample_slot", "runner_id", "observed_actions", "rss_kib")}, sort_keys=True))


if __name__ == "__main__":
    main()
