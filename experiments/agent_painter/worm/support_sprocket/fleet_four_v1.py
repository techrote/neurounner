#!/usr/bin/env python3
"""Four independent *synthetic W0* processes and a model-fleet handshake.

NO model loaded, trained or resident. NO native PAINT. This is the support-side
four-stream IPC/custody rehearsal only; it cannot qualify 2 models per runner.
W0 remains the sole state, legality, seeded RNG, material PNG and Stop owner.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
W0_PATH = HERE.parent / "common" / "worm.py"
if not W0_PATH.is_file():
    raise RuntimeError("shepherd-owned common/worm.py is required")
sys.path.insert(0, str(W0_PATH.parent))
import worm  # noqa: E402; source is imported and NEVER copied/changed

VERSION = "sprocket-support-four-stream-w0/1"
# Distinct worlds/streams, not four correlated theme views of one world.
# Reused DEVELOPMENT worlds are evaluation demonstrations, NOT held-out claims.
SPECS = (
    ("w0-01", "greebles", "fidget", 313),
    ("w0-02", "pipework", "random", 337),
    ("w0-03", "architecture", "wall", 347),
    ("w0-04", "pipework", "fidget", 359),
)
STEPS = 96
SIZE = 96
MAX_WAIT_SECONDS = 50


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_identity():
    return {"path": "experiments/agent_painter/worm/common/worm.py",
            "sha256": digest(W0_PATH.read_bytes()),
            "contract": "cybersand.worm.observation-action/0-experimental"}


def signatures(record):
    events = record["events"]
    path = record["trajectory"]
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    counts = Counter(e["applied"] for e in events)
    # A true dead-end is a state with none of the three forward turn actions.
    dead = [e["observation"]["tick"] for e in events
            if not set(e["observation"]["legal"]).intersection(("STRAIGHT", "LEFT", "RIGHT"))]
    wall = [e["observation"]["tick"] for e in events
            if "#" in "".join(e["observation"]["local_5x5"])]
    streak, longest, previous = 0, 0, None
    for e in events:
        action = e["applied"]
        streak = streak + 1 if action == previous else 1
        longest = max(longest, streak)
        previous = action
    return {
        "action_counts": {a: counts[a] for a in worm.ACTIONS},
        "wall_nearby_ticks": len(wall), "first_wall_nearby_tick": wall[0] if wall else None,
        "actual_dead_end_ticks": dead,
        "branch_timestamps": [e["observation"]["tick"] for e in events if e["applied"] == "BRANCH"],
        "retrace_timestamps": [e["observation"]["tick"] for e in events if e["applied"] == "RETRACE"],
        "max_action_streak": longest,
        "bounding_box": [min(xs), min(ys), max(xs), max(ys)],
        "bounding_box_wh": [max(xs)-min(xs)+1, max(ys)-min(ys)+1],
        "maximum_manhattan_reach": max(abs(x-xs[0])+abs(y-ys[0]) for x, y in path),
        "unique_cells": len(set(map(tuple, path))),
        "revisit_contacts": record["metrics"]["total_revisit_contacts"],
    }


def verify(spec, record, png):
    sid, family, mode, seed = spec
    if (record["family"], record["mode"], record["seed"], record["size"],
        record["step_budget"]) != (family, mode, seed, SIZE, STEPS):
        raise ValueError("wrong source world/controller identity: " + sid)
    if len(record["events"]) != record["steps_executed"] or len(record["trajectory"]) != len(record["events"])+1:
        raise ValueError("event/path count inconsistency: " + sid)
    for event in record["events"]:
        obs = event["observation"]
        if not set(obs["legal"]).issubset(worm.ACTIONS) or len(set(obs["legal"])) != len(obs["legal"]):
            raise ValueError("illegal or duplicated host action list")
        if event["raw_legal"] != (event["raw_choice"] in obs["legal"]):
            raise ValueError("raw legality receipt mismatch")
        if event["applied"] != (event["raw_choice"] if event["raw_legal"]
                                  else ("RETRACE" if "RETRACE" in obs["legal"] else "STOP")):
            raise ValueError("host fallback mismatch")
        if not event["applied"] in obs["legal"]:
            raise ValueError("host applied illegal action")
    if record["steps_executed"] > STEPS or record["metrics"]["invalid_raw_choices"] != 0:
        raise ValueError("invalid action/budget")
    # Independently rerun the EXACT shared W0 engine and compare original bytes.
    replay = worm.run(seed, mode, family, STEPS, SIZE)
    if canonical(record) != canonical(replay) or png != worm.png_bytes(replay):
        raise ValueError("non-replayable full state or PNG: " + sid)
    return signatures(record)


def _worker(spec, conn):
    sid, family, mode, seed = spec
    try:
        pid = os.getpid()
        conn.send({"kind": "loaded_w0", "id": sid, "pid": pid,
                   "source_sha256": source_identity()["sha256"],
                   "note": "loaded local W0 Python host; NO model loaded"})
        command = conn.recv()
        if command != "START_W0":
            raise ValueError("no bounded START_W0 lease")
        entered = time.monotonic_ns()
        conn.send({"kind": "entered_host_run", "id": sid, "pid": pid, "when_ns": entered})
        cpu0 = time.process_time_ns()
        record = worm.run(seed, mode, family, STEPS, SIZE)
        png = worm.png_bytes(record)
        cpu_ns = time.process_time_ns() - cpu0
        exited = time.monotonic_ns()
        conn.send({"kind": "completed_w0", "id": sid, "pid": pid, "when_ns": exited,
                   "cpu_ns": cpu_ns, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "record": record, "png": png})
    except BaseException as exc:
        conn.send({"kind": "failed", "id": sid, "pid": os.getpid(),
                   "error": repr(exc)})
    finally:
        conn.close()


def run_four(out):
    if len(SPECS) != 4 or len({s[0] for s in SPECS}) != 4 or len({s[3] for s in SPECS}) != 4:
        raise ValueError("not four independently seeded identities")
    if not (0 < STEPS <= 96 and SIZE == 96):
        raise ValueError("pilot exceeds support budget")
    if out.exists() and any(out.iterdir()):
        raise ValueError("refuse overwriting prior evidence")
    out.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    children = []
    # Each worker owns one OS PID, one W0 world and one host RNG seed.
    for spec in SPECS:
        parent, child = context.Pipe()
        p = context.Process(target=_worker, args=(spec, child), name=spec[0])
        p.start()
        child.close()
        children.append((spec, p, parent))
    runtime = {s[0]: {"id": s[0], "family": s[1], "policy": s[2], "world_seed": s[3],
                      "transitions": ["spawned"]} for s, _, _ in children}
    completed = {}
    started = {}
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    try:
        def collect(wanted):
            pending = {conn: (spec, proc) for spec, proc, conn in children
                       if (spec[0] not in (completed if wanted == "completed_w0" else started)
                           if wanted != "loaded_w0" else "loaded_w0" not in runtime[spec[0]]["transitions"])}
            # 'loaded_w0' and 'entered_host_run' phases handled by separate stored flags.
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("four-stream finite deadline")
                ready = wait(list(pending), timeout=min(remaining, 2.0))
                if not ready and any(not p.is_alive() for s, p in pending.values()):
                    raise RuntimeError("child exited before handshake")
                for conn in ready:
                    spec, proc = pending[conn]
                    msg = conn.recv()
                    if msg.get("kind") == "failed":
                        raise RuntimeError("worker failed " + repr(msg))
                    if msg.get("kind") != wanted or msg.get("id") != spec[0] or msg.get("pid") != proc.pid:
                        raise RuntimeError("out-of-order/misattributed IPC phase: " + repr(msg)[:180])
                    status = runtime[spec[0]]
                    status["transitions"].append(wanted)
                    if wanted == "loaded_w0":
                        if msg["source_sha256"] != source_identity()["sha256"]:
                            raise ValueError("source hash differs across processes")
                        status["pid"] = msg["pid"]
                    elif wanted == "entered_host_run":
                        started[spec[0]] = msg["when_ns"]
                        status["started_ns"] = msg["when_ns"]
                    else:
                        completed[spec[0]] = (msg["record"], msg["png"])
                        status.update({"ended_ns": msg["when_ns"],
                                       "cpu_ns": msg["cpu_ns"],
                                       "max_rss_kib": msg["max_rss_kib"]})
                    del pending[conn]

        collect("loaded_w0")
        if len({runtime[s[0]]["pid"] for s in SPECS}) != 4 or not all(p.is_alive() for _, p, _ in children):
            raise RuntimeError("not four independently LIVE W0 processes at barrier")
        for _, _, conn in children:
            conn.send("START_W0")
        collect("entered_host_run")
        collect("completed_w0")
        # True W0 execution windows must intersect. This is *not* model residency.
        overlap = min(runtime[s[0]]["ended_ns"] for s in SPECS) - max(started.values())
        if overlap <= 0:
            raise RuntimeError("no observed four-process simultaneous W0 execution")
        identities = []
        for spec in SPECS:
            sid, family, mode, seed = spec
            record, png = completed[sid]
            sig = verify(spec, record, png)
            raw = canonical(record)
            (out / (sid + ".json")).write_bytes(raw)
            (out / (sid + ".png")).write_bytes(png)
            identities.append({
                "stream_id": sid, "wrangler_id": "sprocket-support-w0", "lease_id": "w0-fleet-pilot-" + sid,
                "world_seed": seed, "family": family, "policy": mode,
                "model": None, "checkpoint": None, "adapter": None,
                "host_kind": "synthetic-python-W0-not-native-not-inference",
                "source_sha256": source_identity()["sha256"], "turn_budget": STEPS,
                "steps_executed": record["steps_executed"], "stop": record["metrics"]["terminated_by_stop"],
                "json_sha256": digest(raw), "png_sha256": digest(png),
                "signature": sig,
            })
        if len({a["png_sha256"] for a in identities}) != 4:
            raise ValueError("four streams did not give independent images")
        manifest = {
            "schema": VERSION, "source": source_identity(), "streams": identities,
            "process_verdict": "four real concurrent W0 OS processes; ZERO resident models",
            "native": False, "trained": False, "valid_model_processes": 0,
            "development_split_claim": False,
            "benchmark_gate": "separate primary/Commander 1-vs-2 models-per-4-vCPU runner required",
            "next_budget_gate": "do not scale to 128 native turns without native leases, Stop, horizon and memory qualification",
            "optimizer_updates": 0,
        }
        (out / "manifest.json").write_bytes(canonical(manifest))
        (out / "runtime.json").write_bytes(canonical({
            "schema": VERSION + "-runtime", "streams": runtime,
            "overlap_ns": overlap, "all_four_loaded_w0_at_barrier": True,
            "all_four_active_host_windows_intersect": True,
            "sum_process_peak_rss_kib_NOT_simultaneous_peak": sum(
                runtime[s[0]]["max_rss_kib"] for s in SPECS),
            "qualification": "W0 process/IPC only; model residency, 2x2 and native untested"}))
        sums = sorted((digest(p.read_bytes()), p.name) for p in out.iterdir() if p.is_file())
        (out / "SHA256SUMS.txt").write_text("".join(h + "  " + n + "\n" for h, n in sums))
        return manifest
    finally:
        for _, proc, conn in children:
            conn.close()
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=3)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = run_four(args.output)
    print(json.dumps({"status": "PASS", "streams": len(manifest["streams"]),
                      "model_processes": 0, "source": manifest["source"],
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
