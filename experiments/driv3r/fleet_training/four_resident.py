"""FT-002: one genuine four-resident, four-CPU synthetic game-policy training wrangler.

Extends completed FT-001 capacity measurement with 1->2->4 concurrent worker
epochs on ONE hosted Linux runner: common monotonic clock and /proc CPU pulses.
Uses a separately pinned read-only upstream Population trainer and four unique
curriculum seeds. These are real trained 24-unit MLPs, not pretrained SLM,
source-faithful W0 or native Godot.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import fleet_bench as base

MAX_PHASE_S = 300
POLL_S = 0.05
RSS_LIMIT_KIB = 900000


def validated_request(r):
    if (r.get("schema") != "cybersand.fleet-training.request/2"
        or r.get("request_id") != "FT-002-20261116-4-resident-CPU-population"
        or r.get("upstream_commit") != "534cd8ffac6fbe33a714ae03a62c16045bf10259"
        or r.get("upstream_source_blob") != "8a974a072dd8a5172f9242d83a373f3de345bb3b"):
        raise ValueError("Refuse unpinned source/request")
    seeds = r["seeds"]
    if (len(seeds) != 4 or len(set(seeds)) != 4 or
        any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds)
        or (r["population"], r["train_episodes"], r["selection_episodes"],
            r["test_episodes"]) != (2, 12, 4, 6)):
        raise ValueError("Refuse duplicate/bad four-way lease or work budget")
    return r


def cpu_ticks(pid):
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return int(tail[11]) + int(tail[12])
    except (FileNotFoundError, ValueError, IndexError):
        return None


def read_json(path):
    return json.loads(path.read_text())


def run_group(mode, seeds, upstream_source, root, req):
    root.mkdir(parents=True, exist_ok=False)
    clock = time.perf_counter()
    group_size = {"serial": 1, "dual": 2, "quad": 4}[mode]
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    jobs = []
    pulse_segments = []
    first_all_alive = None
    four_rising_ticks = False
    for start in range(0, len(seeds), group_size):
        group = []
        descriptors = []
        last_ticks = None
        four_live_samples = 0
        try:
            for seed in seeds[start:start + group_size]:
                path = root / str(seed)
                logfile = root / f"{seed}.stdout.log"
                errfile = root / f"{seed}.stderr.log"
                stdout, stderr = logfile.open("w"), errfile.open("w")
                descriptors.append((stdout, stderr))
                p = subprocess.Popen(
                    [sys.executable, str(upstream_source), "--output", str(path),
                     "--seed", str(seed), "--population", str(req["population"]),
                     "--train-episodes", str(req["train_episodes"]),
                     "--selection-episodes", str(req["selection_episodes"]),
                     "--test-episodes", str(req["test_episodes"])],
                    stdout=stdout, stderr=stderr, env=env, start_new_session=True)
                j = {"seed": seed, "lease_id": f"{req['request_id']}/{mode}/{seed}",
                     "pid": p.pid, "process": p, "start_monotonic": time.perf_counter(),
                     "peak_rss_kib": 0, "finish_monotonic": None, "returncode": None}
                group.append(j)
            while any(j["returncode"] is None for j in group):
                if time.perf_counter() - clock > MAX_PHASE_S:
                    raise TimeoutError("FT-002 phase deadline exceeded")
                rss_sum = 0
                all_live = True
                ticks = {}
                for j in group:
                    if j["returncode"] is not None:
                        all_live = False
                        continue
                    p = j["process"]
                    rc = p.poll()
                    if rc is None:
                        mem = base.rss_kib(j["pid"])
                        j["peak_rss_kib"] = max(j["peak_rss_kib"], mem)
                        rss_sum += mem
                        ticks[j["pid"]] = cpu_ticks(j["pid"])
                    else:
                        j["finish_monotonic"] = time.perf_counter()
                        j["returncode"] = rc
                        all_live = False
                if rss_sum > RSS_LIMIT_KIB:
                    raise MemoryError("Four-way runner RSS cap exceeded")
                if all_live and len(group) == 4 and all(x is not None for x in ticks.values()):
                    four_live_samples += 1
                    if first_all_alive is None:
                        first_all_alive = time.perf_counter()
                    if last_ticks and all(ticks[k] > last_ticks.get(k, -1) for k in ticks):
                        four_rising_ticks = True
                    last_ticks = ticks
                time.sleep(POLL_S)
            for j in group:
                if j["returncode"] != 0:
                    raise RuntimeError(f"Original worker seed {j['seed']} exited {j['returncode']}")
                report = base.verified_result(root / str(j["seed"]), j["seed"], req)
                j["verified_training"] = report
                j["elapsed_s"] = round(j["finish_monotonic"] - j["start_monotonic"], 5)
                jobs.append({k: v for k, v in j.items() if k != "process"})
        except BaseException:
            for j in group:
                if j["process"].poll() is None:
                    os.killpg(j["pid"], signal.SIGKILL)
                    j["process"].wait()
            raise
        finally:
            for a, b in descriptors:
                a.close()
                b.close()
        if len(group) == 4:
            pulse_segments.append({"four_live_samples": four_live_samples,
                                   "interval_s": POLL_S,
                                   "all_four_cpu_counters_rising_between_common_samples": four_rising_ticks})
    return {"mode": mode, "wall_s": round(time.perf_counter() - clock, 5),
            "workers": jobs, "four_way_same_clock_samples": pulse_segments}


def run(req_path, upstream, out):
    req = validated_request(read_json(req_path))
    source = base.source_check(upstream, req)
    out.mkdir(parents=True, exist_ok=False)
    state = {"schema": "cybersand.fleet-training.four-resident/1",
             "request_id": req["request_id"], "source_commit": req["upstream_commit"],
             "source_blob": req["upstream_source_blob"],
             "source_sha256": base.sha_file(source),
             "single_runner": True, "pretrained_lm": False, "optimizer": "NumPy Adam",
             "game": "SYNTHETIC 22x22 goal/material/protection/retrace",
             "native": False, "source_faithful_W0": False,
             "seeds": req["seeds"], "epochs": [], "status": "starting"}
    try:
        for mode in ("serial", "dual", "quad"):
            state["status"] = mode
            state["epochs"].append(run_group(mode, req["seeds"], source, out / mode, req))
        serial, dual, quad = state["epochs"]
        if not quad["four_way_same_clock_samples"] or not quad["four_way_same_clock_samples"][0]["all_four_cpu_counters_rising_between_common_samples"]:
            raise AssertionError("No witnessed common-time four-live-and-progressing CPU worker interval")
        if len({j["pid"] for j in quad["workers"]}) != 4:
            raise AssertionError("Not four independent model processes")
        for seed in req["seeds"]:
            comparisons = []
            for epoch in (serial, dual, quad):
                entry = next(j for j in epoch["workers"] if j["seed"] == seed)
                comparisons.append(entry["verified_training"]["files_sha256"])
            if not comparisons[0] == comparisons[1] == comparisons[2]:
                raise AssertionError(f"Optimizer checkpoint/gallery parity differs for seed {seed}")
        state["status"] = "completed"
        state["one_to_two_speedup"] = round(serial["wall_s"] / dual["wall_s"], 5)
        state["one_to_four_speedup"] = round(serial["wall_s"] / quad["wall_s"], 5)
        state["two_to_four_speedup"] = round(dual["wall_s"] / quad["wall_s"], 5)
        state["observed_four_distinct_same_host_resident_processes"] = True
        state["four_live_cpu_forward_progress_witness"] = True
        state["all_12_candidate_outcomes_source_replay_same_seeds"] = True
        state["quad_peak_sampled_sum_rss_kib"] = sum(j["peak_rss_kib"] for j in quad["workers"])
        state["native_material_edits"] = 0
    except BaseException as exc:
        state["status"] = "failed"
        state["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
        digests = []
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name != "SHA256SUMS":
                digests.append(base.sha_file(f) + "  " + str(f.relative_to(out)))
        (out / "SHA256SUMS").write_text("\n".join(digests) + "\n")
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    result = run(a.request, a.upstream, a.output)
    print(json.dumps({k: result[k] for k in
        ("status", "seeds", "one_to_two_speedup", "one_to_four_speedup",
         "two_to_four_speedup", "quad_peak_sampled_sum_rss_kib",
         "observed_four_distinct_same_host_resident_processes",
         "four_live_cpu_forward_progress_witness")}, indent=2))
