"""FT-003: matched training-dose generalization experiment on real tiny game-policy models.

Four actual optimizer episodes: 2 seeds x (12,48) training-world budgets.
The same independent selection/heldout worlds compare curriculum exposure.
All models are source-fenced 24-hidden NumPy policies in an external synthetic
22x22 host, not pretrained SLMs, native PAINT, or W0. Negative results persist.
"""
from __future__ import annotations

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
from four_resident import cpu_ticks

SCHEMA = "cybersand.fleet-training.dose-request/1"
REQUEST_ID = "FT-003-20261116-population-dose-matched"
COMMIT = "534cd8ffac6fbe33a714ae03a62c16045bf10259"
BLOB = "8a974a072dd8a5172f9242d83a373f3de345bb3b"
NICHES = base.NICHES
POLL = 0.05
PHASE_CAP_S = 240
RSS_CAP_KIB = 1_800_000
WORKER_RSS_CAP_KIB = 900_000


def validated_request(r):
    if not (isinstance(r, dict) and r.get("schema") == SCHEMA and
            r.get("request_id") == REQUEST_ID and
            r.get("upstream_commit") == COMMIT and r.get("upstream_source_blob") == BLOB):
        raise ValueError("FT-003 source identity mismatch")
    if (r.get("seeds") != [9401, 9403] or r.get("train_episodes") != [12, 48]
        or (r.get("population"), r.get("selection_episodes"), r.get("test_episodes")) != (2, 4, 12)
        or r.get("max_parallel") != 2 or
        r.get("resources") != {"runner_count": 1, "phase_timeout_s": PHASE_CAP_S,
                               "rss_cap_kib": RSS_CAP_KIB, "worker_rss_cap_kib": WORKER_RSS_CAP_KIB}):
        raise ValueError("FT-003 workload or resources differ from bounded approval")
    return r


def expected_seed_hash(seed, split, n):
    values = []
    for i in range(n):
        # Upstream names the held-out seed domain "test" while reporting it as "heldout".
        domain = "test" if split == "heldout" else split
        msg = "|".join(map(str, (seed, domain, i))).encode()
        values.append(int.from_bytes(hashlib.sha256(msg).digest()[:8], "big") % (2**32))
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def validate_split(result, seed, dose, r):
    assert result["seed"] == seed and result["train_seed_count"] == dose
    assert result["selection_seed_count"] == r["selection_episodes"]
    assert result["heldout_seed_count"] == r["test_episodes"]
    for split, count in (("train", dose), ("selection", r["selection_episodes"]),
                         ("heldout", r["test_episodes"])):
        if result["seed_hashes"][split] != expected_seed_hash(seed, split, count):
            raise AssertionError("Trainer world split SHA differs from exact expected seed list")
    return result


def compare_pair(seed, short, long):
    if short["seed"] != seed or long["seed"] != seed:
        raise AssertionError("Unmatched world seeds")
    if (short["selection_seed_count"] != long["selection_seed_count"] or
        short["heldout_seed_count"] != long["heldout_seed_count"] or
        short["seed_hashes"]["selection"] != long["seed_hashes"]["selection"] or
        short["seed_hashes"]["heldout"] != long["seed_hashes"]["heldout"]):
        raise AssertionError("Invalid matched held-out and selection custody")
    if short["heldout_controls"] != long["heldout_controls"]:
        raise AssertionError("Controls differ on allegedly identical held-out worlds")
    rows = []
    for niche in NICHES:
        s = short["heldout_champions"][niche]
        l = long["heldout_champions"][niche]
        ctl = short["heldout_controls"][niche]["nonstop_random"]
        rows.append({
            "niche": niche,
            "reward_12": s["mean_reward"], "reward_48": l["mean_reward"],
            "reward_delta": round(l["mean_reward"] - s["mean_reward"], 5),
            "mean_new_ink_12": s["mean_painted"], "mean_new_ink_48": l["mean_painted"],
            "mean_retrace_12": s["mean_retrace"], "mean_retrace_48": l["mean_retrace"],
            "early_stop_12": s["early_stop_rate"], "early_stop_48": l["early_stop_rate"],
            "nonstop_random_reward": ctl["mean_reward"],
            "delta_12_vs_control": round(s["mean_reward"] - ctl["mean_reward"], 5),
            "delta_48_vs_control": round(l["mean_reward"] - ctl["mean_reward"], 5),
        })
    return {"seed": seed, "heldout_seed_sha256": short["seed_hashes"]["heldout"],
            "selection_seed_sha256": short["seed_hashes"]["selection"], "niches": rows,
            "improved": sum(x["reward_delta"] > 0 for x in rows),
            "regressed": sum(x["reward_delta"] < 0 for x in rows),
            "ties": sum(x["reward_delta"] == 0 for x in rows)}


def run_phase(dose, r, source, out):
    phase = out / ("dose_" + str(dose))
    phase.mkdir()
    started = time.monotonic()
    jobs = []
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    progress = {"two_live_samples": 0, "both_cpu_counters_increased": False,
                "peak_sampled_combined_rss_kib": 0}
    previous = None
    try:
        for seed in r["seeds"]:
            target = phase / str(seed)
            stdout = (phase / (str(seed) + ".stdout.log")).open("w")
            stderr = (phase / (str(seed) + ".stderr.log")).open("w")
            try:
                p = subprocess.Popen(
                    [sys.executable, str(source), "--output", str(target),
                     "--seed", str(seed), "--population", str(r["population"]),
                     "--train-episodes", str(dose),
                     "--selection-episodes", str(r["selection_episodes"]),
                     "--test-episodes", str(r["test_episodes"])],
                    stdout=stdout, stderr=stderr, env=env, start_new_session=True)
            finally:
                stdout.close()
                stderr.close()
            jobs.append({"seed": seed, "dose": dose,
                         "lease_id": REQUEST_ID + "/" + str(dose) + "/" + str(seed),
                         "pid": p.pid, "process": p, "started": time.monotonic(),
                         "returncode": None, "completed": None, "peak_rss_kib": 0})
        while any(j["returncode"] is None for j in jobs):
            if time.monotonic() - started > PHASE_CAP_S:
                raise TimeoutError("FT-003 phase timeout")
            live, rss_sum, ticks = 0, 0, {}
            for j in jobs:
                if j["returncode"] is not None:
                    continue
                code = j["process"].poll()
                if code is None:
                    live += 1
                    rss = base.rss_kib(j["pid"])
                    j["peak_rss_kib"] = max(rss, j["peak_rss_kib"])
                    rss_sum += rss
                    ticks[j["pid"]] = cpu_ticks(j["pid"])
                    if rss > WORKER_RSS_CAP_KIB:
                        raise MemoryError("FT-003 single model RSS cap")
                else:
                    j["returncode"] = code
                    j["completed"] = time.monotonic()
            progress["peak_sampled_combined_rss_kib"] = max(
                progress["peak_sampled_combined_rss_kib"], rss_sum)
            if rss_sum > RSS_CAP_KIB:
                raise MemoryError("FT-003 combined model RSS cap")
            if live == 2 and all(v is not None for v in ticks.values()):
                progress["two_live_samples"] += 1
                if previous and all(ticks[k] > previous.get(k, -1) for k in ticks):
                    progress["both_cpu_counters_increased"] = True
                previous = ticks
            time.sleep(POLL)
        if any(j["returncode"] != 0 for j in jobs):
            raise RuntimeError("FT-003 trainer returned nonzero; original logs preserved")
        results = {}
        for j in jobs:
            folder = phase / str(j["seed"])
            checked = base.verified_result(folder, j["seed"],
                                           {"train_episodes": dose,
                                            "selection_episodes": r["selection_episodes"],
                                            "test_episodes": r["test_episodes"]})
            full = validate_split(json.loads((folder / "results.json").read_text()),
                                  j["seed"], dose, r)
            results[j["seed"]] = full
            j["verified_optimizer_updates"] = checked["optimizer_updates"]
            j["model_sha256"] = checked["model_sha256"]
            j["elapsed_s"] = round(j["completed"] - j["started"], 5)
        if not progress["both_cpu_counters_increased"]:
            raise AssertionError("Two progressing concurrent optimizer PIDs not observed")
        return {"dose": dose, "status": "completed",
                "wall_s": round(time.monotonic() - started, 5),
                "liveness": progress,
                "workers": [{k:v for k,v in j.items() if k != "process"} for j in jobs]}, results
    finally:
        for j in jobs:
            if j["process"].poll() is None:
                os.killpg(j["pid"], signal.SIGKILL)
                j["process"].wait()


def run(request, upstream, out):
    r = validated_request(json.loads(request.read_text()))
    source = base.source_check(upstream, r)
    out.mkdir(parents=True, exist_ok=False)
    status = {"schema": "cybersand.fleet-training.dose-results/1",
              "request_id": REQUEST_ID, "source_commit": COMMIT, "source_blob": BLOB,
              "source_sha256": base.sha_file(source), "request_sha256": base.sha_file(request),
              "runner_source_sha256": base.sha_file(Path(__file__)),
              "class": "REAL_NUMPY_ADAM_SYNTHETIC_GAME_POLICY",
              "not": ["native Godot", "source-faithful W0", "pretrained SLM", "game production"],
              "model_type": "24-hidden-unit NumPy policy", "optimizer": "Adam",
              "status": "starting", "phases": [], "compared_seeds": []}
    try:
        all_results = {}
        for dose in r["train_episodes"]:
            status["status"] = "running"
            phase, results = run_phase(dose, r, source, out)
            status["phases"].append(phase)
            all_results[dose] = results
        for seed in r["seeds"]:
            status["compared_seeds"].append(
                compare_pair(seed, all_results[12][seed], all_results[48][seed]))
        rows = [v for seed in status["compared_seeds"] for v in seed["niches"]]
        status["outcome"] = {
            "niche_seed_pairs": len(rows),
            "improved": sum(x["reward_delta"] > 0 for x in rows),
            "regressed": sum(x["reward_delta"] < 0 for x in rows),
            "tied": sum(x["reward_delta"] == 0 for x in rows),
            "mean_heldout_reward_change": round(sum(x["reward_delta"] for x in rows)/len(rows), 5),
            "mean_new_ink_change": round(sum(x["mean_new_ink_48"]-x["mean_new_ink_12"]
                                             for x in rows)/len(rows), 5),
            "heldout_evaluation_only": True,
            "native_material_edits": 0}
        status["status"] = "completed"
    except BaseException as exc:
        status["status"] = "failed"
        status["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (out / "report.json").write_text(json.dumps(status, indent=2, sort_keys=True, default=str)+"\n")
        files = [p for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
        (out / "SHA256SUMS").write_text("".join(base.sha_file(p) + "  " + str(p.relative_to(out))+"\n"
                                                for p in files))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    result = run(a.request, a.upstream, a.output)
    print(json.dumps({"status": result["status"], "outcome": result["outcome"],
                      "leases": [x["lease_id"] for p in result["phases"] for x in p["workers"]]},
                     indent=2))
