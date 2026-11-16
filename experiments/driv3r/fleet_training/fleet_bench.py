"""Bounded REAL CPU optimizer residency benchmark for synthetic CyberSand-like policies.

Executes immutable upstream Population Game Policy Trainer, never imports game code.
The comparison is 1 vs 2 actual *training* Python processes on one GitHub runner.
There is no claim of native Godot, W0, pretrained SLM or four-way cross-runner overlap.
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

NICHES = (
    "navigator", "branch_artist", "recoverer",
    "texture_worm", "material_cautious", "chaotic_brush",
)
SOURCE_FILE = "experiments/driv3r/training/population/population_lab.py"
MAX_PHASE_SECONDS = 420
MAX_RSS_KIB_PER_WORKER = 1_200_000
MAX_RSS_KIB_PAIR = 2_400_000
POLL_SECONDS = 0.05


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request_check(r):
    assert r["schema"] == "cybersand.fleet-training.request/1"
    assert r["upstream_commit"] == "534cd8ffac6fbe33a714ae03a62c16045bf10259"
    assert r["upstream_source_blob"] == "8a974a072dd8a5172f9242d83a373f3de345bb3b"
    assert r["request_id"] == "FT-001-20261116-CPU-population-2x2"
    assert set(r["seeds"]) == {"alpha", "beta"}
    assert len(set(r["seeds"]["alpha"] + r["seeds"]["beta"])) == 4
    for pair in r["seeds"].values():
        assert len(pair) == 2 and all(type(s) is int and 0 <= s < 2**32 for s in pair)
    assert r["population"] == 2 and r["train_episodes"] == 12
    assert r["selection_episodes"] == 4 and r["test_episodes"] == 6


def source_check(upstream, r):
    got = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if got != r["upstream_commit"]:
        raise RuntimeError("Unpinned upstream commit: " + got)
    blob = subprocess.check_output(
        ["git", "-C", str(upstream), "hash-object", SOURCE_FILE], text=True
    ).strip()
    if blob != r["upstream_source_blob"]:
        raise RuntimeError("Upstream population source blob mismatch: " + blob)
    source = upstream / SOURCE_FILE
    if not source.is_file():
        raise RuntimeError("Pinned population trainer missing")
    return source


def rss_kib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return 0
    return 0


def verified_result(path, seed, req):
    result = json.loads((path / "results.json").read_text())
    if (result["format"] != "population-research-v1" or result["seed"] != seed
        or result["native"] is not False or result["language_model"] is not False
        or result["candidate_count"] != 18
        or set(result["champion_ids"]) != set(NICHES)
        or result["train_seed_count"] != req["train_episodes"]
        or result["selection_seed_count"] != req["selection_episodes"]
        or result["heldout_seed_count"] != req["test_episodes"]):
        raise AssertionError("Trainer identity, provenance or split mismatch")
    if not all(len(result["seed_hashes"][x]) == 64 for x in ("train", "selection", "heldout")):
        raise AssertionError("Dataset seed SHA unavailable")
    updates = 0
    for record in result["selection_lineage"]:
        n = int(record["training"]["optimizer_steps"])
        if n < 1:
            raise AssertionError("Missing real optimizer step receipts")
        updates += n
    if len(result["selection_lineage"]) != 18 or updates < 18:
        raise AssertionError("Not all candidates were optimized")
    files = {}
    for niche in NICHES:
        for rel in (f"models/{niche}.json", f"gallery/{niche}.svg",
                    f"gallery/{niche}-trajectory.json"):
            file = path / rel
            if not file.is_file() or not file.stat().st_size:
                raise AssertionError("Missing original policy/gallery: " + rel)
            files[rel] = sha_file(file)
        model = json.loads((path / "models" / f"{niche}.json").read_text())
        if model["format"] != "population-mlp-v1" or model["optimizer_steps"] < 1:
            raise AssertionError("Not an actual optimized population policy")
        trace = json.loads((path / "gallery" / f"{niche}-trajectory.json").read_text())
        if trace["synthetic"] is not True or trace["native"] is not False:
            raise AssertionError("Native authenticity falsehood")
        obs = {tuple(x) for x in trace["obstacles"]}
        protected = {tuple(x) for x in trace["protected"]}
        painted = {tuple(x) for x in trace["painted"]}
        if protected & painted or obs & painted:
            raise AssertionError("Protected or obstructed cell overwritten")
    files["results.json"] = sha_file(path / "results.json")
    return {"seed": seed, "candidate_count": 18, "optimizer_updates": updates,
            "heldout_reward": {n: result["heldout_champions"][n]["mean_reward"] for n in NICHES},
            "model_sha256": {n: files[f"models/{n}.json"] for n in NICHES},
            "source_faithful_synthetic_population": True, "native": False,
            "files_sha256": files}


def execute_phase(phase, pair, upstream_source, out, req):
    jobs, streams = [], []
    phase_start = time.perf_counter()
    child_env = dict(os.environ)
    child_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                     NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    try:
        for idx, seed in enumerate(pair):
            target = out / phase / str(seed)
            target.parent.mkdir(parents=True, exist_ok=True)
            logfile = target.parent / f"{seed}.stdout.log"
            stderrfile = target.parent / f"{seed}.stderr.log"
            stdout = logfile.open("w")
            stderr = stderrfile.open("w")
            streams.extend([stdout, stderr])
            args = [sys.executable, str(upstream_source), "--output", str(target),
                    "--seed", str(seed), "--population", str(req["population"]),
                    "--train-episodes", str(req["train_episodes"]),
                    "--selection-episodes", str(req["selection_episodes"]),
                    "--test-episodes", str(req["test_episodes"])]
            p = subprocess.Popen(args, stdout=stdout, stderr=stderr, env=child_env,
                                 start_new_session=True)
            jobs.append({"seed": seed, "lease": f"{req['request_id']}-{phase}-{seed}",
                         "pid": p.pid, "process": p, "started": time.perf_counter(),
                         "peak_rss_kib": 0, "finished": None, "returncode": None})
            if phase == "serial":
                monitor([jobs[-1]], phase_start, out, MAX_PHASE_SECONDS)
        if phase == "dual":
            monitor(jobs, phase_start, out, MAX_PHASE_SECONDS)
    except BaseException:
        for j in jobs:
            if j["process"].poll() is None:
                os.killpg(j["pid"], signal.SIGKILL)
                j["process"].wait()
        raise
    finally:
        for s in streams:
            s.close()
    result = []
    for j in jobs:
        if j["returncode"] != 0:
            raise RuntimeError(f"{phase} seed {j['seed']} exit {j['returncode']} (preserved stderr)")
        verified = verified_result(out / phase / str(j["seed"]), j["seed"], req)
        result.append({k: v for k, v in j.items() if k != "process"} | {
            "elapsed_s": round(j["finished"] - j["started"], 5),
            "status": "completed", "training": verified})
    return {"phase": phase, "wall_s": round(time.perf_counter() - phase_start, 5),
            "workers": result}


def monitor(jobs, phase_start, out, timeout):
    sampled_overlap_s = 0.
    while any(j["returncode"] is None for j in jobs):
        if time.perf_counter() - phase_start > timeout:
            raise TimeoutError("Bounded training phase exceeded deadline")
        rss = 0
        live = 0
        for j in jobs:
            if j["returncode"] is not None:
                continue
            p = j["process"]
            code = p.poll()
            if code is None:
                live += 1
                m = rss_kib(j["pid"])
                j["peak_rss_kib"] = max(j["peak_rss_kib"], m)
                rss += m
                if m > MAX_RSS_KIB_PER_WORKER:
                    raise MemoryError("Actual model RSS exceeded per-process budget")
            else:
                j["returncode"] = code
                j["finished"] = time.perf_counter()
        if rss > MAX_RSS_KIB_PAIR:
            raise MemoryError("Actual two-worker combined RSS exceeded budget")
        if live == 2:
            sampled_overlap_s += POLL_SECONDS
        time.sleep(POLL_SECONDS)
    (out / f"{'dual' if len(jobs) == 2 else 'serial'}-overlap-samples.json").write_text(
        json.dumps({"two_live_process_sample_seconds": round(sampled_overlap_s, 3),
                    "sampling_interval_s": POLL_SECONDS,
                    "not_proof_of_simultaneous_optimizer_updates": True}) + "\n")


def run(req_path, upstream, output, runner):
    req = json.loads(req_path.read_text())
    request_check(req)
    source = source_check(upstream, req)
    if runner not in req["seeds"]:
        raise ValueError("Unknown runner cohort")
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.json").write_text(json.dumps({
        "request": req, "runner": runner,
        "source_commit": req["upstream_commit"],
        "source_blob": req["upstream_source_blob"],
        "source_file_sha256": sha_file(source),
        "class": "REAL_CPU_ADAM_SYNTHETIC_POLICY",
        "not": ["native Godot PAINT", "exact W0", "pretrained SLM",
                "independently attested cross-host four-way model overlap"]}, indent=2) + "\n")
    record = {"format": "cybersand.fleet-training.optimizer-benchmark/1",
              "request_id": req["request_id"], "runner": runner,
              "cohort_seeds": req["seeds"][runner],
              "source_commit": req["upstream_commit"],
              "source_blob": req["upstream_source_blob"],
              "actual_model_kind": "24-hidden-unit NumPy Adam game-policy MLP",
              "synthetic_world": "22x22 goal/material/protection/branch/retrace",
              "source_faithful_W0": False, "native": False,
              "epochs": [], "status": "starting"}
    try:
        for phase in ("serial", "dual"):
            record["status"] = "running"
            record["epochs"].append(execute_phase(phase, req["seeds"][runner], source, output, req))
        serial, dual = record["epochs"]
        assert len({j["pid"] for j in dual["workers"]}) == 2
        for a, b in zip(serial["workers"], dual["workers"]):
            if a["training"]["files_sha256"] != b["training"]["files_sha256"]:
                raise AssertionError("Serial/dual model/gallery/trajectory results differ")
        record["status"] = "completed"
        record["ratio_serial_to_dual"] = round(serial["wall_s"] / dual["wall_s"], 5)
        record["peak_sampled_pair_rss_kib"] = sum(j["peak_rss_kib"] for j in dual["workers"])
        record["two_pid_overlap_observed_on_same_runner"] = (
            json.loads((output / "dual-overlap-samples.json").read_text())
            ["two_live_process_sample_seconds"] > 0)
        record["new_native_material_edits"] = 0
        record["training_candidates_per_distinct_seed"] = 18
    except BaseException as exc:
        record["status"] = "failed"
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (output / "bench.json").write_text(json.dumps(record, indent=2, sort_keys=True,
                                                  default=str) + "\n")
        rows = []
        for f in sorted(output.rglob("*")):
            if f.is_file() and f.name != "SHA256SUMS":
                rows.append(f"{sha_file(f)}  {f.relative_to(output)}")
        (output / "SHA256SUMS").write_text("\n".join(rows) + "\n")
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--runner", choices=("alpha", "beta"), required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.request, a.upstream, a.output, a.runner), indent=2, default=str))


if __name__ == "__main__":
    main()
