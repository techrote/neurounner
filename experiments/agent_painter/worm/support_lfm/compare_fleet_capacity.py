#!/usr/bin/env python3
"""Source/seed-matched comparison of real four-vCPU 4x1 and 2x2 LFM inference.

No optimizer lease, model load, production/provider switch or synthetic speedup.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import json
import statistics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def summary(report, density):
    require(report["schema"] == "cybersand.worm.lfm-fleet-audit/1" and
            report["preflight_pass"] and report["four_concurrent_progress_windows"],
            "not a qualified original four-active inference campaign")
    episodes = report["episodes"]
    occupancy = Counter(e["runner"] for e in episodes)
    require(len(episodes) >= 4 and set(occupancy.values()) == {density},
            "wrong resident process topology")
    throughput = defaultdict(float)
    combined_rss = defaultdict(int)
    for e in episodes:
        p = e["pulses"]
        elapsed = p["b"]["observed_at_ms"] - p["a"]["observed_at_ms"]
        delta = p["b"]["observed_actions"] - p["a"]["observed_actions"]
        require(elapsed > 0 and delta > 0, "no real process inference progress")
        throughput[e["runner"]] += delta * 1000 / elapsed
        combined_rss[e["runner"]] += e["peak_rss_kib"]
    return {
        "runner_count": len(occupancy),
        "resident_per_runner": density,
        "total_resident_processes": len(episodes),
        "median_inference_latency_ms": statistics.median(
            e["median_inference_latency_ms"] for e in episodes),
        "observed_actions_per_second": round(sum(throughput.values()), 4),
        "mean_actions_per_second_per_runner": round(statistics.mean(throughput.values()), 4),
        "max_combined_rss_kib_per_runner": max(combined_rss.values()),
        "observed_rss_kib_per_runner": dict(combined_rss)}


def compare(one, two):
    a, b = one["episodes"], two["episodes"]
    require(one["decision_budget"] == two["decision_budget"], "decision-budget mismatch")
    require({x["seed"] for x in a} == {x["seed"] for x in b}, "different obstacle worlds")
    require({x["source_sha"] for x in a} == {x["source_sha"] for x in b},
            "source versions not matched")
    require({x["checkpoint_sha256"] for x in a} ==
            {x["checkpoint_sha256"] for x in b}, "checkpoint identity mismatch")
    return {
        "schema": "cybersand.worm.lfm-fleet-capacity/1",
        "four_runners_one_model": summary(one, 1),
        "two_runners_two_models": summary(two, 2),
        "warning": "Only observed original CPU inference performance, NOT approved 2x2 capacity. "
                   "Separate runner queue/resource/quota admission and ongoing RSS headroom remain necessary."}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one-report", required=True)
    ap.add_argument("--two-report", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    with open(a.one_report) as f:
        one = json.load(f)
    with open(a.two_report) as f:
        two = json.load(f)
    output = compare(one, two)
    with open(a.output, "w") as f:
        json.dump(output, f, sort_keys=True, indent=2)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
