#!/usr/bin/env python3
"""Cross-host UTC witness for genuine DRIV3R four-resident inference.

A GitHub job being 'in_progress' does not establish four model processes.
Linux monotonic clocks from different runners are incomparable.  This module
binds monotonic model-episode intervals to bounded UTC offset ranges obtained
from HTTPS Date responses.  It NEVER calls the result native, independently
authenticated, or simultaneous *forward kernels*.  It performs no dispatch.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

SCHEMA = "driv3r.crossrunner-witness/1"
AUDIT = "driv3r.crossrunner-audit/1"
URL = "https://api.github.com/zen"
NS = 1_000_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bound(s):
    require(isinstance(s, dict), "missing HTTP Date sample")
    before, after = s.get("before_monotonic_ns"), s.get("after_monotonic_ns")
    require(type(before) is int and type(after) is int and
            0 < before <= after and after-before <= 3*NS,
            "bad monotonic sample / excessive remote RTT")
    require(s.get("url") == URL and s.get("transport") == "HTTPS", "untrusted time endpoint")
    stamp = parsedate_to_datetime(s["http_date"])
    require(stamp.tzinfo is not None, "HTTP Date lacks timezone")
    second = int(stamp.astimezone(timezone.utc).timestamp()) * NS
    # A second-granularity Date header has a full 1-second rounding interval.
    return second-after, second+NS-before


def offsets(samples):
    require(isinstance(samples, list) and len(samples) >= 2,
            "need at least two independent time probes per runner")
    ranges = [bound(s) for s in samples]
    require(all(samples[i]["before_monotonic_ns"] >
                samples[i-1]["after_monotonic_ns"] for i in range(1, len(samples))),
            "unordered/overlapping time probes")
    low, high = max(r[0] for r in ranges), min(r[1] for r in ranges)
    require(low <= high, "inconsistent external UTC anchoring")
    return low, high


def collect(n=2):
    require(type(n) is int and 2 <= n <= 5, "bounded sample count")
    result = []
    for _ in range(n):
        req = Request(URL, headers={"Cache-Control": "no-cache",
                                   "User-Agent": "CyberSand-DRIV3R-wormfleet"})
        before = time.monotonic_ns()
        with urlopen(req, timeout=5) as resp:
            require(resp.status == 200, "date provider HTTP failure")
            date = resp.headers.get("Date")
            resp.read(1024)
        after = time.monotonic_ns()
        require(bool(date), "missing HTTPS Date header")
        result.append({"url": URL, "transport": "HTTPS", "http_date": date,
                       "before_monotonic_ns": before, "after_monotonic_ns": after})
        time.sleep(.15)
    offsets(result)
    return result


def audit(runners):
    require(isinstance(runners, list) and len(runners) == 2,
            "need two independent runner witnesses")
    jobs, leases, pids, models = set(), set(), set(), []
    for runner in runners:
        require(runner.get("schema") == SCHEMA, "witness schema")
        job = runner.get("job_id")
        require(isinstance(job, str) and job.isdigit() and job not in jobs,
                "missing/duplicate GitHub job ID")
        jobs.add(job)
        low, high = offsets(runner.get("date_samples"))
        items = runner.get("episodes")
        require(isinstance(items, list) and len(items) == 2, "two PIDs per runner")
        local = []
        for e in items:
            require(e.get("job_id") == job and e.get("status") == "completed",
                    "incomplete/misattributed episode")
            require(e.get("native_execution") is False and e.get("training") is False,
                    "native/training claims forbidden in this synthetic witness")
            require(type(e.get("optimizer_steps")) is int and e["optimizer_steps"] == 0,
                    "optimizer work forbidden")
            lease, pid = e.get("lease_id"), e.get("pid")
            require(isinstance(lease, str) and len(lease) >= 10 and lease not in leases,
                    "missing/duplicate lease")
            require(type(pid) is int and pid > 0 and (job, pid) not in pids,
                    "missing/duplicate PID")
            leases.add(lease)
            pids.add((job, pid))
            require(all(isinstance(e.get(k), str) and len(e[k]) == 64 and
                        all(c in "0123456789abcdef" for c in e[k])
                        for k in ("source_sha256", "model_sha256", "original_artifact_sha256")),
                    "unfenced source/model/artifact")
            require(isinstance(e.get("model_revision"), str) and len(e["model_revision"]) >= 8,
                    "unpinned model revision")
            a,b=e.get("forward_start_ns"),e.get("forward_end_ns")
            require(type(a) is int and type(b) is int and 0 < a < b,
                    "missing actual forward episode interval")
            require(type(e.get("model_forwards")) is int and e["model_forwards"] >= 1,
                    "no genuine model-forward receipts")
            models.append(e)
            local.append((a,b))
        require(min(x[1] for x in local) > max(x[0] for x in local),
                "no local two-resident episode overlap")
        # Conservative: latest possible start and earliest possible end.
        runner["_latest_start"] = max(x[0] for x in local) + high
        runner["_earliest_end"] = min(x[1] for x in local) + low
    require(len(leases) == 4 and len(models) == 4, "four unique episodes")
    require(len({e["source_sha256"] for e in models}) == 1, "source drift")
    guaranteed = min(r["_earliest_end"] for r in runners) - max(
        r["_latest_start"] for r in runners)
    return {"schema": AUDIT, "producer_claimed_bounded_utc_episode_overlap_seconds":
            round(max(0,guaranteed)/NS,6),
            "producer_claimed_four_episode_overlap": guaranteed > 0,
            "four_simultaneous_model_forward_calls_proven": False,
            "independently_verified_active_models": 0,
            "native_model_paint_receipts": 0,
            "runner_jobs": sorted(jobs), "unique_leases": len(leases),
            "source_sha256": models[0]["source_sha256"],
            "qualification": "Producer evidence consistency only. Independently fetch actual "
            "GitHub jobs, source, complete per-PID model-forward receipts and original artifacts "
            "before promoting an overlap claim; HTTP Date is not signed."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("command",choices=("collect","audit"))
    p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.command == "collect":
        result={"schema":SCHEMA,"date_samples":collect()}
    else:
        require(a.input is not None, "audit input required")
        result=audit(json.loads(a.input.read_text()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")

if __name__=="__main__":
    main()
