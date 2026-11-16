#!/usr/bin/env python3
"""Independent original F2 four-resident receipt audit. No model/native execution.

The four genuine model process claims originate from the original primary
GitHub Actions artifact, not from this independent offline process.
Do not infer four simultaneous individual forward calls from overlapping
episode intervals; those calls have no per-forward wall-clock timestamp.
"""
import argparse
import hashlib
import io
import json
import math
import re
import zipfile
import zlib
from pathlib import Path

EXPECTED_SHA = "db96e9ddbf12ea1f075482a2daa116b50036e2aa48b1bc018771783d4aef884d"
SOURCE = "b44038a2e26850bf5f97457243f1b2f9c018d2ca"
MODEL_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
W0 = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
LEGAL = set(("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP"))
WORLD = {(0, 7013, "greebles"), (0, 7027, "pipework"),
         (1, 7043, "architecture"), (1, 7057, "pipework")}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check(archive, expected=EXPECTED_SHA):
    require(digest(archive) == expected, "original ZIP digest mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) == 65, "altered ZIP inventory")
        require(all(not p.startswith("/") and ".." not in Path(p).parts for p in names), "unsafe member")
        require(all(z.getinfo(p).file_size <= 200000 for p in names), "oversize member")
        require(all(z.getinfo(p).CRC == zlib.crc32(z.read(p)) for p in names), "ZIP CRC")
        def obj(path):
            return json.loads(z.read(path))
        checks = 0
        for slot in (0, 1):
            for row in z.read(f"fleet-{slot}/SHA256SUMS.txt").decode().splitlines():
                match = re.fullmatch(r"([0-9a-f]{64})  \./(.+)", row)
                require(match is not None, "bad checksum line")
                path = f"fleet-{slot}/{match[2]}"
                require(path in names and digest(z.read(path)) == match[1], "member SHA " + path)
                checks += 1
        require(checks == 62, "incomplete checksum inventory")
        master = obj("four-way-audit.json")
        require(master["schema"] == "cybersand.worm.sprocket-4-live-evidence/1"
                and master["source_sha"] == SOURCE, "master source mismatch")
        require(master["optimizer_steps_this_run"] == 0 and master["not_native"], "authority mismatch")
        attested = master["original_worm_processes"]
        require(len(attested) == 4, "not four model processes")
        seen_world, seen_pid, seen_lease, periods, rows = set(), set(), set(), [], []
        for slot in (0, 1):
            fleet = obj(f"fleet-{slot}/results/fleet.json")
            require(fleet["slot"] == slot and fleet["source_sha"] == SOURCE
                    and fleet["model_revision"] == MODEL_REV and fleet["w0_commit"] == W0, "slot provenance")
            require(not fleet["native"] and not fleet["sealed_final_opened"]
                    and fleet["optimizer_steps"] == 0, "native/FINAL/optimizer boundary")
            require(fleet["maximum_actually_live_pids"] == 2 and fleet["completed"] == 2
                    and fleet["failed"] == fleet["queued"] == 0 and fleet["decision_bound"] == 24,
                    "incomplete two-model runner")
            for w in fleet["workers"]:
                world = (slot, w["seed"], w["family"])
                require(world in WORLD and world not in seen_world, "world identity reused or unknown")
                seen_world.add(world)
                require((slot, w["pid"]) not in seen_pid and w["pid"] > 0, "PID reused")
                seen_pid.add((slot, w["pid"]))
                require(w["lease"] not in seen_lease, "lease reused")
                seen_lease.add(w["lease"])
                require(w["real_trained_six_score_forwards"] == 24, "missing model forwards")
                require(any(i == slot and original == w for i, original in attested),
                        "missing master process witness")
                require(w["start_utc"] < w["inference_start_utc"] < w["inference_end_utc"] <= w["end_utc"],
                        "invalid process timeline")
                periods.append((w["inference_start_utc"], w["inference_end_utc"]))
                base = f'fleet-{slot}/results/{w["lease"]}/'
                complete = obj(base + "complete.json")
                identity = obj(base + "identity.json")
                require(complete["lease"] == identity["lease"] == w["lease"]
                        and complete["seed"] == identity["seed"] == w["seed"]
                        and complete["pid"] == w["pid"] and complete["family"] == w["family"],
                        "episode/identity mismatch")
                require(complete["revision"] == MODEL_REV and not complete["native"]
                        and complete["optimizer_steps"] == 0 and complete["source_faithful_W0_not_native"],
                        "episode provenance mismatch")
                require(complete["trained"]["steps"] == complete["real_trained_six_score_forwards"] == 24,
                        "episode forwards incomplete")
                original_dir = base + "original-png-and-json/"
                traces = sorted(p for p in names if p.startswith(original_dir) and p.endswith(".json"))
                require(len(traces) == 4, "missing paired model/control traces")
                scored_name = next((p for p in traces if "sprocket-f1-" in p), None)
                require(scored_name, "missing trained model trace")
                for p in traces:
                    trace = obj(p)
                    require(trace["seed"] == w["seed"] and trace["family"] == w["family"]
                            and trace["size"] == 96 and trace["steps_executed"] == 24
                            and len(trace["events"]) == 24, "malformed W0 trajectory")
                    require(all(event["raw_legal"] and event["applied"] in LEGAL
                                and event["applied"] in event["observation"]["legal"]
                                for event in trace["events"]), "illegal applied action")
                    png = p[:-5] + ".png"
                    require(png in names and z.read(png).startswith(b"\x89PNG\r\n\x1a\n"),
                            "paired original PNG missing")
                trained = obj(scored_name)
                receipts = trained["score_receipts"]
                require(len(receipts) == 24, "missing per-decision score receipts")
                changed = 0
                for tick, (r, e) in enumerate(zip(receipts, trained["events"])):
                    require(r["tick"] == e["observation"]["tick"] == tick, "discontinuous tick")
                    require(r["model_forward"] and r["trained_adapter_active"]
                            and r["is_model_logit_conditioned"] and not r["is_eos5_completion"],
                            "model-forward semantics mismatch")
                    require(set(r["full_six_token_logits"]) == LEGAL
                            and all(math.isfinite(v) for v in r["full_six_token_logits"].values()),
                            "invalid six-action scores")
                    require(r["actual_W0_applied"] == r["selected_action"] == e["applied"]
                            and r["selected_host_legal"], "scored/applied divergence")
                    require(r["selected_action"] in r["host_policy_legal"]
                            and set(r["host_policy_legal"]).issubset(e["observation"]["legal"]),
                            "host policy bypass")
                    require(0 <= r["host_uniform_draw"] < 1, "invalid host RNG")
                    changed += bool(r["differs_from_flat"])
                require(changed == complete["trained"]["differs_same_observation_flat"]
                        and trained["metrics"] == complete["trained"]["metrics"],
                        "metrics/scores inconsistent")
                rows.append({
                    "slot": slot, "pid": w["pid"], "seed": w["seed"], "family": w["family"],
                    "lease": w["lease"], "forwards": len(receipts), "different_from_flat": changed,
                    "unique_cells": trained["metrics"]["unique_trail_cells"],
                    "branches": trained["metrics"]["branches"], "retraces": trained["metrics"]["retraces"],
                    "revisits": trained["metrics"]["total_revisit_contacts"],
                    "max_reach": complete["trained"]["spatial"]["max_manhattan_from_start"],
                    "control_cells": {p: complete["controls"][p]["metrics"]["unique_trail_cells"]
                                      for p in ("random", "wall", "fidget")},
                    "original_png_sha256": digest(z.read(scored_name[:-5] + ".png")),
                })
        require(seen_world == WORLD and len(rows) == 4, "not four independent worlds")
        overlap = max(0., min(end for _, end in periods) - max(start for start, _ in periods))
        require(overlap > 0 and abs(overlap - master["actual_four_MODEL_INFERENCE_interval_overlap_seconds"]) < 0.01,
                "four-way model episode overlap mismatch")
        return {
            "schema": "cybersand.worm.sprocket-support-independent-F2/1",
            "original_zip_sha256": digest(archive), "original_source": SOURCE,
            "model_revision": MODEL_REV, "w0_source": W0,
            "verified_member_sha256": checks, "zip_crc": len(names),
            "source_faithful_original_png_json_pairs": 16,
            "host_attested_real_model_forwards": sum(r["forwards"] for r in rows),
            "four_way_model_episode_overlap_seconds": round(overlap, 3),
            "individual_simultaneous_forward_calls_proven": False,
            "independent_weight_inference_repeated": False, "native": False,
            "optimizer_steps": 0, "raw_eos5_completions_proven": False,
            "mean_trained_unique_cells": sum(r["unique_cells"] for r in rows)/4,
            "mean_control_unique_cells": {p: sum(r["control_cells"][p] for r in rows)/4
                                          for p in ("random", "wall", "fidget")},
            "streams": rows,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_zip", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(check(args.original_zip.read_bytes()), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
