#!/usr/bin/env python3
"""W12 original two-VM/four-resident LFM fleet receipt audit; NO MODEL EXECUTION.

Read ONLY the two immutable W12 producer artifact ZIPs. Reproduce host sampling,
W0 traces and PNGs against pinned W0, and independently measure cross-VM PID
overlap. These are model-scored HOST samples, not greedy LFM art.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import zipfile

RUN = 43766388326
SOURCE = "0eea751c37d69589717681eca1703882d17587a9"
W0 = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
MODEL_REV = "fc6221ca597f3315e4f82fc2df606783267b34ba"
ADAPTER = "886750e362f07e3540d164be82ffcdb3ebb1962f86164d57f9f4aed5a7bb9f39"
ARTIFACTS = {
    "alpha": ("6e25c147117453e97fe3d27fe44e196abaf63cd9f026b276a0956895f384c079", (4311, 4331)),
    "beta": ("54a3d314d6e45c9ec6395e0778462fa6e4392f8189662a14ee2cf348ca332d90", (4351, 4371)),
}
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")


def need(ok, what):
    if not ok:
        raise ValueError(what)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit_shard(path, shard, worm):
    need(digest(path.read_bytes()) == ARTIFACTS[shard][0], "immutable original ZIP hash mismatch: " + shard)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        need(z.testzip() is None and len(names) == len(set(names)), "ZIP CRC or duplicate entries")
        need(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts and "\\" not in n for n in names), "unsafe ZIP")
        def read(suffix):
            return z.read(suffix)
        def obj(suffix):
            return json.loads(read(suffix))
        pre = "w12-fleet/" + shard + "/"
        summary = obj(pre + "summary.json")
        need(summary["schema"] == "lfm-worm-w12-two-vm-two-process-four-live-capacity-v1" and summary["status"] == "completed", "not W12 complete")
        need(summary["shard"] == shard and str(summary["run_id"]) == str(RUN) and summary["cpu_affinity_count"] == 4, "wrong Actions run/CPU shard")
        need(summary["model"] == "LiquidAI/LFM2.5-VL-450M" and summary["model_revision"] == MODEL_REV and summary["adapter_sha256"] == ADAPTER, "model pin")
        need(summary["original_W10_source"] == "1e57a4df8d344b209a14df886dd898211dcc4918" and summary["original_W10_run"] == 43754365984, "checkpoint origin")
        need(summary["new_optimizer_updates"] == 0 and summary["native_execution"] is False and summary["sealed_FINAL_opened"] is False, "execution custody")
        need(read("diagnostics/evaluator-source.txt").decode().strip() == SOURCE and read("diagnostics/w0-source.txt").decode().strip() == W0, "source mismatch")
        need(summary["independent_seeds"] == list(ARTIFACTS[shard][1]) and summary["actual_inference_model_forwards"] == summary["actual_W0_applied_decisions"] == 24, "source decision count")
        need(digest(read(pre + "original-png-contact.png")) == summary["original_png_contact_sha256"], "contact PNG hash")
        pids = summary["distinct_loaded_pids"]
        leases = summary["distinct_lease_ids"]
        need(len(set(pids)) == len(set(leases)) == 2, "process and lease isolation")
        samples = obj(pre + "resource-samples.json")
        live = [s["epoch_ns"] for s in samples if set(pids).issubset(s["alive"]) and all((s["rss_kib"].get(str(p)) or 0) > 0 for p in pids)]
        need(len(live) >= 3 and all(0 < (b-a)/1e9 < 1.5 for a,b in zip(live, live[1:])), "no continuously sampled paired PIDs")
        need(live[0] >= summary["start_epoch_ns"] and live[-1] <= summary["end_epoch_ns"], "live samples outside run")
        rows = []
        for idx, seed in enumerate(ARTIFACTS[shard][1]):
            lease = leases[idx]
            base = pre + lease + "/"
            c, l, ready, model = (obj(base + f) for f in ("completed.json", "lease.json", "ready.json", "model-readback.json"))
            trace = obj(base + "original-w0.json")
            decisions = obj(base + "model-decisions.json")
            original_png = read(base + "original-w0.png")
            need(c["status"] == "completed" and c["lease_id"] == l["lease_id"] == ready["lease_id"] == lease, "bad episode lease")
            need(c["pid"] == l["runner_pid"] == ready["pid"] == pids[idx] and c["runner"] == summary["runner_name"], "bad episode process/runner")
            need(c["seed"] == l["seed"] == ready["seed"] == trace["seed"] == seed and c["family"] == l["family"] == trace["family"], "bad seed/family")
            need(c["model"] == "LiquidAI/LFM2.5-VL-450M" and c["revision"] == MODEL_REV and c["adapter_sha256"] == model["adapter_sha256"] == ready["adapter_sha256"] == ADAPTER, "wrong adapter")
            need(model["model_class"] == "Lfm2VlForConditionalGeneration" and model["processor"] == "Lfm2VlProcessor" and model["original_adamw_steps"] == 45 and model["new_optimizer_steps"] == 0, "full model readback")
            need(c["genuine_model_forward"] is True and c["model_score_forwards"] == c["host_applied_steps"] == len(decisions) == len(trace["events"]) == 12, "missing model forwards")
            need(c["new_optimizer_steps"] == 0 and c["native_execution"] is False and c["sealed_FINAL_opened"] is False, "optimizer/native boundary")
            need(trace["contract"] == "cybersand.worm.observation-action/0-experimental" and trace["engine"] == "python-stdlib-baseline-not-native" and trace["mode"] == "callback" and trace["size"] == 96 and trace["step_budget"] == 12, "W0 contract changed")
            need(digest(original_png) == c["png_sha256"], "original PNG hash")
            selected, scored_top = [], []
            for tick, (d, e) in enumerate(zip(decisions, trace["events"])):
                r = d["sampling_receipt"]
                a = d["raw_choice_selected_by_host_sampler"]
                need(d["tick"] == e["observation"]["tick"] == tick and d["legal"] == r["legal"] == e["observation"]["legal"], "decision context mismatch")
                need(a == r["choice"] == e["raw_choice"] == e["applied"] and e["raw_legal"] is True and r["sampler"] == "learned", "wrong sampled action")
                legal = [a for a in ACTIONS if a in r["legal"] and (a != "STOP" or len(r["legal"]) == 1)]
                need(legal == r["sampled_legal"], "legal mask changed")
                scores = d["model_six_code_scores"]
                logits = scores["candidate_logits"]
                need(set(logits) == set(ACTIONS) and scores["six_code_top_action"] == max(ACTIONS, key=lambda a: logits[a]), "score top not genuine")
                scored_top.append(scores["six_code_top_action"])
                scaled = [logits[a]/2.5 for a in legal]
                weights = [math.exp(x-max(scaled)) for x in scaled]
                conditioned = {a:w/sum(weights) for a,w in zip(legal,weights)}
                probs = {a:.7*conditioned[a]+.3/len(legal) for a in legal}
                for a in legal:
                    need(math.isclose(conditioned[a], r["conditioned_model_probs"][a], rel_tol=1e-6, abs_tol=1e-7), "wrong conditioned probability")
                    need(math.isclose(probs[a], r["sample_probs"][a], rel_tol=1e-6, abs_tol=1e-7), "wrong model-flat probability")
                u = r["host_u"]
                need(type(u) is float and 0 <= u < 1, "invalid host random draw")
                cumulative = 0
                expected = legal[-1]
                for a in legal:
                    cumulative += probs[a]
                    if u < cumulative:
                        expected = a
                        break
                need(expected == d["raw_choice_selected_by_host_sampler"], "sample choice not reproducible")
                selected.append(expected)
            replay = worm.run(seed=seed, family=c["family"], mode="replay", steps=12, size=96, raw_actions=selected)
            for field in ("events", "trajectory", "obstacles", "branch_points", "metrics"):
                need(replay[field] == trace[field], "W0 replay mismatch: " + field)
            need(worm.png_bytes(replay) == original_png, "W0 PNG byte replay mismatch")
            baselines = {name: worm.run(seed=seed, family=c["family"], mode=name, steps=12, size=96)["metrics"] for name in ("random", "wall", "fidget")}
            rows.append({
                "lease": lease, "shard": shard, "runner": summary["runner_name"], "pid": c["pid"],
                "seed": seed, "family": c["family"], "original_png_sha256": digest(original_png),
                "genuine_score_forwards": 12, "score_top_counts": dict(Counter(scored_top)),
                "host_sampled_counts": dict(Counter(selected)), "W0_model": trace["metrics"],
                "matched_W0_controls": baselines, "raw_greedy_first_action": c["raw_greedy_probe"]["raw_action"],
            })
        return {"runner": summary["runner_name"], "shard": shard, "window": (live[0], live[-1]),
                "peak_sampled_aggregate_rss_kib": summary["peak_aggregate_resident_rss_kib"],
                "zip_entries": len(names), "episodes": rows}


def audit(alpha, beta, worm):
    shards = [audit_shard(alpha, "alpha", worm), audit_shard(beta, "beta", worm)]
    rows = [r for shard in shards for r in shard["episodes"]]
    need(len(rows) == 4 and len({r["lease"] for r in rows}) == len({r["seed"] for r in rows}) == 4, "not four independent episodes")
    need(shards[0]["runner"] != shards[1]["runner"], "not two independent runners")
    overlap_ns = min(s["window"][1] for s in shards) - max(s["window"][0] for s in shards)
    need(overlap_ns > 0, "zero cross-VM four-process sampled overlap")
    return {
        "schema": "cybersand.worm.lfm-w12-original-support-audit/1", "source_sha": SOURCE,
        "producer_run": RUN, "producer_artifact_hashes": {s: ARTIFACTS[s][0] for s in ARTIFACTS},
        "qualification": "PASS independent original W0 PNG, host receipt and residency audit",
        "simultaneous_sampled_four_resident_seconds": round(overlap_ns/1e9, 3),
        "four_distinct_leases_seeds_model_processes": True, "runner_count": 2,
        "actual_model_score_forwards": 48, "original_W0_png_replays": 4,
        "raw_score_top_counts": dict(sum((Counter(r["score_top_counts"]) for r in rows), Counter())),
        "original_training_updates_preserved": 45, "new_training_updates": 0,
        "trained_steering_useful_freeze": False, "native_execution": False,
        "no_neural_weight_reexecution_by_support": True, "episodes": rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", required=True, type=Path)
    p.add_argument("--beta", required=True, type=Path)
    p.add_argument("--w0-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.w0_dir))
    import worm
    result = audit(args.alpha, args.beta, worm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}, sort_keys=True))


if __name__ == "__main__":
    main()
