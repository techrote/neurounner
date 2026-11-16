"""FT-004: frozen FT-003 checkpoints on shifted worlds and longer horizons.

No optimizer, game modification, native PAINT, pretrained SLM or W0 claim.
Every action is selected by the saved tiny NumPy policy from the host legal menu.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import fleet_bench as base

SCHEMA = "cybersand.fleet-training.transfer-request/1"
ID = "FT-004-20261116-frozen-dose-ood-long"
SOURCE_COMMIT = "534cd8ffac6fbe33a714ae03a62c16045bf10259"
SOURCE_BLOB = "8a974a072dd8a5172f9242d83a373f3de345bb3b"
FT003_SHA = "fec6d5e254a0bbe590b44ba26bbb7427960af507a5c6fe78b1d7e5fe5389c910"
SCENARIOS = ("fresh30", "long60", "dense60", "shifted_guard60")
DOSES = (12, 48)
SEEDS = (9401, 9403)
WORLDS = 6
MAX_SECONDS = 240
MAX_RSS_KIB = 1200000


def request_check(r):
    want = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": SOURCE_COMMIT, "source_blob": SOURCE_BLOB,
        "ft003_actions_run": 43797061143, "ft003_artifact": 14194214028,
        "ft003_archive_sha256": FT003_SHA, "seeds": list(SEEDS),
        "doses": list(DOSES), "scenarios": list(SCENARIOS),
        "worlds_per_scenario": WORLDS, "max_seconds": MAX_SECONDS,
        "rss_cap_kib": MAX_RSS_KIB, "optimizer_steps": 0,
    }
    if r != want:
        raise ValueError("FT-004 immutable source, custody, or budget mismatch")
    return r


def load_source(upstream):
    source = base.source_check(upstream, {
        "upstream_commit": SOURCE_COMMIT, "upstream_source_blob": SOURCE_BLOB
    })
    spec = importlib.util.spec_from_file_location("ft004_pinned_population", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if tuple(module.NICHES) != base.NICHES or len(module.FEATURES) != 20:
        raise ValueError("Pinned host action/feature contract drift")
    return module, source


def archive_check(root, source, request):
    import training_dose as ft003
    files = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or str(p) in files:
            raise ValueError("Unsafe or duplicated archived file")
        files[str(p)] = digest
        if not (root / p).is_file() or base.sha_file(root / p) != digest:
            raise ValueError("Altered FT-003 original member: " + str(p))
    if not files or (root / "report.json").name not in files:
        raise ValueError("Missing original FT-003 manifest")
    report = json.loads((root / "report.json").read_text())
    if (report.get("status") != "completed" or report.get("source_commit") != SOURCE_COMMIT
        or report.get("source_blob") != SOURCE_BLOB or
        report.get("source_sha256") != base.sha_file(source) or
        report.get("request_sha256") != base.sha_file(request.parent / "request_dose.json")
        or len(report.get("phases", [])) != 2):
        raise ValueError("FT-003 source/training receipt provenance mismatch")
    for dose in DOSES:
        for seed in SEEDS:
            p = root / f"dose_{dose}" / str(seed) / "results.json"
            if str(p.relative_to(root)) not in files:
                raise ValueError("Original results not inventoried")
            results = json.loads(p.read_text())
            ft003.validate_split(results, seed, dose, {
                "selection_episodes": 4, "test_episodes": 12
            })
            for niche in base.NICHES:
                model = f"dose_{dose}/{seed}/models/{niche}.json"
                if model not in files:
                    raise ValueError("Checkpoint missing from original SHA ledger")
    return {"original_members_checked": len(files), "original_source_sha256": base.sha_file(source),
            "ft003_report_sha256": files["report.json"], "original_zip_sha256": FT003_SHA}


def load_policy(pop, file):
    data = json.loads(file.read_text())
    if (data.get("format") != "population-mlp-v1" or
        data.get("features") != list(pop.FEATURES) or
        data.get("hidden") != 24 or
        type(data.get("optimizer_steps")) is not int or data["optimizer_steps"] < 1):
        raise ValueError("Untrained, foreign or mismatched checkpoint")
    p = pop.Policy(0, hidden=24, lr=float(data["lr"]),
                   temperature=float(data["temperature"]))
    expected = {k:v.shape for k,v in p.p.items()}
    if set(data["parameters"]) != set(expected):
        raise ValueError("Checkpoint parameter catalogue differs")
    for key, shape in expected.items():
        v = np.asarray(data["parameters"][key], dtype=np.float64)
        if v.shape != shape or not np.isfinite(v).all():
            raise ValueError("Malformed or nonfinite checkpoint tensor")
        p.p[key] = v
    return p


def heldout_world_seed(pop, training_seed, index):
    seed = pop.stable_seed(ID, "OOD-UNSEEN", training_seed, index)
    prior = {pop.stable_seed(training_seed, domain, i)
             for domain, n in (("train", 48), ("selection", 4), ("test", 12))
             for i in range(n)}
    if seed in prior:
        raise ValueError("FT-004 OOD world seed collides with upstream custody")
    return seed


def world_for(pop, seed, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Unknown distribution shift")
    w = pop.World.generate(seed, max_steps=30 if scenario == "fresh30" else 60)
    if scenario == "dense60":
        rng = np.random.default_rng(pop.stable_seed(ID, "density", seed))
        reserved = {w.cursor, w.goal, w.second_goal}
        for y in range(1, w.height - 1):
            for x in range(1, w.width - 1):
                p = (x, y)
                if p not in reserved and p not in w.protected and p not in w.obstacles:
                    if rng.random() < .09:
                        w.obstacles.add(p)
        # Guarantee at least one initial legal movement without altering protected cells.
        if not any(a in pop.DIRECTIONS for a in w.legal()):
            w.obstacles.discard((w.cursor[0] + 1, w.cursor[1]))
    elif scenario == "shifted_guard60":
        new = {(x,y) for x in range(10, 13) for y in range(6, 13)}
        new -= {w.cursor, (17, 4), (17, 17)}
        w.protected = new
        w.goal, w.second_goal = (17, 4), (17, 17)
        w.obstacles -= new
        w.obstacles.discard(w.goal)
        w.obstacles.discard(w.second_goal)
    return w


def state_digest(w):
    doc = {"cursor": w.cursor, "goal": w.goal, "heading": w.heading,
           "step": w.steps, "stopped": w.stopped, "switched": w.switched,
           "painted": sorted(w.painted), "protected": sorted(w.protected),
           "obstacles": sorted(w.obstacles),
           "parents": sorted((k,v) for k,v in w.parent.items())}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def execute(pop, policy, niche, seed, scenario, save_trace=False):
    w = world_for(pop, seed, scenario)
    reward, receipts, retraces, forwards, violations = 0., [], 0, 0, 0
    rng = np.random.default_rng(pop.stable_seed(ID, niche, "control", seed, scenario))
    while not w.stopped and w.steps < w.max_steps:
        menu = w.legal()
        X = np.stack([w.feature(a) for a in menu])
        if policy == "nonstop_random":
            candidates = [i for i,a in enumerate(menu) if a != "STOP"]
            idx = int(rng.choice(candidates)) if candidates else menu.index("STOP")
        elif policy == "persistent":
            idx = next((i for i,a in enumerate(menu) if a == w.heading), None)
            if idx is None:
                idx = next((i for i,a in enumerate(menu) if a in pop.DIRECTIONS), None)
            if idx is None:
                idx = menu.index("RETRACE") if "RETRACE" in menu else menu.index("STOP")
        else:
            scores = policy.logits(X)
            if not np.isfinite(scores).all():
                raise ValueError("Model produced nonfinite action scores")
            idx = int(np.argmax(scores))
            forwards += 1
        chosen = menu[idx]
        before = state_digest(w)
        reward += pop.utility(niche, X[idx], w)
        w.step(chosen)
        if chosen == "RETRACE":
            retraces += 1
        if w.painted & (w.obstacles | w.protected):
            violations += 1
            raise AssertionError("Protected/blocked material was painted")
        if save_trace:
            receipts.append({"turn": w.steps, "menu": menu, "choice": chosen,
                             "before": before, "after": state_digest(w)})
    return {"seed": seed, "scenario": scenario, "niche": niche,
            "reward": round(reward, 5), "reward_per_decision": round(reward/max(1,w.steps),5),
            "decisions": w.steps, "model_forwards": forwards,
            "new_ink": len(w.painted) - 1, "retraces": retraces,
            "early_stop": bool(w.stopped and w.steps < w.max_steps),
            "protected_violations": violations, "descriptor": w.descriptor(),
            "receipts": receipts if save_trace else None}, w


def mean(items, key):
    return round(sum(float(x[key]) for x in items) / len(items), 5)


def summarize(items):
    return {k: mean(items,k) for k in
            ("reward", "reward_per_decision", "decisions", "new_ink", "retraces", "early_stop")}


def audit_receipts(pop, records, niche, seed, scenario):
    w = world_for(pop, seed, scenario)
    for record in records:
        if record["before"] != state_digest(w) or record["menu"] != w.legal():
            raise AssertionError("Synthetic replay pre-state/menu mismatch")
        if record["choice"] not in record["menu"] or record["turn"] != w.steps+1:
            raise AssertionError("Unlawful choice or nonmonotonic receipt")
        w.step(record["choice"])
        if record["after"] != state_digest(w):
            raise AssertionError("Synthetic replay material/cursor mismatch")
    return True


def run(reqfile, upstream, original, output):
    r = request_check(json.loads(reqfile.read_text()))
    pop, source = load_source(upstream)
    previous_request = Path(__file__).parent / "request_dose.json"
    if not previous_request.is_file():
        raise ValueError("FT-003 request source missing")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "cybersand.fleet-training.transfer-evidence/1",
              "request_id": ID, "source_commit": SOURCE_COMMIT,
              "request_sha256": base.sha_file(reqfile), "source_sha256": base.sha_file(source),
              "original_archive_sha256": FT003_SHA, "model_kind": "frozen 24-unit NumPy policy",
              "classification": "REAL_CHECKPOINT_SYNTHETIC_INFERENCE_NOT_NATIVE_OR_W0",
              "optimizer_steps": 0, "native_edits": 0, "status": "starting",
              "rows": [], "workers": [], "failures": []}
    started = time.monotonic()
    try:
        report["original_audit"] = archive_check(original, source, previous_request)
        report["status"] = "running"
        seen = set()
        for training_seed in SEEDS:
            world_seeds = [heldout_world_seed(pop, training_seed, i) for i in range(WORLDS)]
            report.setdefault("world_seeds", {})[str(training_seed)] = world_seeds
            for niche in base.NICHES:
                for scenario in SCENARIOS:
                    controls = {}
                    for arm in ("nonstop_random", "persistent"):
                        episodes = [execute(pop, arm, niche, s, scenario)[0] for s in world_seeds]
                        controls[arm] = summarize(episodes)
                    pair = {}
                    for dose in DOSES:
                        path = original / f"dose_{dose}/{training_seed}/models/{niche}.json"
                        policy = load_policy(pop, path)
                        lease = f"{ID}/{training_seed}/{dose}/{niche}/{scenario}"
                        if lease in seen:
                            raise AssertionError("Duplicate source/model/world lease")
                        seen.add(lease)
                        episodes = []
                        for i,s in enumerate(world_seeds):
                            item, world = execute(pop, policy, niche, s, scenario, save_trace=(i == 0))
                            if item["model_forwards"] != item["decisions"]:
                                raise AssertionError("Missing genuine model-scored host action")
                            if i == 0:
                                audit_receipts(pop, item["receipts"], niche, s, scenario)
                                pathout = output / f"gallery/{training_seed}/{dose}/{scenario}"
                                pathout.mkdir(parents=True, exist_ok=True)
                                (pathout / f"{niche}.svg").write_text(pop.draw_svg(world))
                                (pathout / f"{niche}.json").write_text(json.dumps(item, sort_keys=True)+"\n")
                            episodes.append(item)
                        pair[dose] = summarize(episodes)
                        report["workers"].append({
                            "lease_id": lease, "status": "completed", "source_seed": training_seed,
                            "dose": dose, "niche": niche, "scenario": scenario,
                            "checkpoint_sha256": base.sha_file(path), "world_seeds": world_seeds,
                            "model_forwards": sum(x["model_forwards"] for x in episodes),
                            "host_applied_actions": sum(x["decisions"] for x in episodes),
                            "native_material_edits": 0, "new_synthetic_ink": sum(x["new_ink"] for x in episodes)})
                    report["rows"].append({
                        "training_seed": training_seed, "niche": niche, "scenario": scenario,
                        "dose12": pair[12], "dose48": pair[48], "controls": controls,
                        "long_minus_short_reward_per_decision":
                            round(pair[48]["reward_per_decision"]-pair[12]["reward_per_decision"],5),
                        "long_minus_short_new_ink":
                            round(pair[48]["new_ink"]-pair[12]["new_ink"],5)})
                    if time.monotonic()-started > MAX_SECONDS:
                        raise TimeoutError("FT-004 inference budget exceeded")
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
                        raise MemoryError("FT-004 RSS exceeded cap")
        report["status"] = "completed"
        report["aggregate"] = {
            "matched_comparison_rows": len(report["rows"]),
            "better_48_per_decision": sum(x["long_minus_short_reward_per_decision"] > 0 for x in report["rows"]),
            "worse_48_per_decision": sum(x["long_minus_short_reward_per_decision"] < 0 for x in report["rows"]),
            "total_forward_selected_actions": sum(x["model_forwards"] for x in report["workers"]),
            "total_synthetic_ink": sum(x["new_synthetic_ink"] for x in report["workers"]),
            "inference_wall_s": round(time.monotonic()-started,5),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "concurrent_resident_model_processes": 1,
            "genuine_neural_inference": True, "native_godot": False}
    except BaseException as exc:
        report["status"] = "failed"
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2)+"\n")
        (output / "SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--ft003", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    data = run(args.request, args.upstream, args.ft003, args.output)
    print(json.dumps({"status":data["status"], "aggregate":data["aggregate"]},sort_keys=True))
