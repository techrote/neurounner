"""STOP-margin ablation for dual-clock worms: TRAIN weights, DEV sequential controls.

A margin is a policy-side diagnostic, not a change to the legal action menu.
STOP remains admissible and may still win. Four predeclared variants share
the same exact training states and their underlying matched checkpoints.
Only synthetic DEVELOPMENT worlds are evaluated; held-out is never read.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import resource
import statistics
import time
import numpy as np

from . import dual_clock_lab as d
from .train import HORIZONS, MAX_RECOVERIES, sha_file, source_pins, _png
from .world import MOVE, World, digest, replay, specs

VERSION = "driv3r-stop-margin/1"
VARIANTS = (("executed_clock", 0.0), ("dual_clock", 0.0),
            ("dual_clock", 1.0), ("dual_clock", 2.5))
HORIZONS_EVAL = (128, 160)
SCHEDULES = ("wave", "wave_jag")


def variant_id(arm, penalty):
    if (arm, penalty) not in VARIANTS:
        raise ValueError("unregistered variant")
    return arm + "-margin" + str(penalty).replace(".", "p")


def score(world, style, arm, model, penalty):
    variant_id(arm, penalty)
    menu, x = d.features(world, style, arm)
    raw = x @ model.w
    logits = raw.copy()
    # Penalize STOP only when genuinely host-legal MOVE alternatives exist.
    # Do not remove any action, and never fabricate forward movement.
    if "STOP" in menu and any(a in MOVE for a in menu):
        logits[menu.index("STOP")] -= penalty
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    choice = menu[int(np.argmax(p))]
    original_choice = menu[int(np.argmax(raw))]
    return menu, p, choice, original_choice


def episode(spec, style, arm, penalty, model, horizon, forced):
    if spec.split != "development" or horizon not in HORIZONS_EVAL or style not in SCHEDULES:
        raise ValueError("unapproved evaluation fixture")
    identity = variant_id(arm, penalty)
    world = World.create(spec)
    decisions, forces, times = [], [], []
    reverted_stop = 0
    matches = []
    for turn in range(horizon):
        if world.stopped:
            break
        active = d.current_style(style, turn)
        t0 = time.perf_counter_ns()
        menu, p, chosen, raw_choice = score(world, active, arm, model, penalty)
        times.append((time.perf_counter_ns() - t0) / 1e6)
        reference = d.teacher(world, active)
        actual, injection = d.intervene(world, chosen, turn, model.seed, forced)
        if raw_choice == "STOP" and chosen != "STOP" and any(a in MOVE for a in menu):
            reverted_stop += 1
        matches.append(chosen == reference)
        decisions.append({"turn": turn, "style": active, "chosen": chosen,
                          "raw_choice": raw_choice, "teacher": reference,
                          "p": {a: round(float(v), 8) for a, v in zip(menu, p)}})
        if injection:
            forces.append([turn, injection])
        world.apply(actual, MAX_RECOVERIES)
    if replay(spec, world.decisions, MAX_RECOVERIES).state_sha() != world.state_sha():
        raise ValueError("replay mismatch")
    return {"version": VERSION, "split": "development", "world_id": spec.ident,
            "style": style, "variant": identity, "arm": arm, "margin": penalty,
            "model_seed": model.seed, "horizon": horizon, "forced": forced,
            "base_sha": world.base_sha(), "events": world.decisions,
            "decisions": decisions, "forces": forces,
            "stop_overrides": reverted_stop,
            "teacher_match": round(statistics.mean(matches), 6) if matches else None,
            "metrics": world.metrics(horizon), "final_sha": world.state_sha(),
            "trajectory_sha": digest(world.decisions),
            "action_distribution": dict(sorted(Counter(e["chosen"] for e in world.decisions).items())),
            "mean_latency_ms": round(statistics.mean(times), 6) if times else None,
            "native_receipts": 0}


def verify_episode(e, model):
    if e["version"] != VERSION or e["split"] != "development" or e["native_receipts"] != 0:
        raise ValueError("provenance mismatch")
    if e["model_seed"] != model.seed or e["variant"] != variant_id(e["arm"], e["margin"]):
        raise ValueError("wrong model/variant")
    matches = [s for s in specs("development", 1) if s.ident == e["world_id"]]
    if len(matches) != 1 or e["horizon"] not in HORIZONS_EVAL or e["style"] not in SCHEDULES:
        raise ValueError("unapproved world")
    spec = matches[0]
    w = World.create(spec)
    if e["base_sha"] != w.base_sha() or e["trajectory_sha"] != digest(e["events"]):
        raise ValueError("source mismatch")
    if len(e["events"]) != len(e["decisions"]):
        raise ValueError("truncated path")
    forces, matches_teacher, overrides = [], [], 0
    for turn, (record, host_record) in enumerate(zip(e["decisions"], e["events"])):
        if w.stopped or turn >= e["horizon"]:
            raise ValueError("post-STOP transcript")
        active = d.current_style(e["style"], turn)
        menu, p, chosen, raw = score(w, active, e["arm"], model, e["margin"])
        ref = d.teacher(w, active)
        if (record["turn"] != turn or record["style"] != active or
            record["chosen"] != chosen or record["raw_choice"] != raw or
            record["teacher"] != ref or set(record["p"]) != set(menu) or
            any(abs(record["p"][a] - float(v)) > 2e-8 for a,v in zip(menu,p))):
            raise ValueError("probability/choice tampering")
        actual, injection = d.intervene(w, chosen, turn, model.seed, e["forced"])
        if injection:
            forces.append([turn, injection])
        if raw == "STOP" and chosen != "STOP" and any(a in MOVE for a in menu):
            overrides += 1
        matches_teacher.append(chosen == ref)
        if actual != host_record["chosen"] or w.apply(actual, MAX_RECOVERIES) != host_record:
            raise ValueError("host transition tampering")
    agreement = round(statistics.mean(matches_teacher), 6) if matches_teacher else None
    if (e["forces"] != forces or e["stop_overrides"] != overrides or
        e["teacher_match"] != agreement or e["metrics"] != w.metrics(e["horizon"]) or
        e["final_sha"] != w.state_sha() or
        e["action_distribution"] != dict(sorted(Counter(x["chosen"] for x in w.decisions).items()))):
        raise ValueError("geometry/metric tampering")
    return len(w.decisions)


def run(output, seed=584, updates=30):
    output = Path(output)
    if output.exists():
        raise FileExistsError("original artifacts are immutable")
    if seed not in (584, 585) or not 1 <= updates <= 50:
        raise ValueError("bounded experiment only")
    output.mkdir(parents=True)
    begun = time.perf_counter()
    models = {a: d.Adapter(seed) for a in d.ARMS}
    training = []
    for stage, horizon in enumerate(HORIZONS):
        xs, masks, ys, corpus = d.dataset(seed, 1, horizon)
        for arm, model in models.items():
            loss = model.train(xs[arm], masks, ys, updates, seed+stage*1009)
            training.append({"arm": arm, "stage": horizon, "corpus_sha": corpus,
                             "examples": len(ys), **loss})
    checkpoints = {}
    for arm, model in models.items():
        path = output / (arm + ".npz")
        model.save(path)
        check = d.Adapter.load(path)
        if check.updates != len(HORIZONS)*updates or not np.array_equal(check.w, model.w):
            raise ValueError("checkpoint roundtrip")
        checkpoints[arm] = {"sha256": sha_file(path),
                            "updates": check.updates, "params": len(check.w)}
    records = []
    for arm, penalty in VARIANTS:
        model = models[arm]
        for spec in specs("development", 1):
            for style in SCHEDULES:
                for horizon in HORIZONS_EVAL:
                    for forced in (False, True):
                        e = episode(spec, style, arm, penalty, model, horizon, forced)
                        verify_episode(e, model)
                        name = (variant_id(arm, penalty) + "-" + spec.ident + "-" +
                                style + "-" + str(horizon) + "-f" + str(int(forced)))
                        (output / (name+".json")).write_text(json.dumps(e, sort_keys=True))
                        if spec == specs("development", 1)[0] and horizon == 160 and forced:
                            _png(output / (name+".png"), replay(spec,e["events"],MAX_RECOVERIES))
                        records.append(e)
    summary = {}
    for arm, penalty in VARIANTS:
        for horizon in HORIZONS_EVAL:
            for forced in (False, True):
                rows = [e for e in records if (e["arm"],e["margin"],e["horizon"],e["forced"]) ==
                        (arm,penalty,horizon,forced)]
                summary[variant_id(arm,penalty)+"/"+str(horizon)+"/f"+str(int(forced))] = {
                    "episodes": len(rows),
                    "mean_ink": round(statistics.mean(e["metrics"]["unique_painted_cells"] for e in rows),3),
                    "mean_branches": round(statistics.mean(e["metrics"]["branch_cells"] for e in rows),3),
                    "early_stops": sum(e["metrics"]["stopped_early"] for e in rows),
                    "stop_overrides": sum(e["stop_overrides"] for e in rows),
                    "mean_teacher_match": round(statistics.mean(e["teacher_match"] for e in rows if e["teacher_match"] is not None),4),
                    "action_distribution": dict(sorted(sum((Counter(e["action_distribution"]) for e in rows),Counter()).items())),
                    "mean_latency_ms": round(statistics.mean(e["mean_latency_ms"] for e in rows),5)}
    pins = {**source_pins(), "dual_clock_lab.py": sha_file(Path(d.__file__)),
            "stop_calibration_lab.py": sha_file(Path(__file__))}
    report = {"version": VERSION, "seed": seed, "training": training,
              "checkpoints": checkpoints, "summary": summary,
              "source_sha256": pins, "episodes": len(records),
              "verified_decisions": sum(len(e["events"]) for e in records),
              "elapsed_s": round(time.perf_counter()-begun,3),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "structural_heldout_sealed": True, "native_receipts": 0,
              "native_logical_move_ceiling": 64,
              "note": "Margins fixed before DEV evaluation; STOP remains legal; no native success claim."}
    (output/"report.json").write_text(json.dumps(report, sort_keys=True, indent=2))
    (output/"SHA256SUMS.json").write_text(json.dumps({
        p.name: sha_file(p) for p in output.iterdir()},sort_keys=True,indent=2))
    return report


def verify(output):
    output = Path(output)
    report = json.loads((output/"report.json").read_text())
    pins = {**source_pins(), "dual_clock_lab.py": sha_file(Path(d.__file__)),
            "stop_calibration_lab.py": sha_file(Path(__file__))}
    if (report["version"] != VERSION or report["source_sha256"] != pins or
        not report["structural_heldout_sealed"] or report["native_receipts"] != 0 or
        report["native_logical_move_ceiling"] != 64):
        raise ValueError("source/provenance drift")
    manifest = json.loads((output/"SHA256SUMS.json").read_text())
    actual = {p.name:sha_file(p) for p in output.iterdir() if p.name != "SHA256SUMS.json"}
    if actual != manifest:
        raise ValueError("mutated artifact")
    models = {}
    for arm in d.ARMS:
        model = d.Adapter.load(output/(arm+".npz"))
        if (sha_file(output/(arm+".npz")) != report["checkpoints"][arm]["sha256"] or
            model.updates != report["checkpoints"][arm]["updates"]):
            raise ValueError("checkpoint provenance")
        models[arm] = model
    count = turns = 0
    for file in output.glob("*.json"):
        if file.name in ("report.json","SHA256SUMS.json"):
            continue
        e = json.loads(file.read_text())
        turns += verify_episode(e,models[e["arm"]])
        count += 1
    if (count,turns) != (report["episodes"],report["verified_decisions"]):
        raise ValueError("truncated experiment")
    return {"episodes":count,"verified_decisions":turns,"manifest_files":len(actual)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--verify", type=Path)
    p.add_argument("--seed", type=int, default=584)
    p.add_argument("--updates", type=int, default=30)
    a=p.parse_args()
    if bool(a.output)==bool(a.verify):
        p.error("pass exactly one of --output or --verify")
    print(json.dumps(verify(a.verify) if a.verify else run(a.output,a.seed,a.updates),sort_keys=True))


if __name__=="__main__":
    main()
