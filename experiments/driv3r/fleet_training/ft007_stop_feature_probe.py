"""FT-007: counterfactual early-feature sensitivity in frozen real CPU policy weights.

New worlds are diagnostic only, never TRAIN. The pinned host supplies all legal
actions; this experiment never edits the host, checkpoints or the actual game.
Counterfactual feature values hold the entire synthetic world/host menu fixed:
they are not real trajectories, while archived autonomous policy rollouts are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import fleet_bench as base
import ft004_shift as transfer
import ft005_curriculum as ft005
import ft006_stop_ablation as ft006

ID = "FT-007-20261116-frozen-stop-feature-sensitivity"
SCHEMA = "cybersand.fleet-training.stop-feature-probe-request/1"
SEEDS = (9401, 9403)
DOSES = (12, 48)
SCENARIOS = ("fresh30", "long60", "dense60", "shifted_guard60")
PROBE_TURNS = (0, 8, 16, 24, 48)
EARLY_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PROBE_WORLDS = 3
EVAL_WORLDS = 1
MAX_SECONDS = 240
MAX_RSS_KIB = 900000


def check_request(request):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": transfer.SOURCE_COMMIT,
        "source_blob": transfer.SOURCE_BLOB,
        "ft003_actions_run": 43797061143,
        "ft003_artifact": 14194214028,
        "ft003_archive_sha256": transfer.FT003_SHA,
        "training_seeds": list(SEEDS), "frozen_doses": list(DOSES),
        "scenarios": list(SCENARIOS),
        "probe_turns": list(PROBE_TURNS),
        "early_grid": list(EARLY_GRID),
        "probe_worlds_per_scenario": PROBE_WORLDS,
        "autonomous_worlds_per_scenario": EVAL_WORLDS,
        "max_seconds": MAX_SECONDS, "max_rss_kib": MAX_RSS_KIB,
        "max_runner_jobs": 1, "optimizer_steps": 0,
        "native_material_edits": 0,
    }
    if type(request) is not dict or request != expected:
        raise ValueError("FT-007 immutable input, model, custody or budget mismatch")
    return request


def world_seed(pop, domain, seed, niche, scenario, index):
    if (domain not in ("PROBE", "AUTONOMOUS") or
        seed not in SEEDS or niche not in base.NICHES or
        scenario not in SCENARIOS or type(index) is not int or
        index < 0 or index >= (PROBE_WORLDS if domain == "PROBE" else EVAL_WORLDS)):
        raise ValueError("Invalid new-world domain, lease or index")
    return pop.stable_seed(ID, domain, seed, niche, scenario, index)


def custody_digest(pop):
    previous = set()
    for seed in SEEDS:
        for split, n in (("train", 48), ("selection", 4), ("test", 12)):
            previous.update(pop.stable_seed(seed, split, i) for i in range(n))
        previous.update(transfer.heldout_world_seed(pop, seed, i)
                        for i in range(transfer.WORLDS))
        for niche in base.NICHES:
            for domain, scenarios, count, name in (
                ("TRAIN", ft005.TRAIN_SCENARIOS, ft005.TRAIN_WORLDS_PER_SCENARIO, ft005.ID),
                ("TEST", ft005.EVAL_SCENARIOS, ft005.TEST_WORLDS, ft005.ID),
                ("TRAIN", ft006.TRAIN_SCENARIOS, ft006.TRAIN_WORLDS, ft006.ID),
                ("TEST", ft006.TEST_SCENARIOS, ft006.TEST_WORLDS, ft006.ID),
            ):
                previous.update(pop.stable_seed(name, domain, seed, niche, scen, i)
                                for scen in scenarios for i in range(count))
    generated = set()
    for seed in SEEDS:
        for niche in base.NICHES:
            for scenario in SCENARIOS:
                for domain, count in (("PROBE", PROBE_WORLDS), ("AUTONOMOUS", EVAL_WORLDS)):
                    for i in range(count):
                        val = world_seed(pop, domain, seed, niche, scenario, i)
                        if val in generated or val in previous:
                            raise AssertionError("Diagnostic custody collision")
                        generated.add(val)
    return hashlib.sha256(json.dumps(sorted(generated)).encode()).hexdigest()


def teacher_action(pop, world, seed):
    """Deterministic legal scaffolding for feature probing, NOT a learned policy."""
    menu = world.legal()
    if "RETRACE" in menu and world.steps % 9 == 8:
        return "RETRACE"
    moves = [action for action in menu if action in pop.DIRECTIONS]
    if moves:
        index = pop.stable_seed(ID, "scaffold", seed, world.steps) % len(moves)
        return moves[index]
    return "RETRACE" if "RETRACE" in menu else "STOP"


def diagnostic_states(pop, seed, scenario):
    world = transfer.world_for(pop, seed, scenario)
    samples, actions = [], []
    while not world.stopped and world.steps < world.max_steps:
        if world.steps in PROBE_TURNS:
            menu = world.legal()
            X = np.stack([world.feature(a) for a in menu])
            samples.append((world.steps, menu, X, transfer.state_digest(world)))
        action = teacher_action(pop, world, seed)
        if action not in world.legal():
            raise AssertionError("Scaffold escaped host-authorized legal actions")
        world.step(action)
        actions.append(action)
        if world.painted & (world.protected | world.obstacles):
            raise AssertionError("Scaffold painted protected or blocked material")
    return samples, actions


def score_state(policy, features, menu, tick, state_hash, world_id, early_index):
    """Save exact real checkpoint scores; feature interventions are not rollouts."""
    if len(menu) < 2:
        return None
    if "STOP" not in menu or features.shape[0] != len(menu):
        raise AssertionError("Missing host STOP or malformed feature matrix")
    ref = np.asarray(policy.logits(features), dtype=float)
    if not np.isfinite(ref).all():
        raise ValueError("Nonfinite frozen policy reference scores")
    stop = menu.index("STOP")
    alternatives = [i for i, a in enumerate(menu) if a != "STOP"]
    grid, margins, actions = [], [], []
    for early in EARLY_GRID:
        X = features.copy()
        X[:, early_index] = early
        scores = np.asarray(policy.logits(X), dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite counterfactual scores")
        choice = menu[int(np.argmax(scores))]
        margin = float(scores[stop] - max(scores[i] for i in alternatives))
        grid.append({"early": early, "scores": [float(x) for x in scores],
                     "choice": choice, "stop_margin": margin})
        margins.append(margin)
        actions.append(choice)
    original_margin = float(ref[stop] - max(ref[i] for i in alternatives))
    return {
        "world_seed": world_id, "tick": tick, "state_sha256": state_hash,
        "host_legal_menu": menu, "original_features": features.tolist(),
        "original_early": float(features[0, early_index]),
        "original_scores": [float(x) for x in ref],
        "original_choice": menu[int(np.argmax(ref))],
        "original_stop_margin": original_margin,
        "grid": grid, "stop_rank_flip": min(margins) <= 0 < max(margins),
        "any_action_flip": len(set(actions)) > 1,
        "stop_margin_span": max(margins) - min(margins),
    }


def check_budget(start):
    if time.monotonic() - start > MAX_SECONDS:
        raise TimeoutError("FT-007 bounded frozen inference wall exceeded")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
        raise MemoryError("FT-007 frozen inference RSS exceeded")


def run(request_file, upstream, ft003, output):
    check_request(json.loads(request_file.read_text()))
    pop, source = transfer.load_source(upstream)
    if "early" not in pop.FEATURES:
        raise ValueError("Pinned source lacks time feature")
    early_index = pop.FEATURES.index("early")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "cybersand.fleet-training.stop-feature-probe-evidence/1",
        "request_id": ID, "status": "starting",
        "class": "REAL_FROZEN_NUMPY_POLICY_SYNTHETIC_NOT_NATIVE_OR_W0",
        "source_commit": transfer.SOURCE_COMMIT,
        "source_blob": transfer.SOURCE_BLOB,
        "source_sha256": base.sha_file(source),
        "executor_sha256": base.sha_file(Path(__file__)),
        "request_sha256": base.sha_file(request_file),
        "ft003_archive_sha256": transfer.FT003_SHA,
        "optimizer_steps": 0, "native_material_edits": 0,
        "resident_model_processes": 1,
        "leases": [], "failure": None
    }
    try:
        report["original_audit"] = transfer.archive_check(
            ft003, source, Path(__file__).parent / "request_dose.json")
        report["custody_sha256"] = custody_digest(pop)
        report["status"] = "running"
        seen = set()
        for seed in SEEDS:
            for dose in DOSES:
                for niche in base.NICHES:
                    checkpoint = ft003 / f"dose_{dose}/{seed}/models/{niche}.json"
                    policy = transfer.load_policy(pop, checkpoint)
                    lease = f"{ID}/{seed}/{dose}/{niche}"
                    if lease in seen:
                        raise AssertionError("Duplicate inference lease")
                    seen.add(lease)
                    panels = []
                    true_actions = true_ink = 0
                    probes = 0
                    for scenario in SCENARIOS:
                        for i in range(PROBE_WORLDS):
                            probe_seed = world_seed(pop, "PROBE", seed, niche, scenario, i)
                            states, _ = diagnostic_states(pop, probe_seed, scenario)
                            for tick, menu, X, state_hash in states:
                                record = score_state(policy, X, menu, tick, state_hash,
                                                     probe_seed, early_index)
                                if record is not None:
                                    panels.append({"scenario": scenario, **record})
                                    probes += 1
                            check_budget(started)
                        for i in range(EVAL_WORLDS):
                            eval_seed = world_seed(pop, "AUTONOMOUS", seed, niche, scenario, i)
                            receipt, world = transfer.execute(pop, policy, niche, eval_seed,
                                                              scenario, save_trace=True)
                            transfer.audit_receipts(pop, receipt["receipts"], niche,
                                                    eval_seed, scenario)
                            if receipt["model_forwards"] != receipt["decisions"]:
                                raise AssertionError("Original inference lacks model forwards")
                            outdir = output / "gallery" / str(seed) / str(dose) / niche
                            outdir.mkdir(parents=True, exist_ok=True)
                            (outdir / (scenario + ".svg")).write_text(pop.draw_svg(world))
                            (outdir / (scenario + ".json")).write_text(
                                json.dumps(receipt, sort_keys=True) + "\n")
                            true_actions += receipt["model_forwards"]
                            true_ink += receipt["new_ink"]
                    panelpath = output / "panels" / str(seed) / str(dose) / (niche + ".json")
                    panelpath.parent.mkdir(parents=True, exist_ok=True)
                    panelpath.write_text(json.dumps(panels, sort_keys=True) + "\n")
                    report["leases"].append({
                        "lease_id": lease, "status": "completed",
                        "checkpoint_sha256": base.sha_file(checkpoint),
                        "training_seed": seed, "dose": dose, "niche": niche,
                        "probe_states": probes,
                        "counterfactual_score_batches": len(EARLY_GRID) * probes,
                        "stop_rank_flips": sum(p["stop_rank_flip"] for p in panels),
                        "any_action_flips": sum(p["any_action_flip"] for p in panels),
                        "mean_stop_margin_span": (sum(p["stop_margin_span"] for p in panels)
                                                  / len(panels) if panels else None),
                        "autonomous_model_choices": true_actions,
                        "autonomous_synthetic_new_ink": true_ink,
                        "native_material_edits": 0
                    })
        leases = report["leases"]
        report["aggregate"] = {
            "frozen_checkpoint_leases": len(leases),
            "diagnostic_states": sum(p["probe_states"] for p in leases),
            "counterfactual_score_batches": sum(p["counterfactual_score_batches"]
                                               for p in leases),
            "stop_rank_flips": sum(p["stop_rank_flips"] for p in leases),
            "any_action_flips": sum(p["any_action_flips"] for p in leases),
            "genuine_autonomous_model_choices": sum(p["autonomous_model_choices"]
                                                   for p in leases),
            "synthetic_new_ink": sum(p["autonomous_synthetic_new_ink"] for p in leases),
            "original_autonomous_svg_count": len(list((output / "gallery").rglob("*.svg"))),
            "model_training_updates": 0,
            "wall_s": round(time.monotonic() - started, 5),
            "rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        }
        if len(leases) != len(SEEDS) * len(DOSES) * len(base.NICHES):
            raise AssertionError("Missing source-pinned model lease")
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2,
                                                         sort_keys=True) + "\n")
        (output / "SHA256SUMS").write_text("".join(
            base.sha_file(p) + "  " + str(p.relative_to(output)) + "\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--ft003", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    result = run(args.request, args.upstream, args.ft003, args.output)
    print(json.dumps({"status": result["status"],
                      "aggregate": result["aggregate"]}, sort_keys=True))
