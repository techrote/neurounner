"""FT-006: equal-exposure early STOP-target ablation for real tiny-policy Adam.

Twelve source-pinned seed/niche checkpoints produce matched plain-target and
early-STOP-penalized continuations. Both arms see the SAME deterministic,
host-legal synthetic teacher trajectory and have equal optimizer update counts.
Independent held-out rollouts permit autonomous STOP. Neither arm edits Godot.
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
import ft005_curriculum as previous

SCHEMA = "cybersand.fleet-training.stop-ablation-request/1"
ID = "FT-006-20261116-equal-exposure-stop-ablation"
SEEDS = (9401, 9403)
TRAIN_SCENARIOS = ("long60", "dense60", "shifted_guard60")
TEST_SCENARIOS = ("fresh30", "long60", "dense60", "shifted_guard60")
TRAIN_WORLDS = 8
TEST_WORLDS = 8
EARLY_STOP_HORIZON = 24
EARLY_STOP_LOGIT_PENALTY = 2.0
TARGET_TEMP = 0.42
MAX_SECONDS = 360
MAX_RSS_KIB = 900000
ARMS = ("plain_teacher", "stop_deferred_teacher")


def checked_request(data):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "population_source_commit": transfer.SOURCE_COMMIT,
        "population_source_blob": transfer.SOURCE_BLOB,
        "ft003_run_id": 43797061143, "ft003_artifact_id": 14194214028,
        "ft003_zip_sha256": transfer.FT003_SHA,
        "train_seeds": list(SEEDS), "frozen_dose": 48,
        "training_scenarios": list(TRAIN_SCENARIOS),
        "test_scenarios": list(TEST_SCENARIOS),
        "training_worlds_per_scenario": TRAIN_WORLDS,
        "test_worlds_per_scenario": TEST_WORLDS,
        "early_stop_horizon": EARLY_STOP_HORIZON,
        "early_stop_logit_penalty": EARLY_STOP_LOGIT_PENALTY,
        "target_temperature": TARGET_TEMP,
        "max_seconds_per_seed": MAX_SECONDS,
        "max_rss_kib_per_seed": MAX_RSS_KIB,
        "max_parallel_runners": 2,
        "native_material_edits": 0,
        "pretrained_slm_optimizer_steps": 0
    }
    if type(data) is not dict or data != expected:
        raise ValueError("FT-006 immutable request/source/custody/budget mismatch")
    return data


def lease(seed, niche, arm):
    if seed not in SEEDS or niche not in base.NICHES or arm not in ARMS:
        raise ValueError("Unregistered optimizer lease")
    return f"{ID}/{seed}/{niche}/{arm}"


def world_seed(pop, domain, seed, niche, scenario, index):
    count = TRAIN_WORLDS if domain == "TRAIN" else TEST_WORLDS
    scenarios = TRAIN_SCENARIOS if domain == "TRAIN" else TEST_SCENARIOS
    if (domain not in ("TRAIN", "TEST") or seed not in SEEDS
        or niche not in base.NICHES or scenario not in scenarios
        or type(index) is not int or index < 0 or index >= count):
        raise ValueError("Unregistered world seed/domain")
    value = pop.stable_seed(ID, domain, seed, niche, scenario, index)
    upstream = {pop.stable_seed(seed, split, i)
                for split, n in (("train", 48), ("selection", 4), ("test", 12))
                for i in range(n)}
    original_ood = {transfer.heldout_world_seed(pop, seed, i)
                    for i in range(transfer.WORLDS)}
    prior = {previous.domain_seed(pop, olddomain, seed, niche, oldscenario, i)
             for olddomain, modes, n in (
                 ("TRAIN", previous.TRAIN_SCENARIOS, previous.TRAIN_WORLDS_PER_SCENARIO),
                 ("TEST", previous.EVAL_SCENARIOS, previous.TEST_WORLDS))
             for oldscenario in modes for i in range(n)}
    if value in upstream or value in original_ood or value in prior:
        raise AssertionError("FT-006 overlaps previous train/selection/test custody")
    return value


def custody_digest(pop):
    all_seeds = []
    for seed in SEEDS:
        for niche in base.NICHES:
            for domain, scenarios, n in (
                ("TRAIN", TRAIN_SCENARIOS, TRAIN_WORLDS),
                ("TEST", TEST_SCENARIOS, TEST_WORLDS)):
                for scenario in scenarios:
                    for i in range(n):
                        all_seeds.append(world_seed(pop, domain, seed, niche, scenario, i))
    if len(set(all_seeds)) != len(all_seeds):
        raise AssertionError("FT-006 train/test/world cross-lease collision")
    return hashlib.sha256(json.dumps(sorted(all_seeds)).encode()).hexdigest()


def reward_targets(pop, niche, world, menu, features, penalize_stop):
    utility = np.asarray([pop.utility(niche, x, world) for x in features],
                         dtype=np.float64)
    if (penalize_stop and world.steps < EARLY_STOP_HORIZON
        and "STOP" in menu and any(a != "STOP" for a in menu)):
        utility[menu.index("STOP")] -= EARLY_STOP_LOGIT_PENALTY
    return pop.softmax(utility / TARGET_TEMP)


def teacher_choice(pop, world, menu, rng):
    """Fixed policy-free legal exposure, identical for both learning arms.

    In TRAIN only, defer STOP while an alternative legal action exists.
    Evaluation always uses the model's complete legal menu, including STOP.
    """
    non_stop = [i for i, a in enumerate(menu) if a != "STOP"]
    if not non_stop:
        return menu.index("STOP")
    if "RETRACE" in menu and rng.random() < 0.20:
        return menu.index("RETRACE")
    directions = [i for i, a in enumerate(menu) if a in pop.DIRECTIONS]
    if directions:
        return int(rng.choice(directions))
    return int(rng.choice(non_stop))


def budget(began):
    if time.monotonic() - began > MAX_SECONDS:
        raise TimeoutError("FT-006 per-runner training/eval wall cap")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
        raise MemoryError("FT-006 process RSS cap")


def train_equal_exposure(pop, policies, niche, seed, began):
    receipts = []
    initial = {arm: policy.updates for arm, policy in policies.items()}
    for scenario in TRAIN_SCENARIOS:
        for i in range(TRAIN_WORLDS):
            wseed = world_seed(pop, "TRAIN", seed, niche, scenario, i)
            world = transfer.world_for(pop, wseed, scenario)
            rng = np.random.default_rng(pop.stable_seed(ID, "teacher", seed, niche, scenario, i))
            steps = 0
            retraces = 0
            while not world.stopped and world.steps < world.max_steps:
                menu = world.legal()
                X = np.stack([world.feature(action) for action in menu])
                for arm, policy in policies.items():
                    policy.teach(X, reward_targets(pop, niche, world, menu, X,
                                                  penalize_stop=(arm == "stop_deferred_teacher")))
                index = teacher_choice(pop, world, menu, rng)
                action = menu[index]
                if action not in world.legal():
                    raise AssertionError("Curriculum violated host legal menu")
                world.step(action)
                steps += 1
                retraces += int(action == "RETRACE")
                if world.painted & (world.protected | world.obstacles):
                    raise AssertionError("Protected/obstacle material changed during training")
                budget(began)
            receipts.append({
                "scenario": scenario, "training_world_seed": wseed,
                "teacher_decisions": steps, "teacher_retraces": retraces,
                "teacher_ink": len(world.painted) - 1,
                "teacher_early_stop": bool(world.stopped and world.steps < world.max_steps)
            })
    deltas = {arm: policy.updates - initial[arm] for arm, policy in policies.items()}
    if min(deltas.values()) <= 0 or len(set(deltas.values())) != 1:
        raise AssertionError("No genuine matched equal-count Adam optimizer updates")
    return receipts, deltas


def evaluate(pop, policies, niche, seed, output, began):
    rows = []
    for scenario in TEST_SCENARIOS:
        seeds = [world_seed(pop, "TEST", seed, niche, scenario, i)
                 for i in range(TEST_WORLDS)]
        summaries = {}
        for arm, policy in policies.items():
            episodes = []
            for i, wseed in enumerate(seeds):
                item, world = transfer.execute(pop, policy, niche, wseed, scenario, save_trace=(i == 0))
                if item["model_forwards"] != item["decisions"] or item["protected_violations"]:
                    raise AssertionError("Invalid model-score or host-material receipt")
                if i == 0:
                    transfer.audit_receipts(pop, item["receipts"], niche, wseed, scenario)
                    folder = output / "gallery" / niche / scenario
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / f"{arm}.svg").write_text(pop.draw_svg(world))
                    (folder / f"{arm}.json").write_text(json.dumps(item, sort_keys=True) + "\n")
                episodes.append(item)
                budget(began)
            summaries[arm] = transfer.summarize(episodes)
        plain = summaries["plain_teacher"]
        ablated = summaries["stop_deferred_teacher"]
        rows.append({
            "niche": niche, "scenario": scenario, "test_world_seeds": seeds,
            "plain_teacher": plain, "stop_deferred_teacher": ablated,
            "reward_per_decision_delta": round(
                ablated["reward_per_decision"] - plain["reward_per_decision"], 5),
            "decisions_delta": round(ablated["decisions"] - plain["decisions"], 5),
            "early_stop_delta": round(ablated["early_stop"] - plain["early_stop"], 5),
            "new_ink_delta": round(ablated["new_ink"] - plain["new_ink"], 5),
            "retraces_delta": round(ablated["retraces"] - plain["retraces"], 5)
        })
    return rows


def run(request, upstream, ft003, output, seed):
    checked_request(json.loads(request.read_text()))
    if seed not in SEEDS:
        raise ValueError("Unknown FT-006 seed lease")
    pop, original_source = transfer.load_source(upstream)
    source_request = Path(__file__).parent / "request_dose.json"
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = {
        "schema": "cybersand.fleet-training.stop-ablation-evidence/1",
        "request_id": ID, "training_seed": seed, "status": "starting",
        "source_commit": transfer.SOURCE_COMMIT, "source_blob": transfer.SOURCE_BLOB,
        "source_sha256": base.sha_file(original_source),
        "executor_sha256": base.sha_file(Path(__file__)),
        "request_sha256": base.sha_file(request),
        "original_ft003_archive_sha256": transfer.FT003_SHA,
        "class": "REAL_CPU_ADAM_EQUAL_EXPOSURE_SYNTHETIC_NOT_SLM_NATIVE_OR_W0",
        "native_edits": 0, "pretrained_slm_optimizer_steps": 0,
        "workers": [], "evaluations": [], "failures": [],
        "actual_resident_processes_per_job": 1,
        "max_parallel_runner_jobs": 2
    }
    try:
        report["original_audit"] = transfer.archive_check(ft003, original_source, source_request)
        report["world_custody_sha256"] = custody_digest(pop)
        report["status"] = "running"
        for niche in base.NICHES:
            source = ft003 / f"dose_48/{seed}/models/{niche}.json"
            policies = {arm: transfer.load_policy(pop, source) for arm in ARMS}
            exposure, updates = train_equal_exposure(pop, policies, niche, seed, began)
            for arm, policy in policies.items():
                dst = output / "models" / arm / f"{niche}.json"
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(policy.serialized(), sort_keys=True) + "\n")
                if base.sha_file(dst) == base.sha_file(source):
                    raise AssertionError("Optimizer produced no changed model")
                report["workers"].append({
                    "lease_id": lease(seed, niche, arm), "status": "completed",
                    "model": "population-mlp-24", "checkpoint_source_sha256": base.sha_file(source),
                    "trained_checkpoint_sha256": base.sha_file(dst),
                    "new_optimizer_updates": updates[arm],
                    "teacher_exposure_worlds": len(exposure),
                    "teacher_exposure_digest": hashlib.sha256(
                        json.dumps(exposure, sort_keys=True).encode()).hexdigest(),
                    "teacher_early_stops": sum(x["teacher_early_stop"] for x in exposure),
                    "teacher_retraces": sum(x["teacher_retraces"] for x in exposure)
                })
            report["evaluations"].extend(evaluate(pop, policies, niche, seed, output, began))
        rows = report["evaluations"]
        n = len(rows)
        report["aggregate"] = {
            "paired_niche_scenarios": n,
            "reward_wins": sum(x["reward_per_decision_delta"] > 0 for x in rows),
            "reward_regressions": sum(x["reward_per_decision_delta"] < 0 for x in rows),
            "reward_ties": sum(x["reward_per_decision_delta"] == 0 for x in rows),
            "mean_reward_per_decision_delta": round(
                sum(x["reward_per_decision_delta"] for x in rows) / n, 5),
            "mean_early_stop_delta": round(sum(x["early_stop_delta"] for x in rows) / n, 5),
            "mean_decisions_delta": round(sum(x["decisions_delta"] for x in rows) / n, 5),
            "mean_new_ink_delta": round(sum(x["new_ink_delta"] for x in rows) / n, 5),
            "mean_retraces_delta": round(sum(x["retraces_delta"] for x in rows) / n, 5),
            "total_real_adam_updates": sum(x["new_optimizer_updates"] for x in report["workers"]),
            "synthetic_model_scored_episodes": n * len(ARMS) * TEST_WORLDS,
            "gallery_svgs": len(list((output / "gallery").rglob("*.svg"))),
            "wall_s": round(time.monotonic() - began, 5),
            "peak_worker_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "native_godot": False, "actual_real_adam": True
        }
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["failures"].append(type(error).__name__ + ": " + str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        (output / "SHA256SUMS").write_text("".join(
            base.sha_file(p) + "  " + str(p.relative_to(output)) + "\n"
            for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--ft003", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    r = run(args.request, args.upstream, args.ft003, args.output, args.seed)
    print(json.dumps({"status": r["status"], "seed": args.seed,
                      "aggregate": r["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
