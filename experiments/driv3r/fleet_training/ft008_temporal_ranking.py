"""FT-008: matched counterfactual STOP-ranking curriculum on real tiny Adam policies.

Two arms continue the same immutable FT-003 checkpoint with precisely matched
host-legal teacher states, gradient-update counts and synthetic worlds. The
temporal intervention (overriding the 'early' FEATURE) is explicitly synthetic,
not a physically realized state. Separately evaluate autonomous unmodified
worlds with legal STOP still available and persist failures/negative outcomes.

Research only: no Godot, native PAINT, source-faithful W0 or pretrained SLM.
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
import ft007_stop_feature_probe as ft007

ID = "FT-008-20261116-matched-temporal-stop-ranking"
SCHEMA = "cybersand.fleet-training.temporal-ranking-request/1"
SEEDS = (9401, 9403)
TRAIN_SCENARIOS = ("long60", "dense60", "shifted_guard60")
EVAL_SCENARIOS = ("fresh30", "long60", "dense60", "shifted_guard60")
DIAG_SCENARIOS = ("long60", "shifted_guard60")
TRAIN_WORLDS = 6
EVAL_WORLDS = 4
DIAG_WORLDS = 1
ARMS = ("sham_counterfactual", "temporal_stop_ranking")
EARLY_PENALTY = 3.0
LATE_BONUS = 4.0
LATE_THRESHOLD = .80
TARGET_TEMPERATURE = .42
MAX_SECONDS = 300
MAX_RSS_KIB = 900000


def validate_request(r):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": transfer.SOURCE_COMMIT,
        "source_blob": transfer.SOURCE_BLOB,
        "ft003_run_id": 43797061143,
        "ft003_artifact_id": 14194214028,
        "ft003_zip_sha256": transfer.FT003_SHA,
        "seeds": list(SEEDS), "frozen_dose": 48,
        "train_scenarios": list(TRAIN_SCENARIOS),
        "eval_scenarios": list(EVAL_SCENARIOS),
        "diagnostic_scenarios": list(DIAG_SCENARIOS),
        "train_worlds_per_scenario": TRAIN_WORLDS,
        "eval_worlds_per_scenario": EVAL_WORLDS,
        "diagnostic_worlds_per_scenario": DIAG_WORLDS,
        "arms": list(ARMS),
        "early_stop_penalty": EARLY_PENALTY,
        "late_stop_bonus": LATE_BONUS,
        "late_fraction_threshold": LATE_THRESHOLD,
        "target_temperature": TARGET_TEMPERATURE,
        "max_seconds_per_seed": MAX_SECONDS,
        "max_rss_kib": MAX_RSS_KIB,
        "max_parallel_runner_jobs": 2,
        "native_edits": 0, "pretrained_slm_optimizer_steps": 0
    }
    if type(r) is not dict or r != expected:
        raise ValueError("FT-008 source/checkpoint/custody/optimizer budget mismatch")
    return r


def lease(seed, niche, arm):
    if seed not in SEEDS or niche not in base.NICHES or arm not in ARMS:
        raise ValueError("Unregistered FT-008 optimizer lease")
    return f"{ID}/{seed}/{niche}/{arm}"


def domain_seed(pop, domain, seed, niche, scenario, index):
    variants = {"TRAIN": (TRAIN_SCENARIOS, TRAIN_WORLDS),
                "EVAL": (EVAL_SCENARIOS, EVAL_WORLDS),
                "DIAG": (DIAG_SCENARIOS, DIAG_WORLDS)}
    if domain not in variants or seed not in SEEDS or niche not in base.NICHES:
        raise ValueError("Unregistered FT-008 world domain/model")
    modes, n = variants[domain]
    if scenario not in modes or type(index) is not int or not 0 <= index < n:
        raise ValueError("Unregistered FT-008 scenario/index")
    return pop.stable_seed(ID, domain, seed, niche, scenario, index)


def custody_hash(pop):
    """Exclude all FT-003..007 world domains, including prior held-outs."""
    prior = set()
    for seed in SEEDS:
        for split, count in (("train", 48), ("selection", 4), ("test", 12)):
            prior.update(pop.stable_seed(seed, split, i) for i in range(count))
        prior.update(transfer.heldout_world_seed(pop, seed, i)
                     for i in range(transfer.WORLDS))
        for niche in base.NICHES:
            for name, domains in (
                (ft005.ID, (("TRAIN", ft005.TRAIN_SCENARIOS, ft005.TRAIN_WORLDS_PER_SCENARIO),
                            ("TEST", ft005.EVAL_SCENARIOS, ft005.TEST_WORLDS))),
                (ft006.ID, (("TRAIN", ft006.TRAIN_SCENARIOS, ft006.TRAIN_WORLDS),
                            ("TEST", ft006.TEST_SCENARIOS, ft006.TEST_WORLDS))),
                (ft007.ID, (("PROBE", ft007.SCENARIOS, ft007.PROBE_WORLDS),
                            ("AUTONOMOUS", ft007.SCENARIOS, ft007.EVAL_WORLDS))),
            ):
                prior.update(pop.stable_seed(name, domain, seed, niche, scen, i)
                             for domain, modes, count in domains
                             for scen in modes for i in range(count))
    seen = set()
    for seed in SEEDS:
        for niche in base.NICHES:
            for domain, scenarios, n in (
                ("TRAIN", TRAIN_SCENARIOS, TRAIN_WORLDS),
                ("EVAL", EVAL_SCENARIOS, EVAL_WORLDS),
                ("DIAG", DIAG_SCENARIOS, DIAG_WORLDS)):
                for scenario in scenarios:
                    for i in range(n):
                        val = domain_seed(pop, domain, seed, niche, scenario, i)
                        if val in seen or val in prior:
                            raise AssertionError("FT-008 world custody collision")
                        seen.add(val)
    return hashlib.sha256(json.dumps(sorted(seen)).encode()).hexdigest()


def supervised_targets(pop, niche, world, X, menu, arm, temporal_value):
    """Two counterfactual inputs per real host state; no forged host action.

    Both arms see identical features and get two genuine Adam updates.
    Only the contrast arm conditions STOP preference on the synthetic time
    intervention. Later-time STOP is an *experimental* teacher objective,
    not a declaration that stopping is universally artistically correct.
    """
    if arm not in ARMS or temporal_value not in (0.0, 1.0):
        raise ValueError("Invalid comparison arm/time intervention")
    if "STOP" not in menu or X.shape[0] != len(menu):
        raise ValueError("Illegal/mismatched host action menu")
    early_index = pop.FEATURES.index("early")
    if not np.allclose(X[:, early_index], temporal_value):
        raise AssertionError("Feature intervention not applied to every legal row")
    value = np.asarray([pop.utility(niche, x, world) for x in X], dtype=np.float64)
    active = len(menu) > 1
    if arm == "temporal_stop_ranking" and active:
        stop = menu.index("STOP")
        if temporal_value == 1.0:
            value[stop] -= EARLY_PENALTY
        elif (world.steps / world.max_steps >= LATE_THRESHOLD
              or world.cursor == world.goal):
            value[stop] += LATE_BONUS
    target = pop.softmax(value / TARGET_TEMPERATURE)
    if target.shape != (len(menu),) or not np.isfinite(target).all():
        raise AssertionError("Invalid teacher distribution")
    return target


def budget(t0):
    if time.monotonic() - t0 > MAX_SECONDS:
        raise TimeoutError("FT-008 seed-run wall limit")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MAX_RSS_KIB:
        raise MemoryError("FT-008 process RSS limit")


def train_matched(pop, policies, niche, seed, began):
    start_updates = {arm: p.updates for arm, p in policies.items()}
    exposure = []
    for scenario in TRAIN_SCENARIOS:
        for i in range(TRAIN_WORLDS):
            wseed = domain_seed(pop, "TRAIN", seed, niche, scenario, i)
            w = transfer.world_for(pop, wseed, scenario)
            rng = np.random.default_rng(pop.stable_seed(ID, "scaffold", seed, niche, scenario, i))
            teacher_trace = []
            while not w.stopped and w.steps < w.max_steps:
                menu = w.legal()
                X = np.stack([w.feature(a) for a in menu])
                if len(menu) > 1:
                    for t in (1.0, 0.0):
                        Xcf = X.copy()
                        Xcf[:, pop.FEATURES.index("early")] = t
                        for arm, policy in policies.items():
                            policy.teach(Xcf, supervised_targets(pop, niche, w, Xcf,
                                                                menu, arm, t))
                chosen_index = ft006.teacher_choice(pop, w, menu, rng)
                if type(chosen_index) is not int or not 0 <= chosen_index < len(menu):
                    raise AssertionError("Training scaffold index outside host-legal menu")
                action = menu[chosen_index]
                if action not in w.legal():
                    raise AssertionError("Training scaffold exceeded host authority")
                w.step(action)
                teacher_trace.append(action)
                if w.painted & (w.protected | w.obstacles):
                    raise AssertionError("Synthetic protected/material authority violated")
                budget(began)
            exposure.append({"scenario": scenario, "seed": wseed,
                             "decisions": w.steps, "actions_sha256": hashlib.sha256(
                                 json.dumps(teacher_trace).encode()).hexdigest(),
                             "new_ink": len(w.painted)-1})
    delta = {arm: p.updates - start_updates[arm] for arm, p in policies.items()}
    if min(delta.values()) <= 0 or len(set(delta.values())) != 1:
        raise AssertionError("Unmatched or missing Adam optimizer updates")
    return exposure, delta


def diagnostic_panels(pop, policy, seed, niche, arm, began):
    panels = []
    early_index = pop.FEATURES.index("early")
    for scenario in DIAG_SCENARIOS:
        for i in range(DIAG_WORLDS):
            wseed = domain_seed(pop, "DIAG", seed, niche, scenario, i)
            w = transfer.world_for(pop, wseed, scenario)
            while not w.stopped and w.steps < w.max_steps:
                if w.steps in (0, 24, 48):
                    menu = w.legal()
                    X = np.stack([w.feature(a) for a in menu])
                    p = ft007.score_state(policy, X, menu, w.steps,
                                          transfer.state_digest(w), wseed, early_index)
                    if p is not None:
                        panels.append({"scenario": scenario, **p})
                action = ft007.teacher_action(pop, w, wseed)
                if action not in w.legal():
                    raise AssertionError("Diagnostic scaffold is not host legal")
                w.step(action)
                budget(began)
    return panels


def evaluate(pop, policies, niche, seed, out, began):
    rows = []
    for scenario in EVAL_SCENARIOS:
        worlds = [domain_seed(pop, "EVAL", seed, niche, scenario, i)
                  for i in range(EVAL_WORLDS)]
        arms = {}
        for arm, policy in policies.items():
            episodes = []
            for index, wseed in enumerate(worlds):
                ep, world = transfer.execute(pop, policy, niche, wseed, scenario,
                                             save_trace=(index == 0))
                if ep["decisions"] != ep["model_forwards"] or ep["protected_violations"]:
                    raise AssertionError("Autonomous legal model-forward receipt invalid")
                if index == 0:
                    transfer.audit_receipts(pop, ep["receipts"], niche, wseed, scenario)
                    d = out / "gallery" / niche / scenario
                    d.mkdir(parents=True, exist_ok=True)
                    (d / (arm + ".svg")).write_text(pop.draw_svg(world))
                    (d / (arm + ".json")).write_text(json.dumps(ep, sort_keys=True)+"\n")
                episodes.append(ep)
                budget(began)
            arms[arm] = transfer.summarize(episodes)
        rows.append({"niche": niche, "scenario": scenario,
                     "test_world_seeds": worlds, "arms": arms,
                     "reward_per_decision_delta": round(
                         arms[ARMS[1]]["reward_per_decision"]
                         - arms[ARMS[0]]["reward_per_decision"], 5),
                     "decisions_delta": round(
                         arms[ARMS[1]]["decisions"] - arms[ARMS[0]]["decisions"], 5),
                     "early_stop_delta": round(
                         arms[ARMS[1]]["early_stop"] - arms[ARMS[0]]["early_stop"], 5),
                     "new_ink_delta": round(
                         arms[ARMS[1]]["new_ink"] - arms[ARMS[0]]["new_ink"], 5),
                     "retraces_delta": round(
                         arms[ARMS[1]]["retraces"] - arms[ARMS[0]]["retraces"], 5)})
    return rows


def run(request, upstream, frozen, output, seed):
    validate_request(json.loads(request.read_text()))
    if seed not in SEEDS:
        raise ValueError("Unknown FT-008 seed lease")
    pop, source = transfer.load_source(upstream)
    if "early" not in pop.FEATURES:
        raise ValueError("Pinned host missing time feature")
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = {
        "schema": "cybersand.fleet-training.temporal-ranking-evidence/1",
        "request_id": ID, "seed": seed, "status": "starting",
        "classification": "REAL_NUMPY_ADAM_COUNTERFACTUAL_TRAINING_SYNTHETIC_NOT_NATIVE_W0_OR_SLM",
        "source_commit": transfer.SOURCE_COMMIT, "source_blob": transfer.SOURCE_BLOB,
        "original_source_sha256": base.sha_file(source),
        "executor_sha256": base.sha_file(Path(__file__)),
        "request_sha256": base.sha_file(request),
        "ft003_original_archive_sha256": transfer.FT003_SHA,
        "host_legal_actions_only": True,
        "native_material_edits": 0, "pretrained_slm_optimizer_steps": 0,
        "resident_model_processes_per_runner": 1,
        "workers": [], "evaluations": [], "failures": []
    }
    try:
        report["original_archive_audit"] = transfer.archive_check(
            frozen, source, request.parent / "request_dose.json")
        report["custody_sha256"] = custody_hash(pop)
        report["status"] = "running"
        for niche in base.NICHES:
            source_ckpt = frozen / f"dose_48/{seed}/models/{niche}.json"
            policies = {arm: transfer.load_policy(pop, source_ckpt) for arm in ARMS}
            exposure, deltas = train_matched(pop, policies, niche, seed, began)
            exposure_digest = hashlib.sha256(
                json.dumps(exposure, sort_keys=True).encode()).hexdigest()
            for arm, policy in policies.items():
                dst = output / "models" / arm / (niche+".json")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(json.dumps(policy.serialized(), sort_keys=True)+"\n")
                if base.sha_file(dst) == base.sha_file(source_ckpt):
                    raise AssertionError("No changed real optimized checkpoint")
                panels = diagnostic_panels(pop, policy, seed, niche, arm, began)
                pfile = output / "diagnostics" / arm / (niche+".json")
                pfile.parent.mkdir(parents=True, exist_ok=True)
                pfile.write_text(json.dumps(panels, sort_keys=True)+"\n")
                report["workers"].append({
                    "lease_id": lease(seed, niche, arm), "status": "completed",
                    "model": "population-mlp-24", "seed": seed, "niche": niche,
                    "arm": arm, "original_checkpoint_sha256": base.sha_file(source_ckpt),
                    "optimized_checkpoint_sha256": base.sha_file(dst),
                    "new_adam_updates": deltas[arm],
                    "teacher_exposure_sha256": exposure_digest,
                    "teacher_worlds": len(exposure),
                    "counterfactual_diagnostic_states": len(panels),
                    "diagnostic_stop_rank_flips": sum(x["stop_rank_flip"] for x in panels),
                    "diagnostic_action_flips": sum(x["any_action_flip"] for x in panels),
                    "mean_stop_margin_span": round(sum(x["stop_margin_span"] for x in panels) /
                                                   max(1,len(panels)), 5),
                })
            report["evaluations"].extend(evaluate(pop, policies, niche, seed, output, began))
        rows = report["evaluations"]
        report["aggregate"] = {
            "paired_niche_scenarios": len(rows),
            "reward_improvements": sum(r["reward_per_decision_delta"] > 0 for r in rows),
            "reward_regressions": sum(r["reward_per_decision_delta"] < 0 for r in rows),
            "reward_ties": sum(r["reward_per_decision_delta"] == 0 for r in rows),
            "mean_reward_per_decision_delta": round(sum(r["reward_per_decision_delta"]
                                                       for r in rows)/len(rows), 5),
            "mean_decisions_delta": round(sum(r["decisions_delta"] for r in rows)/len(rows), 5),
            "mean_early_stop_delta": round(sum(r["early_stop_delta"] for r in rows)/len(rows), 5),
            "mean_new_ink_delta": round(sum(r["new_ink_delta"] for r in rows)/len(rows), 5),
            "mean_retraces_delta": round(sum(r["retraces_delta"] for r in rows)/len(rows), 5),
            "diagnostic_stop_rank_flips_by_arm": {
                arm: sum(w["diagnostic_stop_rank_flips"] for w in report["workers"]
                         if w["arm"] == arm) for arm in ARMS},
            "actual_adam_updates": sum(w["new_adam_updates"] for w in report["workers"]),
            "synthetic_evaluation_episodes": len(rows) * len(ARMS) * EVAL_WORLDS,
            "original_gallery_svgs": len(list((output / "gallery").rglob("*.svg"))),
            "elapsed_s": round(time.monotonic()-began, 5),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        }
        if len(report["workers"]) != len(base.NICHES)*len(ARMS):
            raise AssertionError("Missing optimizer lease")
        report["status"] = "completed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failures"].append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        (output / "SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--ft003", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    result = run(args.request, args.upstream, args.ft003, args.output, args.seed)
    print(json.dumps({"status": result["status"], "seed": args.seed,
                      "aggregate": result["aggregate"]}, sort_keys=True))
