"""FT-010 legal-goal corpus proof: external synthetic host, NOT native PAINT/W0.

Legally construct real reached-goal states for subsequent STOP-conditioned training,
without teleporting, changing protected material or mandating A*-like worm art.
No model training or model inference occurs in this source-only preflight.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import resource
import time

import fleet_bench as base
import ft004_shift as transfer
import ft009_situational_stop as previous

ID = "FT-010-20261116-reached-goal-corpus"
SCHEMA = "cybersand.fleet-training.goal-corpus-request/1"
SCENARIOS = ("long60", "dense60", "shifted_guard60")
SPLITS = ("TRAIN_CANDIDATES", "HELDOUT_CANDIDATES")
SEEDS = (9401, 9403)
POOL = 48
MIN_GOALS = 3
WALL_SECONDS = 180
RSS_KIB = 900000


def checked_request(r):
    expected = {
        "schema": SCHEMA, "request_id": ID,
        "source_commit": transfer.SOURCE_COMMIT,
        "source_blob": transfer.SOURCE_BLOB,
        "ft003_original_artifact_id": 14194214028,
        "ft003_archive_sha256": transfer.FT003_SHA,
        "seeds": list(SEEDS), "splits": list(SPLITS),
        "scenarios": list(SCENARIOS), "pool_per_group": POOL,
        "minimum_reachable_nontrivial_goals_per_group": MIN_GOALS,
        "wall_seconds_per_seed": WALL_SECONDS, "rss_kib_cap": RSS_KIB,
        "max_parallel_runners": 2, "optimizer_updates": 0,
        "native_material_edits": 0, "pretrained_slm_optimizer_steps": 0,
    }
    if type(r) is not dict or r != expected:
        raise ValueError("FT-010 immutable job request/custody/budget mismatch")
    return r


def seed_for(pop, split, seed, niche, scenario, i):
    if (split not in SPLITS or seed not in SEEDS or niche not in base.NICHES
            or scenario not in SCENARIOS or type(i) is not int or not 0 <= i < POOL):
        raise ValueError("Invalid goal-corpus lease or held-out custody")
    return pop.stable_seed(ID, split, seed, niche, scenario, i)


def legal_route(pop, world):
    """Non-revisiting path in untouched real host state before halfway switch."""
    if world.steps or world.stopped or world.switched:
        raise ValueError("Route must start from untouched world")
    queue = deque([(world.cursor, ())])
    seen = {world.cursor}
    limit = world.max_steps // 2 - 2
    while queue:
        pos, route = queue.popleft()
        if pos == world.goal:
            return route if 0 < len(route) <= limit else None
        if len(route) == limit:
            continue
        for action, (dx, dy) in pop.DIRECTIONS.items():
            nxt = (pos[0]+dx, pos[1]+dy)
            if nxt not in seen and world.open(nxt):
                seen.add(nxt)
                queue.append((nxt, route + (action,)))
    return None


def legal_replay(pop, world, route):
    """Replay action-by-action through host, retaining original state receipts."""
    goal = world.goal
    receipts = []
    for action in route + ("STOP",):
        menu = world.legal()
        if action not in menu or ((world.cursor == world.goal) != (action == "STOP")):
            raise AssertionError("Invalid real host goal/STOP action")
        if world.switched:
            raise AssertionError("Goal switched before arrival")
        before = transfer.state_digest(world)
        world.step(action)
        receipts.append({"turn": world.steps, "menu": menu, "action": action,
                         "before": before, "after": transfer.state_digest(world)})
        if world.painted & (world.obstacles | world.protected):
            raise AssertionError("Material/protection authority violated")
    if world.cursor != goal or not world.stopped or world.switched:
        raise AssertionError("False legally reached goal")
    return receipts


def run(upstream, output, seed, request):
    checked_request(json.loads(request.read_text()))
    if seed not in SEEDS:
        raise ValueError("Unknown seed lease")
    pop, source = transfer.load_source(upstream)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema": "cybersand.fleet-training.reached-goal-corpus-evidence/1",
        "id": ID, "seed": seed, "status": "starting",
        "classification": "REAL_EXTERNAL_SYNTHETIC_HOST_TEACHER_ROUTE_NOT_MODEL_INFERENCE",
        "source_commit": transfer.SOURCE_COMMIT,
        "source_blob": transfer.SOURCE_BLOB,
        "original_source_sha256": base.sha_file(source),
        "executor_sha256": base.sha_file(Path(__file__)),
        "request_sha256": base.sha_file(request),
        "optimizer_updates": 0, "native_material_edits": 0,
        "model_forward_calls": 0, "rows": [], "failures": []}
    try:
        seen=set()
        for split in SPLITS:
            for niche in base.NICHES:
                for scenario in SCENARIOS:
                    relevant_old = {
                        previous.domain_seed(pop, dom, seed, niche, s, i)
                        for dom, scenarios, count in (
                            ("TRAIN",previous.TRAIN_SCENARIOS,previous.TRAIN_WORLDS),
                            ("EVAL",previous.EVAL_SCENARIOS,previous.EVAL_WORLDS),
                            ("DIAG",previous.DIAG_SCENARIOS,previous.DIAG_WORLDS))
                        for s in scenarios for i in range(count)}
                    reached, nontrivial, unreachable, paths, saved = 0,0,0,[],[]
                    for index in range(POOL):
                        sid = seed_for(pop,split,seed,niche,scenario,index)
                        if sid in seen or sid in relevant_old:
                            raise AssertionError("Reused held-out/prior seed")
                        seen.add(sid)
                        world = transfer.world_for(pop,sid,scenario)
                        route = legal_route(pop,world)
                        if route is None:
                            unreachable += 1
                            continue
                        receipts = legal_replay(pop,world,route)
                        reached += 1
                        if len(receipts[-1]["menu"]) > 1:
                            nontrivial += 1
                        paths.append(len(route))
                        if len(saved) < MIN_GOALS:
                            d=output/"samples"/split/niche/scenario
                            d.mkdir(parents=True,exist_ok=True)
                            (d / f"{index}.svg").write_text(pop.draw_svg(world))
                            (d / f"{index}.json").write_text(json.dumps({
                                "seed":sid,"path":route,"receipts":receipts,
                                "classification":"LEGAL_SYNTHETIC_TEACHER_NOT_AUTONOMOUS"},
                                sort_keys=True)+"\n")
                            saved.append(sid)
                        if time.monotonic()-started > WALL_SECONDS:
                            raise TimeoutError("FT-010 wall budget")
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > RSS_KIB:
                            raise MemoryError("FT-010 RSS budget")
                    report["rows"].append({
                        "split":split,"niche":niche,"scenario":scenario,
                        "candidate_worlds":POOL,"legally_reached":reached,
                        "nontrivial_stop_menus_at_goal":nontrivial,
                        "unreachable_before_goal_switch":unreachable,
                        "shortest_route":min(paths) if paths else None,
                        "longest_route":max(paths) if paths else None,
                        "original_example_seeds":saved})
        report["seed_custody_sha256"]=hashlib.sha256(
            json.dumps(sorted(seen)).encode()).hexdigest()
        report["summary"]={
            "candidate_worlds":len(report["rows"])*POOL,
            "legally_reached":sum(x["legally_reached"] for x in report["rows"]),
            "goal_states_with_choices":sum(
                x["nontrivial_stop_menus_at_goal"] for x in report["rows"]),
            "groups_meeting_3_goal_threshold":sum(
                x["nontrivial_stop_menus_at_goal"] >= MIN_GOALS
                for x in report["rows"]),
            "groups":len(report["rows"]),
            "replayed_examples":sum(
                len(x["original_example_seeds"]) for x in report["rows"]),
            "elapsed_s":round(time.monotonic()-started,5),
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        report["status"]="completed"
    except BaseException as exc:
        report["status"]="failed"
        report["failures"].append(type(exc).__name__+": "+str(exc))
        raise
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        (output/"SHA256SUMS").write_text("".join(
            base.sha_file(p)+"  "+str(p.relative_to(output))+"\n"
            for p in sorted(output.rglob("*")) if p.is_file() and p.name!="SHA256SUMS"))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--request",type=Path,required=True)
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.upstream,args.output,args.seed,args.request)["summary"],sort_keys=True))
