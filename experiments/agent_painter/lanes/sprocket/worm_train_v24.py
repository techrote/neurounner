#!/usr/bin/env python3
"""W24: restore material discrimination while retaining W23 stage-2 frontier coverage.

Experimental model-side optimizer only; SHA-locked W23 stage-2 / W20 / W22 /
W19 parents and external W0. Train, validation, and sealed *new* test seeds
are disjoint. This is not native CyberSand or an update to the 500M root.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, resource, time
from pathlib import Path
import worm_train_v18 as w18
import worm_train_v19 as w19
import worm_train_v20 as w20
import worm_train_v21 as w21
import worm_train_v22 as w22
import worm_train_v23 as w23

VERSION = "sprocket-w24-staged-pareto-material-frontier-rescue-v1"
W23_STAGE2_SHA = "372b8ba2fcb68274c37495c45f6f4bee2b066194395702b49ca6a97506c57f9f"
TRAIN = (933, 937, 941, 947)
VALIDATION = (7301, 7307, 7319)
TEST = (8401, 8419, 8423)
WAVES = 160
STAGES = ("material-hard-pair-rescue", "frontier-protected-retention",
          "onpolicy-branch-replay", "dual-competence-calibration")
CONTROLLERS = ("W20", "W22", "W23-stage2",
               "W24-wave1", "W24-wave2", "W24-wave3", "W24-wave4",
               "W24-selected-T07", "W24-selected-T11", "teacher", "random", "wall")


def eval_all(torch, policy, rows, pairs):
    return w23.eval_all(torch, policy, rows, pairs)


def clone(torch, policy, mat, front):
    result = w23.fusion_class(torch)(mat, front)
    result.load_state_dict(policy.state_dict(), strict=True)
    result.eval()
    return result


def save_checkpoint(torch, model, optimizer, rng, folder, label, result):
    path = folder / (label + ".pt")
    torch.save({"version": VERSION, "stage": label, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "random_state": rng.getstate(),
                "torch_rng": torch.random.get_rng_state(), "assessment": result,
                "W23_stage2_sha256": W23_STAGE2_SHA,
                "W20_sha256": w23.W20_SHA, "W22_sha256": w23.W22_SHA,
                "root_loaded": False, "root_forwards": 0,
                "root_optimizer_steps": 0, "LoRA_updates": 0,
                "native": False, "FINAL_opened": False, "Draft_HOLD": True}, path)
    return {"file": path.name, "sha256": w18.sha(path), "bytes": path.stat().st_size}


def render(torch, w0, policy, seed, family, label, directory, budget=128, temp=0):
    old_steps = w18.STEPS
    try:
        w18.STEPS = budget
        rec = w18.rollout(torch, w0, policy, seed, family, label, temp)
    finally:
        w18.STEPS = old_steps
    stem = f"{family}-{label}-{seed}-n{budget}"
    json_path = directory / (stem + ".json")
    png_path = directory / (stem + ".png")
    w18.save(json_path, rec)
    png_path.write_bytes(w18.material_png(w0, rec))
    return {"family": family, "seed": seed, "controller": label,
            "decision_budget": budget, "actual_steps": rec["steps_executed"],
            "metrics": rec["metrics"], "head_forwards": rec["original_head_forwards"],
            "additional_host_legal_filters": rec["additional_host_legal_filters"],
            "sha_json": w18.sha(json_path), "sha_png": w18.sha(png_path),
            "trajectory_sha256": hashlib.sha256(json.dumps(
                rec["trajectory"], separators=(",", ":")).encode()).hexdigest()}


def aggregates(results):
    groups = {}
    for row in results:
        groups.setdefault(row["controller"], []).append(row)
    output = {}
    for key, rows in groups.items():
        ms = [r["metrics"] for r in rows]
        output[key] = {
            "worlds": len(rows),
            "mean_unique_cells": round(sum(m["unique_trail_cells"] for m in ms) / len(ms), 6),
            "branches": sum(m["branches"] for m in ms),
            "retraces": sum(m["retraces"] for m in ms),
            "mean_actual_steps": round(sum(r["actual_steps"] for r in rows) / len(rows), 6),
            "invalid_raw_choices": sum(m["invalid_raw_choices"] for m in ms)}
    return output


def main(args):
    prior = set(w18.TRAIN + w18.HELDOUT + w19.TRAIN + w19.HELDOUT +
                w20.TRAIN + w20.HELDOUT + w21.TRAIN + w21.HELDOUT +
                w22.TRAIN + w22.HELDOUT + w23.TRAIN + w23.HELDOUT)
    assert len(set(TRAIN + VALIDATION + TEST)) == 10
    assert not prior.intersection(TRAIN + VALIDATION + TEST)
    assert (w18.ROOT, w18.REVISION, w18.EOS) == (
        "kandivault/sprocket-500m", "eb157d4ddddc4619e661d4a2426bb2dd000fc668", 5)
    if args.smoke_only:
        assert len(STAGES) == 4 and len(CONTROLLERS) == 12
        print("W24 source smoke PASS: separated train/validation/test and 640 bounded updates")
        return
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(242424)
    rng = random.Random(242424)
    base, material, frontier = w23.parents(torch, args)
    assert w18.sha(args.w23_stage2) == W23_STAGE2_SHA
    admitted = torch.load(args.w23_stage2, map_location="cpu", weights_only=True)
    assert admitted["version"] == w23.VERSION
    assert admitted["stage"] == "frontier-student-replay"
    assert admitted["root_optimizer_steps"] == admitted["LoRA_updates"] == 0
    model = w23.fusion_class(torch)(material, frontier)
    model.load_state_dict(admitted["model"], strict=True)
    frozen = clone(torch, model, material, frontier)
    frozen_logits = None
    w0 = w18.w0_module(args.w0)
    w18.save(args.output / "provenance.json", {
        "version": VERSION, "source_sha": os.getenv("GITHUB_SHA", "local"),
        "run_id": os.getenv("GITHUB_RUN_ID", "local"), "attempt": os.getenv("GITHUB_RUN_ATTEMPT", "local"),
        "script_sha256": w18.sha(__file__), "W23_stage2_sha256": W23_STAGE2_SHA,
        "W20_sha256": w23.W20_SHA, "W22_sha256": w23.W22_SHA,
        "W19_sha256": w23.W19_SHA, "W0_sha256": w23.W0_SHA,
        "root": w18.ROOT, "revision": w18.REVISION, "EOS": w18.EOS,
        "train_seeds": TRAIN, "validation_seeds": VALIDATION, "test_seeds": TEST,
        "root_model_loaded": False, "root_forwards": 0, "root_optimizer_steps": 0,
        "LoRA_updates": 0, "FINAL_opened": False, "native": False, "Draft_HOLD": True})

    # Preserve W20's same-observation material counterfactual test, *before*
    # imposing the independently generated W22 external obstacle overlay.
    _, train_pairs, train_receipts, _ = w20.interventions(w0, torch, base, TRAIN)
    _, dev_pairs, dev_receipts, _ = w20.interventions(w0, torch, base, VALIDATION)
    _, test_pairs, test_receipts, _ = w20.interventions(w0, torch, base, TEST)
    fixture = w22.overlay_host(w0)
    w18.save(args.output / "external-obstacle-fixture.json", fixture)
    policies = [("teacher", None, 0), ("random", None, 0), ("W20", material, 0),
                ("W22", frontier, 0), ("W23-stage2", frozen, 0),
                ("W23-stage2-T07", frozen, .7)]
    train_rows, train_rollouts = w21.gather(torch, w0, policies, TRAIN)
    dev_rows, dev_rollouts = w21.gather(torch, w0, policies, VALIDATION)
    # New TEST observations are *not* used for training or checkpoint selection.
    w18.save(args.output / "dataset.json", {
        "train_seed_count": len(TRAIN), "validation_seed_count": len(VALIDATION),
        "test_seed_count": len(TEST), "train_rows": len(train_rows),
        "validation_rows": len(dev_rows),
        "train_pairs": len(train_pairs), "validation_pairs": len(dev_pairs),
        "test_pairs": len(test_pairs), "train_rollouts": train_rollouts,
        "validation_rollouts": dev_rollouts,
        "train_material_receipts": train_receipts,
        "validation_material_receipts": dev_receipts,
        "test_material_receipts": test_receipts, "host_original_W0": True,
        "root_loaded": False, "native": False})

    x, m, teacher, legal = w18.tensorize(torch, train_rows)
    planner, _, _ = w22.supervision(torch, train_rows)
    (a0, b0, y0, l0), (a1, b1, y1, l1) = w19.pair_tensors(torch, train_pairs)
    with torch.no_grad():
        frozen_train = frozen(x, m).masked_fill(~legal, -1e4)
        pa = frozen(a0, b0).masked_fill(~l0, -1e4).argmax(1)
        pb = frozen(a1, b1).masked_fill(~l1, -1e4).argmax(1)
    hard_pairs = torch.nonzero((pa != y0) | (pb != y1)).reshape(-1).tolist()
    assert len(hard_pairs) > 0 and len(train_rows) > 100
    w18.save(args.output / "hard-negative-counts.json", {
        "material_counterfactual_hard_pairs": len(hard_pairs),
        "material_counterfactual_all_pairs": len(train_pairs),
        "replay_is_train_only": True})

    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == 8
    optimizer = torch.optim.AdamW(params, lr=.00024, weight_decay=.006)
    stages = [{"stage": "W20-frozen", "assessment": eval_all(torch, material, dev_rows, dev_pairs)},
              {"stage": "W22-frozen", "assessment": eval_all(torch, frontier, dev_rows, dev_pairs)},
              {"stage": "W23-stage2-frozen", "assessment": eval_all(torch, frozen, dev_rows, dev_pairs)}]
    snapshots = {"W23-stage2": frozen}
    losses = []
    for wave, name in enumerate(STAGES):
        replay = None
        if wave >= 2:
            onpolicy, rec = w21.gather(torch, w0, [
                ("W24-train-greedy", clone(torch, model, material, frontier), 0),
                ("W24-train-T07", clone(torch, model, material, frontier), .7)], TRAIN)
            rx, rm, ry, rl = w18.tensorize(torch, onpolicy)
            rp, _, _ = w22.supervision(torch, onpolicy)
            replay = (rx, rm, ry, rl, rp)
            w18.save(args.output / f"replay-wave{wave+1}.json", {
                "rows": len(onpolicy), "receipts": rec, "training_seeds_only": True})
        for step in range(WAVES):
            optimizer.zero_grad(set_to_none=True)
            def take(n, k): return [rng.randrange(n) for _ in range(k)]
            ids = take(len(train_rows), 64)
            pair_ids = take(len(train_pairs), 64)
            hard_ids = [hard_pairs[rng.randrange(len(hard_pairs))] for _ in range(48)]
            logits = model(x[ids], m[ids]).masked_fill(~legal[ids], -1e4)
            plan_loss = torch.nn.functional.cross_entropy(logits, planner[ids])
            teach_loss = torch.nn.functional.cross_entropy(logits, teacher[ids])
            def pair_loss(indices):
                la = model(a0[indices], b0[indices]).masked_fill(~l0[indices], -1e4)
                lb = model(a1[indices], b1[indices]).masked_fill(~l1[indices], -1e4)
                return (torch.nn.functional.cross_entropy(la, y0[indices]) +
                        torch.nn.functional.cross_entropy(lb, y1[indices])) / 2
            mat_loss = pair_loss(pair_ids)
            hard_loss = pair_loss(hard_ids)
            retention = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(logits, dim=1),
                torch.nn.functional.softmax(frozen_train[ids], dim=1), reduction="batchmean")
            # Four deliberately different curriculum stages; compare, never
            # automatically presume the final update is the strongest.
            if wave == 0:
                loss = .43 * mat_loss + .24 * hard_loss + .18 * plan_loss + .08 * teach_loss + .07 * retention
            elif wave == 1:
                loss = .32 * mat_loss + .20 * hard_loss + .29 * plan_loss + .09 * teach_loss + .10 * retention
            else:
                rx, rm, ry, rl, rp = replay
                rid = take(len(rx), 64)
                rlog = model(rx[rid], rm[rid]).masked_fill(~rl[rid], -1e4)
                rloss = torch.nn.functional.cross_entropy(rlog, rp[rid])
                if wave == 2:
                    loss = .28 * mat_loss + .19 * hard_loss + .22 * plan_loss + .16 * rloss + .08 * teach_loss + .07 * retention
                else:
                    loss = .36 * mat_loss + .23 * hard_loss + .16 * plan_loss + .10 * rloss + .08 * teach_loss + .07 * retention
            assert bool(torch.isfinite(loss))
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 2.5)
            if not bool(torch.isfinite(grad)) or float(grad) <= 0:
                raise RuntimeError("zero or nonfinite W24 gradient")
            optimizer.step()
            if (step + 1) % 20 == 0:
                losses.append({"stage": name, "wave_step": step + 1,
                               "global_step": wave * WAVES + step + 1,
                               "loss": round(float(loss.detach()), 6),
                               "gradient_norm": round(float(grad), 6)})
                w18.save(args.output / "partial-loss-history.json", losses)
        stats = eval_all(torch, model, dev_rows, dev_pairs)
        cp = save_checkpoint(torch, model, optimizer, rng, args.output, name, stats)
        stages.append({"stage": name, "optimizer_updates": WAVES,
                       "assessment_validation_only": stats, "checkpoint": cp})
        snapshots[f"W24-wave{wave+1}"] = clone(torch, model, material, frontier)
        w18.save(args.output / "stage-progress.json", stages)

    # Model choice uses only validation worlds, and does not peek at test labels.
    validation_galleries = []
    choices = [("W20", material), ("W22", frontier)] + list(snapshots.items())
    val_metrics = {}
    for label, policy in choices:
        rows = []
        for seed in VALIDATION:
            for family in w18.FAMILIES:
                record = w18.rollout(torch, w0, policy, seed, family, label, 0)
                rows.append(record["metrics"])
        val_metrics[label] = {
            "mean_unique_cells": sum(t["unique_trail_cells"] for t in rows) / len(rows),
            "branches": sum(t["branches"] for t in rows),
            "retraces": sum(t["retraces"] for t in rows)}
    w20_material = stages[0]["assessment"]["material"]["both_teacher_targets_correct"]
    w22_coverage = val_metrics["W22"]["mean_unique_cells"]
    for label, candidate in snapshots.items():
        info = eval_all(torch, candidate, dev_rows, dev_pairs)
        mat_ratio = info["material"]["both_teacher_targets_correct"] / max(1, w20_material)
        frontier_ratio = val_metrics[label]["mean_unique_cells"] / max(1, w22_coverage)
        val_metrics[label].update({
            "material_pairs_both_correct": info["material"]["both_teacher_targets_correct"],
            "material_vs_W20": round(mat_ratio, 6),
            "spatial_vs_W22": round(frontier_ratio, 6),
            "balanced_score": round(min(mat_ratio, frontier_ratio), 6),
            "passes_validation_dual_gate": bool(mat_ratio >= .95 and frontier_ratio >= 1.03)})
    eligible = [k for k in snapshots if val_metrics[k]["passes_validation_dual_gate"]]
    pool = eligible if eligible else list(snapshots)
    selected = max(pool, key=lambda k:(val_metrics[k]["balanced_score"],
                                      val_metrics[k]["mean_unique_cells"], -val_metrics[k]["retraces"]))
    w18.save(args.output / "validation-selection.json", {
        "selection_seed_partition": VALIDATION, "test_metrics_not_used": True,
        "validation_aggregate": val_metrics, "selected": selected,
        "gate_passed": bool(eligible), "fallback_selection": not bool(eligible)})
    test_rows, test_rollouts = w21.gather(torch, w0, policies, TEST)
    test_assessment = {}
    for label, policy in choices:
        test_assessment[label] = eval_all(torch, policy, test_rows, test_pairs)
    w18.save(args.output / "test-assessment.json", {
        "test_seeds": TEST, "source_heldout": True,
        "test_rollout_receipts": test_rollouts, "assessments": test_assessment})

    controllers = [("W20", material, 0), ("W22", frontier, 0),
                   ("W23-stage2", frozen, 0)] + [
                    (f"W24-wave{k}", snapshots[f"W24-wave{k}"], 0) for k in range(1,5)]
    controllers += [("W24-selected-T07", snapshots[selected], .7),
                    ("W24-selected-T11", snapshots[selected], 1.1),
                    ("teacher", None, 0), ("random", None, 0), ("wall", None, 0)]
    assert tuple(k for k, _, _ in controllers) == CONTROLLERS
    images = args.output / "original-synthetic-png-json"
    images.mkdir()
    results = []
    for seed in TEST:
        for family in w18.FAMILIES:
            for label, policy, temp in controllers:
                results.append(render(torch, w0, policy, seed, family, label, images, 128, temp))
        w18.save(args.output / "partial-gallery.json", results)
    assert len(results) == 108
    aggregate = aggregates(results)
    assert all(r["invalid_raw_choices"] == 0 for r in aggregate.values())

    # Secondary exploratory 256-decision cap, with actual executed step counts.
    # W18's STOP choice becomes available from tick 124, so 256 is a cap,
    # not a guarantee of 256 applied decisions.
    horizon = []
    horizon_names = [("W20", material, 0), ("W22", frontier, 0),
                     ("W23-stage2", frozen, 0), ("W24-selected", snapshots[selected], 0),
                     ("W24-selected-T07", snapshots[selected], .7)]
    for seed in TEST:
        for family in w18.FAMILIES:
            for label, policy, temp in horizon_names:
                horizon.append(render(torch, w0, policy, seed, family,
                                      label, images, 256, temp))
    assert len(horizon) == 45
    long_aggregate = aggregates(horizon)
    assert all(r["invalid_raw_choices"] == 0 for r in long_aggregate.values())
    assert all(w18.sha(args.output / s["checkpoint"]["file"]) == s["checkpoint"]["sha256"]
               for s in stages if "checkpoint" in s)
    test_mat = test_assessment[selected]["material"]["both_teacher_targets_correct"]
    base_mat = test_assessment["W20"]["material"]["both_teacher_targets_correct"]
    test_spatial_ratio = aggregate[selected]["mean_unique_cells"] / max(1, aggregate["W22"]["mean_unique_cells"])
    gate = bool(test_mat >= base_mat * .95 and test_spatial_ratio >= 1.03)
    w18.save(args.output / "summary.json", {
        "status": "complete", "version": VERSION, "genuine_aux_optimizer_updates": WAVES * len(STAGES),
        "root_model_forward_calls": 0, "root_model_updates": 0, "LoRA_updates": 0,
        "native": False, "FINAL_opened": False, "production_promotion_allowed": False,
        "selected_using_validation_only": selected,
        "validation_dual_gate": bool(eligible),
        "independent_test_dual_gate": gate,
        "independent_test_material_correct": test_mat, "independent_test_W20_material_correct": base_mat,
        "independent_test_spatial_ratio_vs_W22": round(test_spatial_ratio, 6),
        "synthetic_128_turn_png_json_pairs": len(results),
        "synthetic_256_turn_cap_png_json_pairs": len(horizon),
        "test_controllers": CONTROLLERS,
        "test_controller_aggregate": aggregate,
        "long_horizon_aggregate": long_aggregate,
        "test_assessments": test_assessment,
        "stages": stages,
        "real_aux_head_forwards_in_evidence": sum(v["head_forwards"] for v in results + horizon),
        "cpu_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "256 is a budget cap; W0 permits STOP after decision 124. Native physics and 500M root are not tested."})
    print(json.dumps({"W24": "complete", "optimizer_updates": 640,
                      "128_pairs": len(results), "256_cap_pairs": len(horizon),
                      "selected": selected, "independent_test_gate": gate}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("w23_stage2", "w22", "w20", "w19", "w0", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, dest=name)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    try:
        if not args.smoke_only and not all((args.w23_stage2, args.w22, args.w20,
                                             args.w19, args.w0, args.output)):
            parser.error("W23 stage-2, W22, W20, W19 and W0 must be SHA-admitted")
        main(args)
    except BaseException as exc:
        if args.output and args.output.exists():
            w18.save(args.output / "failure.json",
                     {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
