#!/usr/bin/env python3
"""Independent original-byte and host-replay audit of SPROCKET W6 E1.

No optimizer, model inference, native host, corpus modification or FINAL access.
Source and evaluator artifact identities are deliberately immutable.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import zipfile

from audit_w5_s3_paired import comparison_gallery

ARCHIVE_SHA = "b48f7d5d28d121d58a3bfdb1158ce73b28e8d1a4a45bb52a20d6c361304b7d8e"
ARTIFACT_ID = 14175768454
RUN_ID = 43748411090
PRODUCER = "fbeece84ff2609e564bb43b7f40ecde1d80ef187"
W0_COMMIT = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_SHA = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
MODEL_REV = "eb157d4ddddc4619e661d4a2426bb2dd000fc668"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = (211, 223)
MODES = ("random", "wall", "fidget", "flat", "trained")
TEMPERATURE = 2.25
STEPS = 24


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def original_zip(path):
    require(digest(path.read_bytes()) == ARCHIVE_SHA, "original E1 ZIP SHA-256 mismatch")
    z = zipfile.ZipFile(path)
    require(z.testzip() is None, "original ZIP CRC mismatch")
    members = {n for n in z.namelist() if not n.endswith("/")}
    receipt = {}
    for line in z.read("SHA256SUMS.txt").decode().splitlines():
        h, name = line.split("  ", 1)
        name = name.removeprefix("./")
        require(name in members and name not in receipt and len(h) == 64,
                "invalid checksum inventory entry")
        receipt[name] = h
    require(set(receipt) == members - {"SHA256SUMS.txt"}, "incomplete inventory")
    for name, expected in receipt.items():
        require(digest(z.read(name)) == expected, "member checksum mismatch: " + name)
    return z, len(members), len(receipt)


def original_w0(path):
    require(digest(path.read_bytes()) == W0_SHA, "W0 source pin mismatch")
    spec = importlib.util.spec_from_file_location("worm_support_w6_exact_w0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(tuple(module.ACTIONS) == ACTIONS and tuple(module.FAMILIES) == FAMILIES,
            "incompatible original W0 action/family contract")
    return module


def host_draw(seed, tick):
    payload = f"612:sprocket:score-crn-s2:{seed}:{tick}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / (1 << 64)


def policy_legal(obs):
    original = obs["legal"]
    require(original and set(original) <= set(ACTIONS) and len(set(original)) == len(original),
            "invalid original W0 legal mask")
    if obs["tick"] < 20 and len(original) > 1:
        return [a for a in original if a != "STOP"]
    return list(original)


def sample(logits, allowed, u):
    require(set(logits) == set(ACTIONS) and 0 <= u < 1, "invalid scores/RNG")
    require(allowed and set(allowed) <= set(ACTIONS), "empty/bad policy mask")
    ordered = [a for a in ACTIONS if a in allowed]
    scaled = [float(logits[a]) / TEMPERATURE for a in ordered]
    require(all(math.isfinite(v) for v in scaled), "nonfinite logits")
    weights = [math.exp(v - max(scaled)) for v in scaled]
    total = sum(weights)
    probs = {a: 0.0 for a in ACTIONS}
    for a, weight in zip(ordered, weights):
        probs[a] = weight / total
    cumulative = 0.0
    picked = ordered[-1]
    for a in ordered:
        cumulative += probs[a]
        if u < cumulative:
            picked = a
            break
    flat = ordered[min(int(u * len(ordered)), len(ordered) - 1)]
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    effective = 1 / sum(p * p for p in probs.values() if p > 0)
    return picked, flat, probs, entropy, effective


def spatial(record):
    cells = [tuple(pt) for pt in record["trajectory"]]
    require(bool(cells), "empty W0 trace")
    x, y = zip(*cells)
    x0, y0 = cells[0]
    return {
        "unique_cells": len(set(cells)),
        "bbox_width": max(x) - min(x) + 1,
        "bbox_height": max(y) - min(y) + 1,
        "max_manhattan_from_start": max(abs(a - x0) + abs(b - y0) for a, b in cells),
    }


def verify_score_receipts(record, mode, seed, stats):
    receipts = record["score_receipts"]
    events = record["events"]
    require(len(receipts) == len(events), "score receipt/event mismatch")
    for tick, (r, e) in enumerate(zip(receipts, events)):
        obs = e["observation"]
        require(r["tick"] == tick == obs["tick"], "tick mismatch")
        require(r["original_W0_legal"] == obs["legal"], "original legal mask mismatch")
        permitted = policy_legal(obs)
        require(r["host_policy_legal"] == permitted, "effective STOP mask mismatch")
        require(r["host_stop_deferred"] == (tick < 20 and len(obs["legal"]) > 1),
                "STOP deferral receipt mismatch")
        require(r["observation_sha256"] == digest(
            json.dumps(obs, sort_keys=True, separators=(",", ":")).encode()),
            "source observation hash mismatch")
        u = host_draw(seed, tick)
        require(abs(u - r["host_uniform_draw"]) < 1e-15, "host RNG mismatch")
        action, flat, probs, entropy, effective = sample(
            r["full_six_token_logits"], permitted, u)
        require(r["temperature"] == TEMPERATURE and r["selected_action"] == action
                and r["host_selected"] == action and r["actual_W0_applied"] == action
                and e["raw_choice"] == action and e["applied"] == action
                and e["raw_legal"] is True and action in obs["legal"], "sample/host mismatch")
        require(r["selected_host_legal"] is True, "selected action not legal")
        require(r["same_observation_flat"] == flat
                and r["differs_from_flat"] == (flat != action), "flat counterfactual mismatch")
        require(all(abs(probs[a] - r["masked_probs"][a]) < 1e-9 for a in ACTIONS),
                "model softmax mismatch")
        require(abs(entropy - r["entropy_nats"]) < 1e-9
                and abs(effective - r["effective_actions"]) < 1e-9, "distribution stats mismatch")
        top = max(ACTIONS, key=lambda a: r["full_six_token_logits"][a])
        require(r["raw_unmasked_top_action"] == top
                and r["raw_top_host_legal"] == (top in permitted), "raw top attribution")
        require(r["is_eos5_completion"] is False and r["is_greedy_decoding"] is False,
                "sampler falsely attributed to complete EOS output")
        require(r["model_forward"] == (mode == "trained")
                and r["conditioned_six_action_scores"] == (mode == "trained"),
                "model/flat attribution mismatch")
        if mode == "flat":
            require(all(r["full_six_token_logits"][a] == 0 for a in ACTIONS)
                    and r["prefix_ids"] is None and r["token_ids"] is None,
                    "flat score arm not actually flat")
        else:
            require(len(r["token_ids"]) == 6 and isinstance(r["prompt"], str)
                    and bool(r["prefix_ids"]), "trained score provenance missing")
            stats["trained_raw_" + top] += 1
            stats["trained_same_obs_flat_differences"] += int(flat != action)
        stats[mode + "_decisions"] += 1
        stats[mode + "_selected_" + action] += 1


def audit(archive, trusted_w0, output):
    z, member_count, inventory_count = original_zip(archive)
    w0 = original_w0(trusted_w0)
    manifest = json.loads(z.read("results/comparison.json"))
    request = json.loads(z.read("request.json"))
    require(manifest["complete"] is True and manifest["schema"] ==
            "cybersand.worm.sprocket-w6-e1-host-sampled-first-gallery", "not completed E1")
    require(manifest["train_source"] == request["training_source"] == PRODUCER,
            "incorrect optimizer source")
    require(request["w0_git_commit"] == manifest["W0_common_commit"] == W0_COMMIT,
            "W0 source mismatch")
    require(manifest["revision"] == request["model_revision"] == MODEL_REV
            and manifest["model"] == request["model"] == "kandivault/sprocket-500m",
            "model identity mismatch")
    require(manifest["optimizer_steps_this_run"] == 0
            and manifest["source_faithful_W0_not_native"] is True
            and manifest["sealed_final_opened"] is False, "prohibited provenance claim")
    require(manifest["steps_per_trajectory"] == STEPS
            and manifest["selected_temperature"] == TEMPERATURE
            and manifest["seed_groups"] == list(SEEDS)
            and manifest["families"] == list(FAMILIES)
            and len(manifest["rows"]) == 30, "unexpected cohort")
    reported = {(r["family"], r["world_seed"], r["mode"]): r for r in manifest["rows"]}
    require(len(reported) == 30, "duplicate/missing reported trial")
    counts = Counter()
    gallery = []
    pictures = {}
    traces = {}
    for family in FAMILIES:
        for seed in SEEDS:
            for mode in MODES:
                key = (family, seed, mode)
                require(key in reported, "missing required trial: " + repr(key))
                actual_mode = ("sprocket-w6-e1-" + mode if mode in ("flat", "trained") else mode)
                stem = f"results/png-and-json/{family}-{actual_mode}-{seed}"
                record = json.loads(z.read(stem + ".json"))
                png = z.read(stem + ".png")
                pictures[key] = png
                traces[key] = record
                require(record["family"] == family and record["seed"] == seed
                        and record["mode"] == actual_mode and record["size"] == 96
                        and record["step_budget"] == STEPS, "wrong W0 trace identity")
                require(len(record["events"]) == record["steps_executed"] <= STEPS
                        and reported[key]["steps"] == record["steps_executed"], "step receipt mismatch")
                require(reported[key]["metrics"] == record["metrics"], "metric disagreement")
                require(reported[key]["spatial"] == spatial(record), "spatial disagreement")
                actions = [e["raw_choice"] for e in record["events"]]
                replay = w0.run(seed, "replay", family, steps=STEPS, raw_actions=actions)
                for field in ("events", "trajectory", "obstacles", "branch_points", "metrics"):
                    require(replay[field] == record[field], f"original W0 {field} replay mismatch: {key}")
                require(w0.png_bytes(replay) == png, "original W0 PNG replay mismatch")
                counts["replays"] += 1
                if mode in ("trained", "flat"):
                    verify_score_receipts(record, mode, seed, counts)
                    counts[mode + "_rollouts"] += 1
                    counts[mode + "_unique_cell_sum"] += spatial(record)["unique_cells"]
                    counts[mode + "_reach_sum"] += spatial(record)["max_manhattan_from_start"]
                    counts[mode + "_early_stops"] += int(len(actions) < STEPS)
                    require(reported[key]["decision_count"] == len(actions)
                            and reported[key]["model_forward"] == (mode == "trained"),
                            "reported model use/decision count mismatch")
            sources = [pictures[(family, seed, mode)] for mode in MODES]
            out = output / "gallery" / f"{family}-{seed}-five-arm-original.png"
            comparison_gallery(sources, out)
            gallery.append({"name": out.name, "sha256": digest(out.read_bytes())})
    family_pairs = []
    for seed in SEEDS:
        for mode in ("trained", "flat"):
            rows = [traces[(fam, seed, mode)] for fam in FAMILIES]
            pngs = [pictures[(fam, seed, mode)] for fam in FAMILIES]
            for i in range(3):
                for j in range(i + 1, 3):
                    aa = [e["applied"] for e in rows[i]["events"]]
                    bb = [e["applied"] for e in rows[j]["events"]]
                    first = next((k for k, (x, y) in enumerate(zip(aa, bb)) if x != y), None)
                    if first is None and len(aa) != len(bb):
                        first = min(len(aa), len(bb))
                    cells_i = set(map(tuple, rows[i]["trajectory"]))
                    cells_j = set(map(tuple, rows[j]["trajectory"]))
                    family_pairs.append({
                        "seed": seed, "mode": mode,
                        "families": [FAMILIES[i], FAMILIES[j]],
                        "first_action_divergence_tick": first,
                        "same_original_PNG": pngs[i] == pngs[j],
                        "same_raw_action_sequence": aa == bb,
                        "footprint_iou": round(len(cells_i & cells_j) / len(cells_i | cells_j), 6),
                    })
                    if mode == "flat":
                        require(pngs[i] == pngs[j] and aa == bb,
                                "theme-invariant flat negative control diverged")
    require(counts["replays"] == 30 and counts["trained_decisions"] == 144
            and counts["flat_decisions"] == 141, "unexpected actual model/flat cohort")
    require(counts["trained_rollouts"] == counts["flat_rollouts"] == 6
            and counts["trained_same_obs_flat_differences"] == 46,
            "unexpected original paired ablation")
    return {
        "schema": "cybersand.worm.support-sprocket.w6-e1-independent/1",
        "original_producer_run": RUN_ID, "original_artifact": ARTIFACT_ID,
        "original_archive_sha256": ARCHIVE_SHA,
        "original_ZIP_members": member_count, "reverified_member_SHA256": inventory_count,
        "source_W0_commit": W0_COMMIT, "source_W0_sha256": W0_SHA,
        "model_revision": MODEL_REV, "original_optimizer_source": PRODUCER,
        "measured": dict(counts), "family_pairs": family_pairs, "galleries": gallery,
        "limits": [
            "Only two independent obstacle worlds; three family labels per world are correlated.",
            "STOP is suppressed before tick 20 in trained/flat arms, not random/wall/fidget.",
            "Same-observation flat counterfactual is not identical-trajectory W0 masking after divergence.",
            "Sampled six-code logits are not complete greedy/EOS5 model responses.",
            "Python W0 original pixels; neither Godot-native art nor production provider admission.",
            "Higher trajectory coverage and family discrimination are not artistic quality acceptance.",
        ],
        "optimizer_updates_this_audit": 0, "sealed_FINAL_access": False,
    }


def self_test():
    require(host_draw(211, 0) != host_draw(211, 1), "RNG repeats within seed")
    require(host_draw(211, 0) != host_draw(223, 0), "RNG repeats across worlds")
    equal = {a: 0 for a in ACTIONS}
    for u in (0.0, .1, .499, .9, .9999):
        picked, flat, prob, ent, eff = sample(equal, ["LEFT", "STOP"], u)
        require(picked == flat and abs(sum(prob.values()) - 1) < 1e-12,
                "flat sampling self-test")
        require(1 <= eff <= 2 and ent >= 0, "entropy self-test")
    require(policy_legal({"tick": 0, "legal": ["STRAIGHT", "STOP"]}) == ["STRAIGHT"],
            "STOP deferral self-test")
    require(policy_legal({"tick": 20, "legal": ["STRAIGHT", "STOP"]}) ==
            ["STRAIGHT", "STOP"], "STOP restoration self-test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--trusted-w0", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test()
    if args.self_test:
        print("PASS W6 E1 sampling, CRN, STOP deferral self-tests")
        return
    if not args.archive or not args.trusted_w0 or not args.output:
        ap.error("audit requires --archive --trusted-w0 --output")
    args.output.mkdir(parents=True, exist_ok=True)
    result = audit(args.archive, args.trusted_w0, args.output)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"original_replays": result["measured"]["replays"],
        "model_decisions": result["measured"]["trained_decisions"],
        "model_same_obs_flat_differences":
        result["measured"]["trained_same_obs_flat_differences"],
        "galleries": len(result["galleries"]),
        "trained_mean_unique_cells":
        result["measured"]["trained_unique_cell_sum"] / 6,
        "flat_mean_unique_cells": result["measured"]["flat_unique_cell_sum"] / 6,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
