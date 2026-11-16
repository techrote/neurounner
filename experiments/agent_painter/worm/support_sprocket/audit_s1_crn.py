#!/usr/bin/env python3
"""WORM-001 S1 frozen-logit-tape CRN counterfactual; independent, NO new model inference.

Uses immutable, real hosted S1 logits from the *original* observed trajectory,
but applies them at subsequent counterfactual states without re-querying a model.
Those synthetic paths MUST NOT be described as model-conditioned closed-loop runs.
This isolates host RNG/temperature/legal-mask effects; tests first-step family
score sensitivity separately, where initial W0 states really are matched.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
from zipfile import ZipFile
import zlib

ARCHIVE_SHA256 = "88e12604307af51fb9406b88149558254b09760ac1b57034d9934c33d107d133"
W0_SHA256 = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
S1_SOURCE = "b0a68d45f54ba5f4d9b6b11c505f80d51cc8eb3b"
W0_SOURCE = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
ACTIONS = ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP")
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = (211, 223)
IMAGE_SIZE = 384


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()


def png_chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))


def read_rgb_png(blob):
    require(blob[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature")
    at, compressed, chunks = 8, bytearray(), []
    while at < len(blob):
        n = struct.unpack(">I", blob[at:at+4])[0]
        require(n < 2000000 and at + n + 12 <= len(blob), "PNG bounded chunk")
        kind, data = blob[at+4:at+8], blob[at+8:at+8+n]
        crc = struct.unpack(">I", blob[at+n+8:at+n+12])[0]
        require(zlib.crc32(kind + data) & 0xffffffff == crc, "PNG CRC")
        chunks.append(kind)
        if kind == b"IHDR":
            require(struct.unpack(">IIBBBBB", data) ==
                    (IMAGE_SIZE, IMAGE_SIZE, 8, 2, 0, 0, 0), "PNG W0 format")
        elif kind == b"IDAT":
            compressed.extend(data)
        else:
            require(kind == b"IEND", "PNG chunk")
        at += n + 12
    require(chunks == [b"IHDR", b"IDAT", b"IEND"], "PNG chunk sequence")
    rowlen = IMAGE_SIZE * 3
    raw = zlib.decompress(compressed)
    require(len(raw) == IMAGE_SIZE * (rowlen + 1), "PNG raster length")
    rows = []
    for y in range(IMAGE_SIZE):
        offset = y * (rowlen + 1)
        require(raw[offset] == 0, "W0 PNG unsupported filter")
        rows.append(raw[offset+1:offset+1+rowlen])
    return rows


def compose(rows_by_panel):
    require(len(rows_by_panel) == 6, "six-panel contact")
    width = IMAGE_SIZE * len(rows_by_panel)
    data = bytearray()
    for y in range(IMAGE_SIZE):
        data.append(0)
        for rows in rows_by_panel:
            data.extend(rows[y])
    return (b"\x89PNG\r\n\x1a\n" +
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, IMAGE_SIZE, 8, 2, 0, 0, 0)) +
            png_chunk(b"IDAT", zlib.compress(bytes(data), 9)) +
            png_chunk(b"IEND", b""))


def softmax(logits, legal, temperature):
    require(temperature > 0 and all(a in ACTIONS for a in legal), "legal/temp")
    require(len(legal) == len(set(legal)) and "STOP" in legal, "legal uniqueness")
    scores = [float(logits[a]) / temperature for a in legal]
    require(all(math.isfinite(s) for s in scores), "finite logits")
    m = max(scores)
    weights = [math.exp(s - m) for s in scores]
    total = sum(weights)
    return dict(zip(legal, [v / total for v in weights]))


def choose(probs, uniform):
    require(0 <= uniform < 1, "host uniform range")
    accum = 0.0
    for a in ACTIONS:
        accum += probs.get(a, 0.0)
        if uniform < accum + 1e-15:
            return a
    raise ValueError("softmax probabilities do not sum to one")


def shared_uniform(seed, tick):
    # Common random number across family, temperature and model-prior ablations.
    payload = f"WORM-S1-CRN-1|world={seed}|tick={tick}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def spatial(record):
    path = record["trajectory"]
    xy = set(map(tuple, path))
    x0, y0 = path[0]
    xmin, xmax = min(x for x, _ in xy), max(x for x, _ in xy)
    ymin, ymax = min(y for _, y in xy), max(y for _, y in xy)
    require(len(path) == record["steps_executed"] + 1, "path length")
    require(len(xy) == record["metrics"]["unique_trail_cells"], "unique coverage")
    return dict(unique_cells=len(xy), bbox=[xmax-xmin+1, ymax-ymin+1],
                radius=max(abs(x-x0)+abs(y-y0) for x, y in xy),
                stop=record["metrics"]["terminated_by_stop"],
                branches=record["metrics"]["branches"],
                retraces=record["metrics"]["retraces"],
                revisits=record["metrics"]["total_revisit_contacts"])


def run(artifact, trusted_w0, out):
    raw = Path(artifact).read_bytes()
    require(sha(raw) == ARCHIVE_SHA256, "original S1 ZIP SHA256")
    require(sha(Path(trusted_w0).read_bytes()) == W0_SHA256, "W0 original source SHA")
    spec = importlib.util.spec_from_file_location("trusted_w0", trusted_w0)
    require(spec and spec.loader, "W0 load")
    w0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w0)
    require(tuple(w0.ACTIONS) == ACTIONS, "W0 actions")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pngdir = out / "counterfactual-w0"
    pngdir.mkdir(exist_ok=True)
    gallerydir = out / "contact"
    gallerydir.mkdir(exist_ok=True)
    with ZipFile(artifact) as z:
        names = set(z.namelist())
        require(len(names) == 69 and len(names) == len(z.namelist()), "original ZIP member list")
        require(all(item.CRC == zlib.crc32(z.read(item.filename)) & 0xffffffff
                    for item in z.infolist()), "archive CRC")
        sums = z.read("SHA256SUMS.txt").decode().splitlines()
        require(len(sums) == 68, "original internal sha rows")
        for entry in sums:
            h, filename = entry.split(maxsplit=1)
            name = filename.removeprefix("./")
            require(name in names and h == sha(z.read(name)), "original internal SHA " + name)
        require(z.read("source-sha.txt").decode().strip() == S1_SOURCE, "original S1 source")
        request = json.loads(z.read("request.json"))
        require(request["w0_git_commit"] == W0_SOURCE and
                request["training_source"] == "c14795efe978dadeaa9e0da60f6273e18ab17b4d",
                "S1 producer custody")
        original = {}
        for family in FAMILIES:
            for seed in SEEDS:
                for temperature in (1, 3):
                    p = f"results/png-and-json/{family}-sprocket-scored-t{temperature}-{seed}"
                    r = json.loads(z.read(p + ".json"))
                    require(r["family"] == family and r["seed"] == seed and
                            r["step_budget"] == 24 and len(r["events"]) == len(r["score_receipts"]),
                            "original original S1 record scope")
                    require(r["obstacles"] == [list(p) for p in sorted(w0.initial_world(seed))],
                            "W0 original obstacle world")
                    require(z.read(p + ".png") == w0.png_bytes(r), "original W0 pixel replay")
                    for i, s in enumerate(r["score_receipts"]):
                        require(s["host_legal"] == r["events"][i]["observation"]["legal"],
                                "original legal/observation receipts")
                        require(s["applied_by_host"] == r["events"][i]["applied"] and
                                s["selected_action"] == r["events"][i]["raw_choice"],
                                "original choices")
                        require(s["raw_unmasked_top_action"] == "LEFT", "original S1 top class")
                        require(tuple(s["full_six_token_logits"]) == tuple(sorted(ACTIONS)),
                                "six action-logit labels")  # JSON output sort order
                    original[(family, seed, temperature)] = (r, z.read(p + ".png"))
        first_step = []
        for seed in SEEDS:
            compare = []
            for family in FAMILIES:
                a, b = (original[(family, seed, t)][0] for t in (1, 3))
                obs_a, obs_b = a["events"][0]["observation"], b["events"][0]["observation"]
                require(obs_a == obs_b, "identical initial observation across T")
                one, three = a["score_receipts"][0], b["score_receipts"][0]
                require(one["full_six_token_logits"] == three["full_six_token_logits"],
                        "initial logits equal across T")
                require(one["host_legal"] == three["host_legal"], "initial legal equal")
                compare.append((family, one, obs_a))
            bare = [{k: v for k, v in o.items() if k != "family"} for _, _, o in compare]
            require(bare[0] == bare[1] == bare[2],
                    "same W0 first-step local observation except family")
            for i in range(3):
                for j in range(i+1, 3):
                    fa, a, _ = compare[i]
                    fb, b, _ = compare[j]
                    legal = a["host_legal"]
                    p = softmax(a["full_six_token_logits"], legal, 1)
                    q = softmax(b["full_six_token_logits"], legal, 1)
                    first_step.append({
                        "world_seed": seed, "comparison": [fa, fb],
                        "same_initial_W0_except_family": True,
                        "top_class_both": ["LEFT", "LEFT"],
                        "max_absolute_logit_delta": max(abs(a["full_six_token_logits"][k] -
                                                           b["full_six_token_logits"][k]) for k in ACTIONS),
                        "softmax_total_variation_T1": sum(abs(p[k]-q[k]) for k in legal) / 2,
                    })
        results = []
        for family in FAMILIES:
            for seed in SEEDS:
                baseline = original[(family, seed, 1)][0]
                tape = [s["full_six_token_logits"] for s in baseline["score_receipts"]]
                require(len(tape) == 24, "full first-trajectory logit tape")
                mean_logits = {a: sum(t[a] for t in tape)/24 for a in ACTIONS}
                panels = [original[(family, seed, 1)][1], original[(family, seed, 3)][1]]
                for policy, temp in (("tape-t1", 1), ("tape-t3", 3),
                                     ("mean-prior-t3", 3), ("uniform-t3", 3)):
                    decisions = []
                    def selector(obs):
                        tick, legal = obs["tick"], obs["legal"]
                        logits = ({a: 0.0 for a in ACTIONS} if policy == "uniform-t3" else
                                  mean_logits if policy == "mean-prior-t3" else tape[tick])
                        uniform = shared_uniform(seed, tick)
                        selected = choose(softmax(logits, legal, temp), uniform)
                        decisions.append(dict(tick=tick, legal=legal, host_uniform=uniform,
                                              selected=selected,
                                              used_recorded_logit_tape=policy.startswith("tape")))
                        return selected
                    record = w0.run(seed, "callback", family, steps=24, size=96,
                                    selector=selector)
                    require(len(decisions) == len(record["events"]), "callback receipt count")
                    require(all(e["raw_legal"] and e["applied"] == e["raw_choice"]
                                for e in record["events"]), "W0 legal choices")
                    replay = w0.run(seed, "replay", family, steps=24, size=96,
                                    raw_actions=[event["raw_choice"] for event in record["events"]])
                    require(record["trajectory"] == replay["trajectory"] and
                            record["metrics"] == replay["metrics"] and
                            w0.png_bytes(record) == w0.png_bytes(replay),
                            "counterfactual original W0 exact action-tape pixel replay")
                    output_prefix = f"{family}-{seed}-{policy}"
                    png = w0.png_bytes(record)
                    pngname = f"counterfactual-w0/{output_prefix}.png"
                    jsonname = f"counterfactual-w0/{output_prefix}.json"
                    (out / pngname).write_bytes(png)
                    (out / jsonname).write_bytes(canonical({"record": record,
                        "crn_choices": decisions,
                        "not_model_inference": True,
                        "recorded_logits_from_original_T1_trajectory_only": True}))
                    panels.append(png)
                    results.append(dict(family=family, world_seed=seed, mode=policy,
                                        steps=record["steps_executed"], spatial=spatial(record),
                                        actions=dict(sorted(Counter(e["applied"] for e in record["events"]).items())),
                                        png_sha256=sha(png), json_sha256=sha((out / jsonname).read_bytes())))
                contact = compose([read_rgb_png(b) for b in panels])
                (gallerydir / f"{family}-{seed}-six-policy.png").write_bytes(contact)
    report = {
        "schema": "cybersand.worm.sprocket-support.s1-crn-v1",
        "source_s1": S1_SOURCE, "source_w0": W0_SOURCE,
        "original_s1_artifact_sha256": ARCHIVE_SHA256,
        "trusted_w0_sha256": W0_SHA256,
        "original_s1_zip_crc": 69, "original_s1_member_hashes": 68,
        "original_trained_S1_W0_PNG_replays": 12,
        "original_S1_greedy_unmasked_TOP_LEFT": sum(len(original[k][0]["score_receipts"]) for k in original),
        "original_S1_temperature_sampled_runs": 12,
        "first_step_family_pair_tests": first_step,
        "first_step_model_logit_TV_max": max(row["softmax_total_variation_T1"] for row in first_step),
        "first_step_model_logit_delta_max": max(row["max_absolute_logit_delta"] for row in first_step),
        "new_W0_counterfactual_rollouts": len(results),
        "new_W0_counterfactual_pngs": len(results),
        "counterfactual_original_W0_pixel_exact_action_replays": len(results),
        "original_pixel_six_policy_contact_sheets": 6,
        "crn": "sha256(world-seed,tick) shared across family/T/prior; NOT original S1 host RNG",
        "counterfactual_policy_misrepresentation_guard":
            "FROZEN ORIGINAL T1 SCORE TAPE fed to later counterfactual observations. NOT live learned model inference; changed observations were never scored.",
        "model_optimizer_updates": 0, "Godot_native_paintings": 0,
        "outcomes": results,
        "gallery_columns": ["original-S1-t1", "original-S1-t3",
                            "counterfactual-tape-t1", "counterfactual-tape-t3",
                            "counterfactual-mean-prior-t3", "counterfactual-uniform-t3"],
    }
    (out / "s1-crn-audit-report.json").write_bytes(canonical(report))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{sha(p.read_bytes())}  {p.relative_to(out)}\n" for p in files))
    print(json.dumps({"original_S1_PNG_replays": 12, "counterfactual_W0_replays": len(results),
                      "max_T1_first_step_TV": report["first_step_model_logit_TV_max"],
                      "max_first_step_logit_delta": report["first_step_model_logit_delta_max"],
                      "output": str(out), "not_live_model": True}, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("original_s1_zip", type=Path)
    parser.add_argument("--trusted-w0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.original_s1_zip, args.trusted_w0, args.output)
