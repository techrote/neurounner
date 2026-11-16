#!/usr/bin/env python3
"""Independent W4 recovered ONLINE evidence audit; original W0 replay, no optimizer.

Only the immutable GitHub Actions artifact is accepted. This is not native Godot
painting or a new model evaluation; it rechecks already executed first passes.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from zipfile import ZipFile
import zlib

ARTIFACT_ID = 14165042291
ARCHIVE_SHA256 = "87d4c797698effaf6ed90151cf2ecb050c7bc685f4d810ccdbee75737016f90b"
EVAL_SOURCE = "7953f88bf9d62c116de36537c03f611ba8655517"
TRAIN_SOURCE = "c14795efe978dadeaa9e0da60f6273e18ab17b4d"
W0_SOURCE = "4bb9d6e1d0d0d78183f87d1e3c29648edafacf61"
W0_SHA256 = "81c3923cb38d7f196b99986806216388c34ebc27fd4976e168b3b370c25a323e"
FAMILIES = ("greebles", "pipework", "architecture")
SEEDS = (211, 223)
MODES = ("random", "wall", "fidget", "sprocket-trained")


def check(ok, description):
    if not ok:
        raise ValueError(description)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def spatial(record):
    path = record["trajectory"]
    check(path and len(path) == len(record["events"]) + 1, "trajectory length")
    xy = set(map(tuple, path))
    check(all(0 <= x < 96 and 0 <= y < 96 for x, y in xy), "out of bounds")
    xmin, xmax = min(x for x, _ in xy), max(x for x, _ in xy)
    ymin, ymax = min(y for _, y in xy), max(y for _, y in xy)
    x0, y0 = path[0]
    check(len(xy) == record["metrics"]["unique_trail_cells"], "unique cells")
    return {
        "unique_cells": len(xy),
        "bbox_width": xmax - xmin + 1,
        "bbox_height": ymax - ymin + 1,
        "max_manhattan_from_start": max(abs(x - x0) + abs(y - y0) for x, y in xy),
    }


def source_png_rows(blob):
    check(blob.startswith(b"\x89PNG\r\n\x1a\n"), "PNG signature")
    at, compressed, size, chunks = 8, bytearray(), None, []
    while at < len(blob):
        check(at + 12 <= len(blob), "PNG truncation")
        n = struct.unpack(">I", blob[at:at+4])[0]
        check(0 <= n < 2_000_000 and at + n + 12 <= len(blob), "PNG size")
        kind, raw = blob[at+4:at+8], blob[at+8:at+8+n]
        crc = struct.unpack(">I", blob[at+n+8:at+n+12])[0]
        check(zlib.crc32(kind + raw) & 0xffffffff == crc, "PNG chunk CRC")
        chunks.append(kind)
        if kind == b"IHDR":
            size = struct.unpack(">IIBBBBB", raw)
            check(size == (384, 384, 8, 2, 0, 0, 0), "W0 RGB image format")
        elif kind == b"IDAT":
            compressed.extend(raw)
        else:
            check(kind == b"IEND", "unrecognized chunk")
        at += n + 12
    check(chunks == [b"IHDR", b"IDAT", b"IEND"], "W0 image chunks")
    raster = zlib.decompress(compressed)
    stride = 384 * 3
    check(len(raster) == 384 * (stride + 1), "image raster length")
    rows = []
    for y in range(384):
        offset = y * (stride + 1)
        check(raster[offset] == 0, "unexpected PNG scanline filter")
        rows.append(raster[offset + 1:offset + 1 + stride])
    return rows


def chunk(kind, raw):
    return (struct.pack(">I", len(raw)) + kind + raw +
            struct.pack(">I", zlib.crc32(kind + raw) & 0xffffffff))


def comparison_png(images):
    """Four original W0 pixel panels: random | wall | fidget | trained."""
    groups = [source_png_rows(images[mode]) for mode in MODES]
    w, h = 384 * len(groups), 384
    raster = bytearray()
    for y in range(h):
        raster.append(0)
        for group in groups:
            raster.extend(group[y])
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raster, 6)) +
            chunk(b"IEND", b""))


def audit(archive, trusted_w0, output):
    check(digest(archive.read_bytes()) == ARCHIVE_SHA256, "immutable W4 ZIP digest")
    check(digest(trusted_w0.read_bytes()) == W0_SHA256, "immutable W0 digest")
    spec = importlib.util.spec_from_file_location("w0_independent_w4", trusted_w0)
    check(spec and spec.loader, "W0 import")
    w0 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w0)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact").mkdir(exist_ok=True)
    results, panels, model_sequences, model_pngs = [], {}, set(), {}
    actions = Counter()
    with ZipFile(archive) as z:
        members = z.infolist()
        names = {m.filename for m in members}
        check(len(names) == len(members) == 57, "ZIP member count/duplicate")
        check(all(not n.startswith("/") and "\\" not in n and ".." not in Path(n).parts for n in names), "ZIP paths")
        check(all(m.file_size < 2_000_000 for m in members) and
              sum(m.file_size for m in members) < 2_000_000, "ZIP bounds")
        check(z.testzip() is None, "ZIP CRC")
        inventory = {}
        for line in z.read("SHA256SUMS.txt").decode("ascii").splitlines():
            hash_value, name = line.split("  ", 1)
            name = name.removeprefix("./")
            check(name in names and name not in inventory, "SHA inventory name")
            check(digest(z.read(name)) == hash_value, "SHA mismatch: " + name)
            inventory[name] = hash_value
        check(len(inventory) == 56 and set(inventory) == names - {"SHA256SUMS.txt"}, "SHA manifest coverage")
        check(z.read("source-sha.txt").decode().strip() == EVAL_SOURCE, "actual evaluator source")
        request = json.loads(z.read("request.json"))
        check(request["training_source"] == TRAIN_SOURCE and
              request["training_run_id"] == 43710874563 and
              request["w0_git_commit"] == W0_SOURCE and
              request["checkpoint"] == "checkpoint-000096" and
              request["steps"] == 24 and request["native"] is False, "request identity")
        summary = json.loads(z.read("results/comparison.json"))
        check(summary["complete"] is True and summary["native"] is False and
              summary["sealed_final_opened"] is False and
              summary["training_source"] == TRAIN_SOURCE and
              summary["pinned_w0"] == W0_SOURCE and
              summary["trained_raw_decisions"] == 144 and
              summary["trained_raw_valid_eos5"] == 144 and
              summary["trained_raw_legal"] == 144 and
              summary["real_optimizer_steps_in_this_job"] == 0 and
              len(summary["rollouts"]) == 24, "summary identity")
        seen = set()
        for family in FAMILIES:
            for seed in SEEDS:
                images = {}
                for mode in MODES:
                    key = (family, seed, mode)
                    check(key not in seen, "duplicate record")
                    seen.add(key)
                    stem = f"{family}-{mode}-{seed}"
                    prefix = "results/png-and-json/" + stem
                    obj = json.loads(z.read(prefix + ".json"))
                    png = z.read(prefix + ".png")
                    check(obj["family"] == family and obj["seed"] == seed and
                          obj["mode"] == mode and obj["size"] == 96 and
                          obj["step_budget"] == 24 and obj["steps_executed"] == 24,
                          "rollout identity " + stem)
                    choices = [e["raw_choice"] for e in obj["events"]]
                    policy = "replay" if mode == "sprocket-trained" else mode
                    replay = w0.run(seed, policy, family, 24, 96,
                                    raw_actions=choices if policy == "replay" else None)
                    for field in ("events", "trajectory", "obstacles", "branch_points", "metrics"):
                        check(obj[field] == replay[field], "W0 replay: " + stem + "/" + field)
                    check(w0.png_bytes(replay) == png, "pixel-exact W0 replay " + stem)
                    measurement = spatial(obj)
                    row = [r for r in summary["rollouts"] if
                           (r["family"], r["seed"], r["mode"]) ==
                           (family, seed, "trained" if mode == "sprocket-trained" else mode)]
                    check(len(row) == 1 and row[0]["metrics"] == obj["metrics"] and
                          row[0]["spatial"]["full"] == measurement, "summary metric " + stem)
                    if mode == "sprocket-trained":
                        passes = obj["model_first_passes"]
                        check(len(passes) == 24, "trained first-pass denominator")
                        for i, (event, p) in enumerate(zip(obj["events"], passes)):
                            check(p["tick"] == i and
                                  p["observation"] == event["observation"] and
                                  p["host_choice"] == event["raw_choice"] and
                                  p["applied"] == event["applied"] and
                                  p["eos5"] is True and p["legal_raw"] is True and
                                  p["invalid_host_action"] is False and
                                  p["parsed_action_exact"] == "LEFT" and
                                  p["raw_model_code"] == "1" and
                                  p["raw_output"] == "1<|end|>" and
                                  p["completion_ids"] == [32, 5] and
                                  len(p["prefix_ids"]) > 0 and
                                  event["raw_choice"] == "LEFT" and
                                  event["raw_legal"] is True and
                                  event["applied"] == "LEFT", "unmodified raw LEFT " + stem)
                            actions[event["raw_choice"]] += 1
                        model_sequences.add(tuple(choices))
                        model_pngs[(family, seed)] = digest(png)
                        check(measurement == {"unique_cells": 4, "bbox_width": 2,
                            "bbox_height": 2, "max_manhattan_from_start": 2}, "LEFT-loop extent")
                    images[mode] = png
                    results.append({"family": family, "seed": seed, "mode": mode,
                                    "png_sha256": digest(png), "spatial": measurement,
                                    "metrics": obj["metrics"]})
                out_name = f"{family}-{seed}-random-wall-fidget-trained.png"
                (output / "contact" / out_name).write_bytes(comparison_png(images))
        check(len(seen) == 24 and actions == Counter({"LEFT": 144}) and
              len(model_sequences) == 1, "original action-collapse denominator")
        for seed in SEEDS:
            check(len({model_pngs[(f, seed)] for f in FAMILIES}) == 1,
                  "family-specific output unexpectedly different")
        check(len(set(model_pngs.values())) == 2, "two obstacle-world PNGs")
    report = {
        "schema": "sprocket-support-w4-recovered-independent-original-audit/1",
        "original_artifact_id": ARTIFACT_ID, "original_zip_sha256": ARCHIVE_SHA256,
        "evaluator_source": EVAL_SOURCE, "trainer_source": TRAIN_SOURCE,
        "pinned_w0": W0_SOURCE, "pinned_w0_file_sha256": W0_SHA256,
        "zip_crc_members": 57, "sha256_inventory_members": 56,
        "complete_original_W0_JSON_and_pixel_replays": len(results),
        "trained_original_raw_EOS5_and_legal_decisions": sum(actions.values()),
        "trained_action_histogram": dict(actions),
        "distinct_trained_action_sequences": len(model_sequences),
        "distinct_trained_pngs": len(set(model_pngs.values())),
        "family_identical_per_seed": True, "step_budget": 24,
        "no_optimizer_or_Godot_native_painting": True, "final_opened": False,
        "galleries": sorted(p.name for p in (output / "contact").glob("*.png")),
        "entries": results,
    }
    (output / "w4-recovery-independent-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--trusted-w0", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    opts = p.parse_args()
    r = audit(opts.archive, opts.trusted_w0, opts.output)
    print("PASS: original W4 ZIP", r["zip_crc_members"], "CRC;",
          r["sha256_inventory_members"], "hashes;",
          r["complete_original_W0_JSON_and_pixel_replays"], "pixel/JSON replays;",
          r["trained_original_raw_EOS5_and_legal_decisions"], "original raw legal EOS5;",
          r["distinct_trained_pngs"], "actual seed-group model images; learned LEFT loop")


if __name__ == "__main__":
    main()
