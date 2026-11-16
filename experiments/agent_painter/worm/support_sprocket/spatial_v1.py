#!/usr/bin/env python3
"""WORM SPROCKET support: original W0 24/96-step spatial reach & gallery audit.

Reads the generated *model-free* observed-v2 artifact on the same GitHub runner.
Regenerates original host trajectories and PNG bytes with the pinned W0 callback,
then derives bbox/radius diagnostics. Does not touch TRAIN/DEV or model weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import observed_v2 as source

W0 = source.worm
POLICIES = ("random", "wall", "fidget") + source.POLICIES
PREFIX = 24
FULL = source.GALLERY_BUDGET
VERSION = "sprocket-support-spatial-v1"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def check(condition, why: str) -> None:
    if not condition:
        raise AssertionError(why)


def extent(record: dict) -> dict:
    path = record["trajectory"]
    check(bool(path), "empty W0 trajectory")
    size = record["size"]
    check(all(isinstance(p, list) and len(p) == 2
              and all(isinstance(v, int) and 0 <= v < size for v in p)
              for p in path), "out-of-bounds W0 path")
    xy = {tuple(p) for p in path}
    xmin, xmax = min(x for x, _ in xy), max(x for x, _ in xy)
    ymin, ymax = min(y for _, y in xy), max(y for _, y in xy)
    x0, y0 = path[0]
    bbox = [xmin, ymin, xmax, ymax]
    width, height = xmax - xmin + 1, ymax - ymin + 1
    count = Counter(e["applied"] for e in record["events"])
    check(record["steps_executed"] == len(record["events"])
          and len(path) == len(record["events"]) + 1,
          "W0 event/trajectory length")
    check(len(xy) == record["metrics"]["unique_trail_cells"],
          "W0 unique-cell metric mismatch")
    return {
        "actions_executed": record["steps_executed"],
        "unique_cells": len(xy),
        "bbox_xyxy_inclusive": bbox,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": width * height,
        "max_manhattan_from_start": max(abs(x-x0)+abs(y-y0) for x, y in xy),
        "end_manhattan_from_start": abs(path[-1][0]-x0)+abs(path[-1][1]-y0),
        "branch_actions": count["BRANCH"],
        "retrace_actions": count["RETRACE"],
        "stop_actions": count["STOP"],
        "host_invalid_raw_choices": record["metrics"]["invalid_raw_choices"],
        "revisit_contacts": record["metrics"]["total_revisit_contacts"],
    }


def png_rows(blob: bytes) -> tuple[int, int, list[bytes]]:
    """Read only the exact W0-generated 8-bit RGB/filter-0 PNG subset."""
    check(blob.startswith(b"\x89PNG\r\n\x1a\n"), "PNG signature")
    offset, ids, w, h = 8, [], 0, 0
    raw_parts = []
    while offset < len(blob):
        check(offset + 12 <= len(blob), "truncated PNG chunk")
        n = struct.unpack(">I", blob[offset:offset+4])[0]
        check(n < 1_000_000 and offset + 12 + n <= len(blob), "unsafe PNG chunk")
        kind = blob[offset+4:offset+8]
        payload = blob[offset+8:offset+8+n]
        expected_crc = struct.unpack(">I", blob[offset+8+n:offset+12+n])[0]
        check((zlib.crc32(kind+payload) & 0xffffffff) == expected_crc,
              "PNG CRC mismatch")
        ids.append(kind)
        if kind == b"IHDR":
            check(len(payload) == 13, "PNG IHDR length")
            w, h, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload)
            check((w,h,depth,color,compression,filtering,interlace) ==
                  (384,384,8,2,0,0,0), "unexpected W0 image format")
        elif kind == b"IDAT":
            raw_parts.append(payload)
        elif kind == b"IEND":
            check(offset+12+n == len(blob), "PNG trailing data")
        else:
            raise AssertionError("unexpected PNG chunk type")
        offset += 12+n
    check(ids[0] == b"IHDR" and ids[-1] == b"IEND"
          and ids.count(b"IDAT") == 1, "PNG structure mismatch")
    raw = zlib.decompress(b"".join(raw_parts))
    stride = w*3
    check(len(raw) == h*(stride+1), "PNG decompressed byte count")
    rows = [raw[i*(stride+1)+1:(i+1)*(stride+1)] for i in range(h)]
    check(all(raw[i*(stride+1)] == 0 and len(row) == stride
              for i,row in enumerate(rows)), "PNG is not filter-0 RGB")
    return w,h,rows


def chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I",len(payload))+kind+payload+
            struct.pack(">I",zlib.crc32(kind+payload)&0xffffffff))


def contact_sheet(columns: list[str], images: dict[tuple[str,str],bytes]) -> bytes:
    """Six policy columns; 24-step upper row, 96-step lower row; no interpolation."""
    check(len(columns) == 6, "contact-sheet policy count")
    tiles = {(policy,horizon):png_rows(images[(policy,horizon)])[2]
             for policy in columns for horizon in ("prefix24","full96")}
    w,h = 384*len(columns),384*2
    raster = bytearray()
    for horizon in ("prefix24","full96"):
        for y in range(384):
            raster.append(0)
            for policy in columns:
                raster.extend(tiles[(policy,horizon)][y])
    check(len(raster) == h*(w*3+1), "contact-sheet raster size")
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB",w,h,8,2,0,0,0)) +
            chunk(b"IDAT", zlib.compress(bytes(raster), 6)) +
            chunk(b"IEND", b""))


def audit(gallery: Path, output: Path) -> dict:
    manifest = json.loads((gallery/"manifest.json").read_bytes())
    check(manifest["version"] == source.VERSION
          and manifest["gallery_count"] == 36
          and manifest["model_optimizer_steps"] == 0
          and manifest["matched_steps"] == FULL, "W0 gallery source contract")
    check(len(manifest["entries"]) == 36, "W0 gallery entry count")
    output.mkdir(parents=True,exist_ok=True)
    (output/"prefix24").mkdir(exist_ok=True)
    (output/"contact").mkdir(exist_ok=True)
    seen = set()
    by_world = {}
    results = []
    for item in manifest["entries"]:
        seed, family, policy, stem = (item["world_seed"],item["family"],
                                      item["policy"],item["id"])
        check((seed,family,policy) not in seen
              and stem == f"{family}-{policy}-{seed}"
              and seed in source.GALLERY_WORLDS
              and family in source.FAMILIES
              and policy in POLICIES, "gallery identity collision")
        seen.add((seed,family,policy))
        blob=(gallery/(stem+".json")).read_bytes()
        full_png=(gallery/(stem+".png")).read_bytes()
        check(digest(blob)==item["sha256_json"]
              and digest(full_png)==item["sha256_png"], "original gallery hashes")
        record=json.loads(blob)
        check(record["seed"]==seed and record["family"]==family
              and record["size"]==96 and record["step_budget"]==FULL
              and record["metrics"]==item["metrics"], "W0 original trace identity")
        raw=[e["raw_choice"] for e in record["events"]]
        replay=W0.run(seed,"replay",family,FULL,96,raw_actions=raw)
        for key in ("events","trajectory","obstacles","branch_points","metrics"):
            check(record[key]==replay[key], "original 96-step W0 replay: "+stem+" "+key)
        check(W0.png_bytes(replay)==full_png, "full pixel-exact W0 replay: "+stem)
        prefix=W0.run(seed,"replay",family,PREFIX,96,raw_actions=raw[:PREFIX])
        check(prefix["events"]==record["events"][:len(prefix["events"])]
              and prefix["trajectory"]==record["trajectory"][:len(prefix["trajectory"])]
              and prefix["steps_executed"]==min(PREFIX,record["steps_executed"]),
              "24-step original-action prefix diverged: "+stem)
        before,after=extent(prefix),extent(record)
        check(before["unique_cells"]<=after["unique_cells"]
              and before["max_manhattan_from_start"]<=after["max_manhattan_from_start"],
              "non-monotonic path extent: "+stem)
        prefix_png=W0.png_bytes(prefix)
        (output/"prefix24"/(stem+".png")).write_bytes(prefix_png)
        by_world.setdefault((seed,family),{})[(policy,"prefix24")]=prefix_png
        by_world[(seed,family)][(policy,"full96")]=full_png
        results.append({
            "id":stem,"seed":seed,"family":family,"policy":policy,
            "prefix24":before,"full96":after,
            "prefix24_png_sha256":digest(prefix_png),
            "original_full96_png_sha256":digest(full_png),
            "original_full96_json_sha256":digest(blob),
        })
    check(len(seen)==36 and len(by_world)==6, "incomplete source-fenced cohort")
    contacts=[]
    for seed,family in sorted(by_world):
        png=contact_sheet(list(POLICIES),by_world[(seed,family)])
        name=f"{family}-{seed}-prefix24-vs-full96.png"
        (output/"contact"/name).write_bytes(png)
        contacts.append({"name":name,"seed":seed,"family":family,
                         "sha256":digest(png)})
    # Frozen reviewer examples pin an actual failing/strong control, not a score gate.
    pick={(r["family"],r["policy"],r["seed"]):r for r in results}
    contour=pick[("greebles","contour",211)]["full96"]
    corridor=pick[("architecture","corridor",211)]["full96"]
    check(contour["unique_cells"]==96
          and contour["max_manhattan_from_start"]==9
          and (contour["bbox_width"],contour["bbox_height"])==(12,10),
          "independently reported contour spatial blind spot changed")
    check(corridor["max_manhattan_from_start"]==45
          and (corridor["bbox_width"],corridor["bbox_height"])==(23,39),
          "independently reported corridor reach changed")
    result={
        "version":VERSION,
        "kind":"original replayed W0 Python procedural gallery, no learned model/native",
        "model_optimizer_updates":0,
        "w0_commit":source.W0_COMMIT,"w0_source_sha256":source.W0_SHA256,
        "original_observed_v2_source_sha256":manifest["source_code_sha256"] if "source_code_sha256" in manifest else None,
        "original_manifest_sha256":digest((gallery/"manifest.json").read_bytes()),
        "world_seeds":list(source.GALLERY_WORLDS),
        "independent_world_count":len(source.GALLERY_WORLDS),
        "family_labels":list(source.FAMILIES),
        "policy_order":list(POLICIES),
        "contact_sheet_layout":{"columns":list(POLICIES),
                                "top_row":"exact original 24-action prefix",
                                "bottom_row":"original 96-action rollout",
                                "cell_pixels":[384,384]},
        "full96_png_json_pixel_replays":len(results),
        "prefix24_pngs":len(results),"contact_sheets":len(contacts),
        "contacts":contacts,"entries":results,
        "interpretation":"unique cells alone is not spatial novelty; report bbox/radius beside raw actions, revisits, branch/retrace, and prefix-24 comparisons",
    }
    (output/"spatial-report.json").write_bytes(source.canonical(result)+b"\n")
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--gallery",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    r=audit(args.gallery,args.output)
    print("PASS",r["full96_png_json_pixel_replays"],"full W0 PNG/json replays;",
          r["prefix24_pngs"],"original W0 prefix PNGs;",
          r["contact_sheets"],"unaltered-pixel paired contact sheets")
    for family,policy in (("greebles","contour"),("architecture","corridor"),
                          ("architecture","wall")):
        x=next(e for e in r["entries"] if e["seed"]==211
               and e["family"]==family and e["policy"]==policy)
        print("SPATIAL",family,policy,"seed=211",
              "prefix24",x["prefix24"],"full96",x["full96"])


if __name__=="__main__":
    main()
