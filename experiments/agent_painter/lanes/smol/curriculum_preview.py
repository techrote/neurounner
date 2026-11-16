#!/usr/bin/env python3
"""Review-only compiler raster gallery for three proposed SMOL structures.

NOT native Art output, NOT material readback, NOT model inference, NOT TRAIN.
Only runs after the frozen original TRAIN/DEVELOPMENT source screen passes.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import struct
import zlib

from curriculum_preflight import MANIFEST, SEEDS, compiler, run, sha

SIZE = 96
LABEL = "COMPILER RASTER ONLY — not native painting, not a trained model, not admitted TRAIN"


def png_bytes(grid: list[int], *, size: int = SIZE) -> bytes:
    """Deterministic lossless RGB PNG. One image pixel equals one compiled cell."""
    if len(grid) != size * size or any(type(x) is not int or x not in (0, 58) for x in grid):
        raise ValueError("invalid binary 0/58 compiler raster; refuse preview")
    raw = bytearray()
    for y in range(size):
        raw.append(0)  # PNG filter: none
        for x in grid[y*size:(y+1)*size]:
            raw.extend((224, 224, 224) if x == 58 else (28, 30, 34))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff))
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(bytes(raw), level=9)) + chunk(b"IEND", b""))


def create(corpus: Path, proposals: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("refuse overwrite of prospective review gallery")
    screen = run(corpus, proposals)
    if (screen["final_targets_opened"] is not False or screen["native_execution"] is not False
            or screen["admitted_teacher_rows"] != 0 or len(screen["candidates"]) != 3):
        raise ValueError("not a proposal-only, original-source-qualified screen")
    source = json.loads(proposals.read_text())
    output.mkdir(parents=True)
    candidates = []
    panels = []
    for item in source["proposals"]:
        known = next(x for x in screen["candidates"] if x["id"] == item["id"])
        files = []
        for seed, expected in zip(SEEDS, known["compiled_seed_variants"], strict=True):
            compiled = compiler.compile_program(item["program"], seed=seed,
                                                focus=(0, 0, 96, 96), material=58,
                                                allowed_materials=(58,))
            if (compiled["operations_sha256"] != expected["operations_sha256"]
                    or compiled["program_sha256"] != expected["program_sha256"]
                    or compiled["unique_cells"] != expected["unique_cells"]):
                raise ValueError("compiler raster differs from source-screen receipt")
            grid = compiler.rasterize_operations(compiled["operations"], (0, 0, SIZE, SIZE))
            if sum(x == 58 for x in grid) != compiled["unique_cells"]:
                raise ValueError("preview cells disagree with strict compiler")
            filename = f'{item["id"]}-seed{seed}.png'
            data = png_bytes(grid)
            (output / filename).write_bytes(data)
            files.append({"seed": seed, "file": filename,
                          "png_sha256": hashlib.sha256(data).hexdigest(),
                          "material58_cells": compiled["unique_cells"],
                          "operations_sha256": compiled["operations_sha256"]})
        candidates.append({"id": item["id"], "family": item["family"],
                           "request": item["request"], "witness": item["witness"],
                           "files": files, "native_admitted": False})
        pics = "".join(f'<figure><img src="{escape(f["file"])}" width="384" height="384" '
                       f'alt="96 by 96 compiler-only {escape(item["family"])} raster">'
                       f'<figcaption>Seed {f["seed"]}: {f["material58_cells"]} proposed cells</figcaption>'
                       f'</figure>' for f in files)
        panels.append(f'<section><h2>{escape(item["id"])} — {escape(item["family"])}</h2>'
                      f'<p>{escape(item["request"])}</p><p><b>Witness:</b> '
                      f'{escape(item["witness"])}</p><div class="views">{pics}</div></section>')
    manifest = {"schema": "cybersand.agent-painter.smol-compiler-review-gallery/1",
                "claim": LABEL, "native_source": screen["native_source_sha"],
                "original_native_manifest_sha256": MANIFEST,
                "compiler_sha256": screen["compiler_sha256"],
                "proposals_sha256": sha(proposals),
                "final_targets_opened": False, "actual_model_outputs": 0,
                "actual_native_positive_views": 0, "shared_train_groups_admitted": 0,
                "candidates": candidates}
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width">'
            '<title>SMOL proposed compiler rasters — NOT NATIVE</title>'
            '<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;'
            'padding:0 1rem;background:#151719;color:#eee}section{border-top:1px solid #555;'
            'padding:1rem 0}.views{display:flex;flex-wrap:wrap;gap:1rem}figure{margin:0}'
            'img{image-rendering:pixelated;max-width:100%;height:auto;border:1px solid #999}'
            'figcaption{color:#bbb}strong{color:#fff}</style>'
            f'<h1>SMOL source-v2 proposal review</h1><p><strong>{escape(LABEL)}</strong></p>'
            '<p>Three hand-authored strict-v2 proposals, each compiled with two fixed host '
            'seeds. Seed views are NOT independent structural samples. These PNGs are '
            'deterministic compiler-oracle rasters, NOT Godot Art material readback, NOT '
            'original native PNGs, and NOT outputs from a trained painter. '
            'Independent novelty review and native Art admission remain required.</p>'
            + "".join(panels) + '</html>')
    (output / "index.html").write_text(html, encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--proposals", type=Path,
                   default=Path(__file__).with_name("curriculum_proposals_v2.json"))
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    result = create(a.corpus, a.proposals, a.output)
    print(json.dumps({"status": "PROPOSAL_PREVIEW_ONLY", "families": len(result["candidates"]),
                      "images": sum(len(c["files"]) for c in result["candidates"]),
                      "native_views": 0, "TRAIN_groups": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
