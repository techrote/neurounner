"""Read-only D-2 DEVELOPMENT corpus morphology/recovery audit.

Independent material PNG decoding, record/custody checks and morphology metrics;
NOT an independent native readback and NOT a replacement for lab.replay().
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

VERSION = "d2.morphology-audit/1"
INK = bytes((232, 222, 199, 255))
EMPTY = bytes((24, 33, 43, 255))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def png_cells(blob):
    """Read exact, unfiltered, four-channel synthetic material PNGs; reject ambiguity."""
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    offset, data, dimensions = 8, bytearray(), None
    seen = []
    while offset < len(blob):
        assert offset + 12 <= len(blob)
        length = struct.unpack_from(">I", blob, offset)[0]
        tag = blob[offset + 4:offset + 8]
        end = offset + 12 + length
        assert end <= len(blob)
        payload = blob[offset + 8:offset + 8 + length]
        assert zlib.crc32(tag + payload) & 0xffffffff == struct.unpack_from(">I", blob, offset + 8 + length)[0]
        seen.append(tag)
        if tag == b"IHDR":
            assert len(payload) == 13 and dimensions is None
            w, h, depth, color, compression, filt, interlace = struct.unpack(">IIBBBBB", payload)
            assert 1 <= w <= 256 and 1 <= h <= 256 and (depth, color, compression, filt, interlace) == (8, 6, 0, 0, 0)
            dimensions = w, h
        elif tag == b"IDAT":
            assert dimensions is not None
            data.extend(payload)
        elif tag == b"IEND":
            assert payload == b"" and end == len(blob)
        else:
            raise ValueError("Unexpected PNG chunk")
        offset = end
    assert seen[0] == b"IHDR" and seen[-1] == b"IEND" and b"IDAT" in seen
    w, h = dimensions
    raw = zlib.decompress(bytes(data))
    assert len(raw) == h * (1 + 4 * w)
    cells = set()
    for y in range(h):
        row = raw[y * (1 + 4 * w):(y + 1) * (1 + 4 * w)]
        assert row[0] == 0
        for x in range(w):
            pixel = row[1 + x * 4:1 + (x + 1) * 4]
            assert pixel in (INK, EMPTY)
            if pixel == INK:
                cells.add((x, y))
    return (w, h), cells


def components(points):
    unseen = set(points)
    sizes = []
    while unseen:
        start = unseen.pop()
        queue = [start]
        for x, y in queue:
            for q in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if q in unseen:
                    unseen.remove(q)
                    queue.append(q)
        sizes.append(len(queue))
    sizes.sort(reverse=True)
    return sizes


def bbox(points):
    if not points:
        return None
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def audit_episode(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_bytes())
    assert manifest["qualification"] == "OFFLINE_SYNTHETIC_ONLY"
    assert manifest["native_replay_verified"] is False and manifest["model_decisions"] == 0
    assert manifest["provenance"]["native"] is None and manifest["provenance"]["model"] is None
    for name, meta in manifest["artifacts"].items():
        assert name not in ("manifest.json",) and "/" not in name and ".." not in name
        data = (folder / name).read_bytes()
        assert meta == {"bytes": len(data), "sha256": sha(data)}, name
    recipe = json.loads((folder / "recipe.json").read_bytes())
    assert recipe["version"] == "d2.offline/1"
    dimensions = tuple(recipe["canvas"])
    images = [png_cells((folder / f"{name}.png").read_bytes()) for name in ("before", "midpoint", "after")]
    assert all(size == dimensions for size, _ in images)
    initial, midpoint, final = (cells for _, cells in images)
    track = json.loads((folder / "trajectory.json").read_bytes())
    events = json.loads((folder / "events.json").read_bytes())
    geometry = json.loads((folder / "geometry.json").read_bytes())
    changed = initial ^ final
    # The actual synthetic per-step material count must be independently
    # checked against the before/after images; a net delta is insufficient.
    additions = final - initial
    removals = initial - final
    assert len(track) == manifest["summary"]["steps"]
    assert len(final) == manifest["summary"]["final_cells"]
    assert len(final) - len(initial) == manifest["summary"]["net_delta"]
    assert sum(e["kind"] == "STEP" for e in events) == len(track)
    assert [e["point"] for e in events if e["kind"] == "STEP"] == track
    assert all(e["sequence"] == i for i, e in enumerate(events))
    prev = initial
    # Reconstruct each synthetic material edit from logged admitted footprints.
    for event in events:
        if event["kind"] == "STEP":
            assert sha(canonical(sorted(map(list, prev)))) == event["material_before_sha256"]
            admitted = {tuple(p) for p in event["admitted"]}
            swept = {tuple(p) for p in event["swept"]}
            assert admitted <= swept
            after = (prev | admitted) if event["effect"] == "add" else (prev - admitted)
            assert len(after ^ prev) == event["actual_changed"]
            assert sha(canonical(sorted(map(list, after)))) == event["material_after_sha256"]
            prev = after
        elif event["kind"] in ("REVERSAL", "BRUSH_UP_SEEK_FUTURE"):
            assert event["material_before_sha256"] == event["material_after_sha256"]
        elif event["kind"] == "UNDO_MATERIAL_EDIT_UNAVAILABLE":
            assert event["executed"] is False and event["native_receipt"] is None
    assert prev == final
    cc = components(changed)
    final_cc = components(final)
    enters = [e for e in events if e["kind"] == "GUIDANCE_REENTRY"]
    exits = [e for e in events if e["kind"] == "GUIDANCE_EXIT"]
    seeks = [e for e in events if e["kind"] == "BRUSH_UP_SEEK_FUTURE"]
    assert len(enters) == len(seeks)
    assert all(e["virtual_offline_transfer"] and not e["continuous_physical_path_verified"] for e in seeks)
    # Measure actual subsequent deterministic actions unlocked by each reentry.
    leverage = []
    for seek in seeks:
        start = seek["sequence"]
        tail = (e for e in events[start + 1:] if e["kind"] in ("GUIDANCE_EXIT", "TRAPPED", "COVERAGE_END"))
        end = next(tail, None)
        end_seq = end["sequence"] if end else len(events)
        leverage.append(sum(e["kind"] == "STEP" for e in events[start + 1:end_seq]))
    # Check curve index skips and physical continuity separately from visit uniqueness.
    curve = geometry.get("ordered_curve", [])
    if curve:
        assert len(curve) == len({tuple(p) for p in curve})
        assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(curve, curve[1:]))
    return dict(
        family=recipe["family"], recipe_sha256=sha(canonical(recipe)),
        initial_cells=len(initial), final_cells=len(final), added_cells=len(additions),
        removed_cells=len(removals), changed_cells=len(changed),
        changed_bbox=bbox(changed), changed_component_count=len(cc),
        largest_changed_component=cc[0] if cc else 0,
        final_component_count=len(final_cc),
        largest_final_component=final_cc[0] if final_cc else 0,
        changed_fraction=round(len(changed) / (dimensions[0] * dimensions[1]), 6),
        strokes=len(track), zero_change_strokes=sum(e["kind"] == "STEP" and e["actual_changed"] == 0 for e in events),
        reversals=sum(e["kind"] == "REVERSAL" for e in events),
        guidance_exits=len(exits), virtual_seeks=len(seeks), reentries=len(enters),
        reentry_followup_steps=leverage, mean_reentry_leverage=round(sum(leverage)/len(leverage), 3) if leverage else None,
        trapped=sum(e["kind"] == "TRAPPED" for e in events),
        source_material_sha256=sha((folder / "before.png").read_bytes()),
        final_material_sha256=sha((folder / "after.png").read_bytes()),
        native_material_verified=False,
    )


def audit_corpus(root):
    root = Path(root)
    index = json.loads((root / "index.json").read_bytes())
    assert index["qualification"] == "DEVELOPMENT_SYNTHETIC_ONLY"
    assert index["native_replay_verified"] is False
    cases = {}
    for case in index["cases"]:
        name = case["id"]
        assert name not in cases and name.replace("-", "").isalnum()
        target = root / name
        assert sha((target / "manifest.json").read_bytes()) == case["manifest_sha256"]
        assert sha((target / "recipe.json").read_bytes()) == case["recipe_sha256"]
        metrics = audit_episode(target)
        assert metrics["recipe_sha256"] == case["recipe_sha256"]
        cases[name] = metrics
    report = dict(version=VERSION, scope="DEVELOPMENT_SYNTHETIC_ONLY",
                  native_replay_verified=False, model_decisions=0,
                  input_index_sha256=sha((root / "index.json").read_bytes()),
                  cases=cases)
    report["report_sha256"] = sha(canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit_corpus(args.corpus)
    output = canonical(result) + b"\n"
    if args.out:
        args.out.write_bytes(output)
    else:
        print(output.decode(), end="")
