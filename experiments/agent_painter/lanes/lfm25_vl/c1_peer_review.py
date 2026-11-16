#!/usr/bin/env python3
"""Read-only C1 Greebles near-neighbour witness, not a dataset admission gate.

Compares pinned *hand-authored source programs* (panel boxes, not pixels) from
LFM/VOID/SPROCKET. Does not open FINAL, train, run Art or infer model quality.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

# Git blob object IDs are for the exact original source bytes at pinned commits.
BLOBS = {
    "lfm": "226ce959c68a9e2649e88fe15c10713437cf82be",
    "void": "0e594de0b683af2547ca3c09e978cf61a349392e",
    "sprocket": "b3ac98a8cd06158b39a2f58776fab845046f245f",
}
PRODUCERS = {
    "lfm": "5fd5e33aad38296155c7bb47cf37a07f0ee07946",
    "void": "ea056e395295f1fd3f8ed64c039b746555060d61",
    "sprocket": "917425fee7c2b4a5ede2d9d8bbf085d4b4091d63",
}
C1_ID = "lfm-c1-six-bay-service-chase"
PEERS = ("void-v3-G01", "void-v3-G02", "sprocket-G08", "sprocket-G09")
CHASEx = (40, 56)
CANVAS = 96


def verified_bytes(path: Path, expected_blob: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked source: " + str(path))
    b = path.read_bytes()
    if hashlib.sha1(b"blob " + str(len(b)).encode("ascii") + b"\0" + b).hexdigest() != expected_blob:
        raise ValueError("foreign or modified source blob: " + str(path))
    return b


def sprocket_studies(raw: bytes) -> dict:
    """Literal-only AST data extraction: never execute arbitrary peer Python."""
    tree = ast.parse(raw.decode("utf-8"))
    matches = [n.value for n in tree.body if isinstance(n, ast.Assign)
               and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == "STUDIES"]
    if len(matches) != 1:
        raise ValueError("pinned SPROCKET STUDIES literal missing")
    return ast.literal_eval(matches[0])


def panel_boxes(program: dict) -> list[tuple[int, int, int, int]]:
    if program.get("family") != "greebles":
        raise ValueError("not a Greebles program")
    components = program.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("missing panels")
    boxes = []
    for c in components:
        if c.get("kind") != "panel" or not isinstance(c.get("box"), list):
            raise ValueError("non-panel component")
        x, y, w, h = c["box"]
        if (any(type(v) is not int for v in (x, y, w, h))
                or not (0 <= x < CANVAS and 0 <= y < CANVAS)
                or not (0 < w <= CANVAS - x and 0 < h <= CANVAS - y)):
            raise ValueError("invalid panel geometry")
        boxes.append((x, y, w, h))
    return boxes


def box_mask(boxes: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    # Source panel bounding boxes only; not an inference about drawn native pixels.
    return {(x, y) for x0, y0, w, h in boxes
            for x in range(x0, x0 + w) for y in range(y0, y0 + h)}


def paired_tiers(boxes: list[tuple[int, int, int, int]]) -> bool:
    if len(boxes) != 6:
        return False
    left = sorted((b for b in boxes if b[0] + b[2] <= CHASEx[0]), key=lambda b: b[1])
    right = sorted((b for b in boxes if b[0] >= CHASEx[1]), key=lambda b: b[1])
    if len(left) != 3 or len(right) != 3:
        return False
    for bank in (left, right):
        if any(bank[i][1] + bank[i][3] + 4 > bank[i+1][1] for i in range(2)):
            return False
    return all(abs(2*l[1] + l[3] - 2*r[1] - r[3]) <= 16
               for l, r in zip(left, right))


def measures(boxes: list[tuple[int, int, int, int]]) -> dict:
    mask = box_mask(boxes)
    return {
        "panels": len(boxes),
        "panel_box_pixels": len(mask),
        "full_height_central_chase_clear": not any(CHASEx[0] <= x < CHASEx[1] for x, _ in mask),
        "paired_three_tiers": paired_tiers(boxes),
        "cross_chase_panel_count": sum(x < CHASEx[1] and x + w > CHASEx[0]
                                       for x, y, w, h in boxes),
    }


def overlap(a: list[tuple[int, int, int, int]],
            b: list[tuple[int, int, int, int]]) -> dict:
    ma, mb = box_mask(a), box_mask(b)
    intersection = len(ma & mb)
    return {"box_intersection": intersection, "box_union": len(ma | mb),
            "box_iou": round(intersection / len(ma | mb), 6)}


def compare(lfm: dict, void: dict, sprocket: dict) -> dict:
    if (lfm.get("status") != "PROPOSED_NOT_TRAIN_NOT_NATIVE_ADMITTED"
            or void.get("state") != "unadmitted-research"):
        raise ValueError("research-status contract changed")
    candidates = [x for x in lfm.get("proposals", []) if x.get("id") == C1_ID]
    if len(candidates) != 1:
        raise ValueError("C1 source proposal missing")
    c1 = panel_boxes(candidates[0]["program"])
    c1m = measures(c1)
    if c1m["panels"] != 6 or not c1m["full_height_central_chase_clear"] or not c1m["paired_three_tiers"]:
        raise ValueError("C1 motif no longer meets its specific composition witness")
    v = {x["id"]: x for x in void.get("candidates", [])}
    if any(v[id]["training_admitted"] is not False for id in PEERS[:2]):
        raise ValueError("peer research label changed")
    peer_programs = {
        "void-v3-G01": v["void-v3-G01"]["program"],
        "void-v3-G02": v["void-v3-G02"]["program"],
        "sprocket-G08": sprocket["G08"]["program"],
        "sprocket-G09": sprocket["G09"]["program"],
    }
    peers = {}
    for id in PEERS:
        boxes = panel_boxes(peer_programs[id])
        m = measures(boxes)
        if m["paired_three_tiers"] and m["full_height_central_chase_clear"]:
            raise ValueError("unexpected same peer motif requires human re-review: " + id)
        peers[id] = {**m, **overlap(c1, boxes)}
    return {
        "status": "SOURCE_BOX_MOTIF_DIFFERENCE_ONLY_NOT_GLOBAL_TRAIN_ADMISSION",
        "c1": c1m, "comparisons": peers,
        "review_scope": "four pinned research Greebles programs only; not all visible split or native rasters",
        "native_positive_from_this_audit": 0, "train_groups_admitted": 0,
        "optimizer_updates": 0, "model_calls": 0, "final_targets_opened": False,
    }


def screen(paths: dict[str, Path]) -> dict:
    sources = {k: verified_bytes(paths[k], BLOBS[k]) for k in BLOBS}
    result = compare(json.loads(sources["lfm"]), json.loads(sources["void"]),
                     sprocket_studies(sources["sprocket"]))
    result["sources"] = {k: {"commit": PRODUCERS[k], "git_blob": BLOBS[k],
                            "sha256": hashlib.sha256(sources[k]).hexdigest()}
                         for k in BLOBS}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("lfm", "void", "sprocket", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    args = vars(p.parse_args())
    output = args.pop("output")
    if output.exists():
        raise ValueError("refusing to overwrite existing review receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = screen(args)
    except Exception as e:
        report = {"status": "FAIL_NO_ADMISSION", "error": repr(e),
                  "train_groups_admitted": 0, "final_targets_opened": False}
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "peers": len(report["comparisons"]), "train": 0}))


if __name__ == "__main__":
    main()
