#!/usr/bin/env python3
"""SPROCKET source-pinned, read-only peer/visible-split semantic witness v1.

Necessary differences are NOT global novelty certification, native admission,
model inference or authority to alter TRAIN. Never access sealed FINAL.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha1, sha256
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

LANE = Path(__file__).resolve().parent
ROOT = LANE.parents[1]
sys.path.insert(0, str(LANE))
sys.path.insert(0, str(ROOT))

from threefamily_v3 import STUDIES, program_for
from proposal_pipework import (
    NATIVE_SOURCE, NATIVE_MANIFEST_SHA, ORIGINAL_TRAIN_SHA, ORIGINAL_DEV_SHA,
    ORIGINAL_COMPILER_SHA, file_sha, public_split_signatures,
)

# Every peer file is read-only, source- and Git-blob-pinned; neither a peer
# branch name nor current main can silently substitute its candidate program.
PEERS = {
    "void_v3": {
        "source": "19c344e0cfa9c82617925bd3eccf98e66be6d064",
        "path": "experiments/agent_painter/lanes/void/v3_research_proposals.json",
        "blob": "0e594de0b683af2547ca3c09e978cf61a349392e",
    },
    "lfm_contrastive": {
        "source": "3d4aecc72366f30630100151f23a827f73bf4003",
        "path": "experiments/agent_painter/lanes/lfm25_vl/contrastive_proposals_v1.json",
        "blob": "226ce959c68a9e2649e88fe15c10713437cf82be",
    },
    "smol_v2": {
        "source": "a6da3faad83dce35744ddd4e4ea91c0de7cd228b",
        "path": "experiments/agent_painter/lanes/smol/curriculum_proposals_v2.json",
        "blob": "0b0adae3c02d1ccc9bda2cdca393f272ca7ecf63",
    },
    "void_v4": {
        "source": "77919468a896ecf32c65c4ccbc8957b9e7a3953b",
        "path": "experiments/agent_painter/lanes/void/v4_shaft_proposals.json",
        "blob": "12a78c4583580b0526e1ce70eaa0fa4d5dd1078f",
    },
}
SPROCKET_V3_BLOB = "b3ac98a8cd06158b39a2f58776fab845046f245f"
VISIBLE_SPLITS = (("TRAIN", ORIGINAL_TRAIN_SHA), ("DEVELOPMENT", ORIGINAL_DEV_SHA))

def git_blob_sha(raw: bytes) -> str:
    return sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()

def checked_pinned_json(raw: bytes, expected_blob: str) -> dict:
    if git_blob_sha(raw) != expected_blob:
        raise ValueError("pinned peer git blob drift; refuse silent replacement")
    parsed = json.loads(raw)
    if type(parsed) is not dict:
        raise ValueError("pinned peer document must be object")
    return parsed

def load_peers() -> tuple[dict, dict]:
    peers, custody = {}, {}
    for key, item in PEERS.items():
        url = ("https://raw.githubusercontent.com/techrote/cybersand/"
               + item["source"] + "/" + item["path"])
        req = Request(url, headers={"User-Agent": "cybersand-sprocket-readonly-witness"})
        with urlopen(req, timeout=35) as response:
            raw = response.read(350_001)
        if len(raw) > 350_000:
            raise ValueError("peer document unexpectedly large")
        peers[key] = checked_pinned_json(raw, item["blob"])
        custody[key] = {
            "source_sha": item["source"], "path": item["path"],
            "git_blob_sha1": item["blob"], "sha256": sha256(raw).hexdigest(),
        }
    return peers, custody

def panels(program: dict) -> dict:
    assert program["family"] == "greebles"
    boxes = [c["box"] for c in program["components"] if c["kind"] == "panel"]
    details = sorted(c["detail"] for c in program["components"] if c["kind"] == "panel")
    ordered = sorted(boxes, key=lambda b: (b[1], b[0]))
    counts = []
    for x, y, width, height in ordered:
        if not counts or y - counts[-1][0] > 8:
            counts.append([y, 1])
        else:
            counts[-1][1] += 1
    # Aspect ratios stated as numerators/denominators, not rounded floats.
    return {
        "count": len(boxes), "detail_vocabulary": details,
        "top_y_bands": [v[1] for v in counts],
        "widths": [b[2] for b in ordered],
        "heights": [b[3] for b in ordered],
        "wide_panels_w_gt_2h": sum(b[2] > 2 * b[3] for b in boxes),
        "narrow_panels_w_le_2h": sum(b[2] <= 2 * b[3] for b in boxes),
        "ordered_center_x_twice": [2*b[0]+b[2] for b in ordered],
        "ordered_center_y_twice": [2*b[1]+b[3] for b in ordered],
    }

def architecture(program: dict) -> dict:
    assert program["family"] == "architecture"
    components = program["components"]
    rooms = [c for c in components if c["kind"] == "room"]
    platforms = [c for c in components if c["kind"] == "platform"]
    # Largest room supplies only a reproducible spatial frame, not a gameplay
    # or room-adjacency proof.
    main = max(rooms, key=lambda c: c["box"][2] * c["box"][3]) if rooms else None
    relations = []
    if main:
        top, bottom = main["box"][1], main["box"][1] + main["box"][3]
        for platform in platforms:
            y, h = platform["box"][1], platform["box"][3]
            relations.append("above" if y+h <= top else "below" if y >= bottom
                             else "within_vertical_room_extent")
    req = program.get("requirements", {})
    route = req.get("route")
    if route and len(route) == 2:
        axis = ("vertical" if route[0][0] == route[1][0] else
                "horizontal" if route[0][1] == route[1][1] else "diagonal")
    else:
        axis = "not_requested"
    return {
        "room_count": len(rooms), "room_boxes": [c["box"] for c in rooms],
        "opening_sides": dict(sorted(Counter(
            o["side"] for c in rooms for o in c["openings"]).items())),
        "component_kinds": dict(sorted(Counter(c["kind"] for c in components).items())),
        "platform_main_room_vertical_relations": sorted(relations),
        "route_axis": axis, "clearance": req.get("clearance"),
    }

def features(program: dict) -> dict:
    if program["family"] == "greebles":
        return panels(program)
    if program["family"] == "architecture":
        return architecture(program)
    return {"family": program["family"]}

def candidate(peers: dict, key: str, identifier: str) -> dict:
    sequence = (peers[key]["proposals"] if key in ("lfm_contrastive", "smol_v2")
                else peers[key]["candidates"])
    result = [x for x in sequence if x["id"] == identifier]
    if len(result) != 1:
        raise ValueError(f"{key}: missing or ambiguous candidate {identifier}")
    return result[0]["program"]

def visible_groups(corpus: Path) -> list[dict]:
    # Existing strict screen first verifies native manifest, exact original
    # byte hashes and 12+12 rows; it never touches FINAL.
    public_split_signatures(corpus)
    seen = set()
    result = []
    for name, expected in VISIBLE_SPLITS:
        manifest = json.loads((corpus / "admitted-corpus.json").read_text())
        entry = manifest["split_files"][name]
        path = (corpus / entry["path"]).resolve()
        if not path.is_relative_to(corpus.resolve()) or file_sha(path) != expected:
            raise ValueError("original public split path/hash drift")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            key = (name, row["structural_group"])
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "split": name, "structural_group": row["structural_group"],
                "family": row["program"]["family"],
                "features": features(row["program"]),
            })
    if len(result) != 12 or Counter(x["split"] for x in result) != Counter({"TRAIN":6,"DEVELOPMENT":6}):
        raise ValueError("public visible groups count drift")
    return result

def evaluate(corpus: Path, peers: dict, custody: dict) -> dict:
    if git_blob_sha((LANE / "threefamily_v3.py").read_bytes()) != SPROCKET_V3_BLOB:
        raise ValueError("source-pinned SPROCKET v3 programs changed")
    if file_sha(ROOT / "compiler.py") != ORIGINAL_COMPILER_SHA:
        raise ValueError("original strict compiler changed")
    if any(item.get("original_native_source") != NATIVE_SOURCE for item in peers.values()):
        raise ValueError("peer source provenance mismatch")
    sources = {
        "SPROCKET_G08": program_for("G08"), "SPROCKET_G09": program_for("G09"),
        "SPROCKET_A07": program_for("A07"), "SPROCKET_A08": program_for("A08"),
        "VOID_G01": candidate(peers, "void_v3", "void-v3-G01"),
        "VOID_A01": candidate(peers, "void_v3", "void-v3-A01"),
        "VOID_A02": candidate(peers, "void_v4", "void-v4-A02"),
        "LFM_C3": candidate(peers, "lfm_contrastive", "lfm-c3-gantry-room-and-underdeck-stairs"),
        "SMOL_G06": candidate(peers, "smol_v2", "smol_g06_stepped_absence"),
        "SMOL_A01": candidate(peers, "smol_v2", "smol_a01_unequal_alcove_hall"),
    }
    witness = {key: features(prog) for key, prog in sources.items()}
    g08, v01 = witness["SPROCKET_G08"], witness["VOID_G01"]
    g09, smol = witness["SPROCKET_G09"], witness["SMOL_G06"]
    a07, lfm = witness["SPROCKET_A07"], witness["LFM_C3"]
    a08, va01 = witness["SPROCKET_A08"], witness["VOID_A01"]
    va02 = witness["VOID_A02"]
    if not (g08["count"] == v01["count"] == 5 and
            g08["detail_vocabulary"] == v01["detail_vocabulary"] and
            g08["top_y_bands"] == [2, 2, 1] and v01["top_y_bands"] == [3, 2]):
        raise ValueError("G08/VOID G01 source witness drift")
    if not (g09["count"] == smol["count"] == 3 and
            g09["wide_panels_w_gt_2h"] == 3 and smol["wide_panels_w_gt_2h"] == 0):
        raise ValueError("G09/SMOL G06 organization witness drift")
    if not (a07["room_count"] == lfm["room_count"] == 1 and
            a07["component_kinds"]["platform"] == lfm["component_kinds"]["platform"] == 2 and
            a07["component_kinds"]["support"] == lfm["component_kinds"]["support"] == 2 and
            a07["platform_main_room_vertical_relations"] == ["within_vertical_room_extent"] * 2 and
            lfm["platform_main_room_vertical_relations"] == ["above", "above"] and
            lfm["component_kinds"].get("stairs") == 1):
        raise ValueError("A07/LFM C3 spatial support witness drift")
    if not (a08["room_count"] == 3 and va01["room_count"] == 2 and
            a08["route_axis"] == va01["route_axis"] == "horizontal" and
            a08["opening_sides"] == {"left":3, "right":3}):
        raise ValueError("A08/VOID A01 near-alias witness drift")
    if not (a07["route_axis"] == "horizontal" and va02["route_axis"] == "vertical" and
            va02["room_count"] == 1 and va02["component_kinds"]["platform"] == 3):
        raise ValueError("A07/VOID A02 cross-orientation witness drift")
    return {
        "schema": "cybersand.sprocket-pinned-peer-visible-semantic-witness/1",
        "status": "READ_ONLY_NECESSARY_DIFFERENCES_GLOBAL_REVIEW_HOLD",
        "sprocket_original_source_sha": "f564e1e441953fad04389e3a859783aa4c70ae1f",
        "sprocket_v3_program_blob": SPROCKET_V3_BLOB,
        "original_native_source_sha": NATIVE_SOURCE,
        "original_native_manifest_sha256": NATIVE_MANIFEST_SHA,
        "original_train_sha256": ORIGINAL_TRAIN_SHA,
        "original_development_sha256": ORIGINAL_DEV_SHA,
        "original_compiler_sha256": ORIGINAL_COMPILER_SHA,
        "pinned_peer_sources": custody,
        "comparisons": [
            {"a":"SPROCKET_G08","b":"VOID_G01","finding":"same-five-panels-detail-vocabulary-different-2-2-1-vs-3-2-row-hierarchy","review":"NEAR_ALIAS_HOLD"},
            {"a":"SPROCKET_G09","b":"SMOL_G06","finding":"both-three-panels-all-wide-bands-vs-none-wide-small-cassettes","review":"GLOBAL_REVIEW_PENDING"},
            {"a":"SPROCKET_A07","b":"LFM_C3","finding":"same-one-room-two-decks-two-supports-but-interior-vs-above-roof-and-stairs","review":"NEAR_ALIAS_HOLD"},
            {"a":"SPROCKET_A08","b":"VOID_A01","finding":"three-vs-two-side-door-rooms-same-horizontal-facing-door-motif","review":"QUARANTINE_RECOMMENDED"},
            {"a":"SPROCKET_A07","b":"VOID_A02","finding":"horizontal-two-interior-decks-vs-vertical-three-deck-shaft","review":"DISTINCT_WITNESS_NOT_TRAIN_APPROVAL"},
        ],
        "program_features": witness,
        "original_visible_groups": visible_groups(corpus),
        "opened_partitions": ["TRAIN", "DEVELOPMENT"],
        "final_targets_opened": False,
        "new_native_views": 0, "model_generated_views": 0,
        "optimizer_updates": 0, "shared_train_groups_admitted": 0,
        "reviewer_approved": False,
        "next_gate": "Shepherd and independent reviewer must decide semantic near-alias disposition, global IDs and shared TRAIN admission; this report cannot authorize optimization.",
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError("refuse overwrite of peer witness report")
    peers, custody = load_peers()
    report = evaluate(args.corpus, peers, custody)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "cross_peer_pairs": len(report["comparisons"]),
        "original_visible_groups": len(report["original_visible_groups"]),
        "new_train_groups": 0, "new_optimizer_steps": 0,
    }, sort_keys=True))

if __name__ == "__main__":
    main()
