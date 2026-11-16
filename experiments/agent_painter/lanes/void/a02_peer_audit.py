#!/usr/bin/env python3
"""Read-only original-byte VOID A02 vs public TRAIN/DEVELOPMENT and SPROCKET A07/A08.

Research witness only: neither FINAL, optimization, native rerun nor shared TRAIN admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

ZIP_SHA256 = {
    "void": "deeabf0c0a79b924bdc6092b4f9514b376ae5e3bdbe013571bea46bbd472c38a",
    "sprocket": "3a7397883b146a28da990f0a37b891897285c2c7c81cd9dd2aaa3895d0d196dd",
    "public": "50a9ad9b255b2a57180edf5558ca2b47dab9affd9ef9d2438f9011d648f81f39",
}
SPLIT_SHA256 = {
    "TRAIN": "5cb437db0663d15496adced3c255c32a5a274d99a69dd6113938b2dbc15ddbb1",
    "DEVELOPMENT": "8c73edd82d9289a78721ef7968ab2f939f858b666a106731709bc2d80b347572",
}
MANIFEST_SHA256 = "2fdef5a3639606fb9af68319f9747a1de1e9a573da5c6197a2dcbc96579cb6cc"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_zip(path: Path, key: str) -> zipfile.ZipFile:
    if not path.is_file() or path.is_symlink() or sha(path.read_bytes()) != ZIP_SHA256[key]:
        raise ValueError(f"{key}: missing or wrong original ZIP bytes")
    z = zipfile.ZipFile(path)
    # Public archive contains sealed FINAL: never testzip() or decompress it.
    # zipfile.read() validates CRC only for explicitly permitted named members.
    if key != "public" and z.testzip() is not None:
        raise ValueError(f"{key}: ZIP CRC failure")
    return z


def native_mask(z: zipfile.ZipFile, member: str) -> set[int]:
    raw = z.read(member)  # Exact named material readback, never FINAL.
    if len(raw) != 96 * 96 * 4:
        raise ValueError(f"{member}: wrong raw 96x96 size")
    cells = struct.unpack("<9216i", raw)
    if set(cells) - {0, 58}:
        raise ValueError(f"{member}: unexpected material ID")
    return {i for i, v in enumerate(cells) if v == 58}


def metrics(base: set[int], other: set[int]) -> dict:
    intersect = len(base & other)
    union = len(base | other)
    return {"material58_cells": len(other), "same_coordinate_intersection": intersect,
            "same_coordinate_iou": round(intersect / union, 5)}


def audit(void: Path, sprocket: Path, public: Path) -> dict:
    with checked_zip(void, "void") as v, checked_zip(sprocket, "sprocket") as s, checked_zip(public, "public") as p:
        manifest = p.read("corpus/admitted-corpus.json")
        if sha(manifest) != MANIFEST_SHA256:
            raise ValueError("original admitted manifest identity changed")
        info = json.loads(manifest)
        if info["source_sha"] != "633920373a3d406a2ede61ce4137c324f76be23b":
            raise ValueError("original native producer identity changed")
        base = native_mask(v, "void-v4-shaft-native/void-v4-A02-s104729.materials.i32le")
        twin = native_mask(v, "void-v4-shaft-native/void-v4-A02-s130363.materials.i32le")
        if base != twin or len(base) != 524:
            raise ValueError("A02 native seed or 524-cell custody mismatch")
        peers = []
        for split in ("TRAIN", "DEVELOPMENT"):
            raw = p.read(f"corpus/{split.lower()}.jsonl")
            if sha(raw) != SPLIT_SHA256[split]:
                raise ValueError(f"{split}: original public split changed")
            rows = [json.loads(line) for line in raw.splitlines()]
            if len(rows) != 12 or len({r["group_id"] for r in rows}) != 6:
                raise ValueError(f"{split}: public structural grouping changed")
            selections = [r for r in rows if r["family"] == "architecture" and r["seed"] == 104729]
            if len(selections) != 2:
                raise ValueError(f"{split}: expected exactly two architecture groups")
            for r in selections:
                member = f"corpus/evidence/{r['native_case_id']}.materials.i32le"
                m = native_mask(p, member)
                comps = r["program"]["components"]
                peer = {"name": r["group_id"], "split": split,
                        "room_count": sum(c["kind"] == "room" for c in comps),
                        "opening_sides": [o["side"] for c in comps if c["kind"] == "room" for o in c["openings"]]}
                peer.update(metrics(base, m))
                peers.append(peer)
        for name in ("A07", "A08"):
            m = native_mask(s, f"sprocket-v3-native/sprocket-{name}-s104729.materials.i32le")
            peers.append({"name": f"sprocket-{name}", "split": "UNADMITTED_PEER", **metrics(base, m)})
        return {"status": "READ_ONLY_SPATIAL_WITNESS", "zip_sha256": ZIP_SHA256,
                "public_manifest_sha256": MANIFEST_SHA256, "public_split_sha256": SPLIT_SHA256,
                "target": {"name": "void-v4-A02", "one_independent_composition": True,
                           "native_seed_views": 2, "native_seed_masks_identical": True,
                           "material58_cells": len(base)}, "neighbours": peers,
                "opened_splits": ["TRAIN", "DEVELOPMENT"], "final_targets_opened": False,
                "new_shared_train_groups": 0, "model_optimizer_updates": 0,
                "reviewer_global_independence_approved": False,
                "note": "Same-coordinate raster IoU is a limited spatial descriptor, never proof of independent semantics or native/model competence."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--void-zip", type=Path, required=True)
    ap.add_argument("--sprocket-zip", type=Path, required=True)
    ap.add_argument("--public-zip", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(audit(args.void_zip, args.sprocket_zip, args.public_zip), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
