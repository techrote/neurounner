#!/usr/bin/env python3
"""Read-only original native-material neighbour witness for SMOL review.

Two seeded 96x96 Art research designs versus independently hosted peers.
Raster IoU is descriptive only; it never proves structural split independence,
TRAIN admission, movement playability, model quality, or new native execution.
The fixed source artifacts are not rewritten, repaired, or promoted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

SEEDS = (104729, 130363)
ZIP_SHA256 = {
    "smol": "71ddc40478e06e3350f05eecd5e16ed77c082300f066be97bbccf8afb2370ff1",
    "sprocket": "3a7397883b146a28da990f0a37b891897285c2c7c81cd9dd2aaa3895d0d196dd",
    "void": "d8f1c1dbce65ef14d392a7bd557bd89e8af21563da14d4a5a61b84386285944b",
}
KNOWN_CELLS = {
    "G06": (547, 535),
    "P01": (596, 596),
    "A01": (704, 704),
    "G09": (847, 900),
    "A02-v5": (662, 662),
}


def check_zip(path: Path, family: str) -> zipfile.ZipFile:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{family}: missing or symlink ZIP")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != ZIP_SHA256[family]:
        raise ValueError(f"{family}: original archive SHA256 mismatch")
    archive = zipfile.ZipFile(path)
    if archive.testzip() is not None:
        archive.close()
        raise ValueError(f"{family}: original ZIP CRC failure")
    return archive


def find_material(archive: zipfile.ZipFile, marker: str, seed: int) -> tuple[str, set[int]]:
    suffix = f"-s{seed}.materials.i32le"
    members = [n for n in archive.namelist()
               if n.endswith(suffix) and marker in Path(n).name]
    if len(members) != 1:
        raise ValueError(f"{marker}/{seed}: expected exactly one original material readback, got {len(members)}")
    member = members[0]
    raw = archive.read(member)
    if len(raw) != 96 * 96 * 4:
        raise ValueError(f"{member}: wrong 96x96 int32 material size")
    values = struct.unpack("<9216i", raw)
    if set(values) - {0, 58}:
        raise ValueError(f"{member}: unexpected material domain")
    return member, {i for i, value in enumerate(values) if value == 58}


def compare(a: set[int], b: set[int]) -> dict:
    common = len(a & b)
    total = len(a | b)
    return {
        "intersection_cells": common,
        "union_cells": total,
        "same_coordinate_iou": round(common / total, 6) if total else 1.0,
        "only_first_cells": len(a - b),
        "only_second_cells": len(b - a),
    }


def collect(archive: zipfile.ZipFile, marker: str, expected: tuple[int, int]) -> tuple[list[set[int]], list[dict]]:
    masks, evidence = [], []
    for i, seed in enumerate(SEEDS):
        member, mask = find_material(archive, marker, seed)
        if len(mask) != expected[i]:
            raise ValueError(f"{marker}/{seed}: native cell count differs from original receipt")
        masks.append(mask)
        evidence.append({"seed": seed, "original_member": member, "material58_cells": len(mask)})
    return masks, evidence


def audit(smol: Path, sprocket: Path, void: Path) -> dict:
    with check_zip(smol, "smol") as s, check_zip(sprocket, "sprocket") as p, check_zip(void, "void") as v:
        group = {
            "G06": collect(s, "smol_g06_stepped_absence", KNOWN_CELLS["G06"]),
            "P01": collect(s, "smol_p01_nested_branch_tree", KNOWN_CELLS["P01"]),
            "A01": collect(s, "smol_a01_unequal_alcove_hall", KNOWN_CELLS["A01"]),
            "G09": collect(p, "sprocket-G09", KNOWN_CELLS["G09"]),
            "A02-v5": collect(v, "void-v5-A02", KNOWN_CELLS["A02-v5"]),
        }
        pairs = []
        for target, peer in (("G06", "G09"), ("A01", "A02-v5")):
            for index, seed in enumerate(SEEDS):
                pairs.append({
                    "target": target, "peer": peer, "seed": seed,
                    **compare(group[target][0][index], group[peer][0][index]),
                })
        within = {
            k: compare(v[0][0], v[0][1])
            for k, v in group.items()
        }
        return {
            "schema": "cybersand.smol-native-peer-mask-witness/1",
            "status": "READ_ONLY_ORIGINAL_NATIVE_MASK_WITNESS",
            "purpose": "cross-lane spatial neighbour screening only",
            "original_zip_sha256": ZIP_SHA256,
            "seeds": list(SEEDS),
            "material_receipts": {k: v[1] for k, v in group.items()},
            "same_seed_peer_pairs": pairs,
            "within_group_seed_variants": within,
            "independent_structures": 5,
            "physical_seed_views_read": 10,
            "new_native_executions": 0,
            "new_optimizer_steps": 0,
            "new_model_generated_views": 0,
            "new_shared_train_admissions": 0,
            "final_targets_opened": False,
            "global_structural_independence_proven": False,
            "shared_train_admission": "HOLD",
            "note": "Same-coordinate raster overlap cannot establish semantic or compositional independence; review original PNGs and split neighbours.",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smol", type=Path, required=True)
    parser.add_argument("--sprocket", type=Path, required=True)
    parser.add_argument("--void", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError("refuse overwrite of immutable original-byte review")
    result = audit(a.smol, a.sprocket, a.void)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"],
                      "native_source_archives": 3,
                      "original_seed_views_read": result["physical_seed_views_read"],
                      "peer_comparisons": len(result["same_seed_peer_pairs"]),
                      "TRAIN": 0, "model_steps": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
