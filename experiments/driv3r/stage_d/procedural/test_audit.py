"""Independent synthetic morphology readback and integrity checks (stdlib only)."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from audit import audit_corpus, audit_episode, canonical, components, png_cells, sha, INK, EMPTY


def png_for(w, h, cells):
    scan = bytearray()
    for y in range(h):
        scan.append(0)
        for x in range(w):
            scan.extend(INK if (x, y) in cells else EMPTY)
    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(scan))) + chunk(b"IEND", b"")


def fixture(root):
    folder = Path(root) / "probe"
    folder.mkdir()
    recipe = {"version": "d2.offline/1", "family": "branching_growth", "seed": 3, "canvas": [4, 4]}
    p = (1, 1)
    prev = set()
    nxt = {p}
    event = dict(sequence=0, kind="STEP", point=[1, 1], effect="add",
                 admitted=[[1, 1]], swept=[[1, 1]], brush_radius=0,
                 actual_changed=1, material_before_sha256=sha(canonical([])),
                 material_after_sha256=sha(canonical([[1, 1]])), native_receipt=None)
    files = {
        "recipe.json": canonical(recipe),
        "events.json": canonical([event]),
        "trajectory.json": canonical([[1, 1]]),
        "geometry.json": canonical({}),
        "before.png": png_for(4, 4, prev),
        "midpoint.png": png_for(4, 4, nxt),
        "after.png": png_for(4, 4, nxt),
    }
    for name, data in files.items():
        (folder / name).write_bytes(data)
    manifest = {"qualification": "OFFLINE_SYNTHETIC_ONLY", "native_replay_verified": False,
                "model_decisions": 0, "provenance": {"native": None, "model": None},
                "summary": {"steps": 1, "final_cells": 1, "net_delta": 1},
                "artifacts": {name: {"bytes": len(data), "sha256": sha(data)} for name, data in files.items()}}
    (folder / "manifest.json").write_bytes(canonical(manifest))
    index = {"qualification": "DEVELOPMENT_SYNTHETIC_ONLY", "native_replay_verified": False,
             "cases": [{"id": "probe", "manifest_sha256": sha(canonical(manifest)),
                        "recipe_sha256": sha(canonical(recipe))}]}
    (Path(root) / "index.json").write_bytes(canonical(index))
    return folder


class MorphologyAudit(unittest.TestCase):
    def test_raster_decoding_and_components(self):
        png = png_for(4, 4, {(0, 0), (0, 1), (3, 3)})
        self.assertEqual(png_cells(png), ((4, 4), {(0, 0), (0, 1), (3, 3)}))
        self.assertEqual(components({(0, 0), (0, 1), (3, 3)}), [2, 1])
        with self.assertRaises(AssertionError):
            png_cells(png[:-1] + b"a")

    def test_crosscheck_png_and_replayed_material_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixture(tmp)
            result = audit_episode(folder)
            self.assertEqual(result["changed_cells"], 1)
            self.assertEqual(result["final_component_count"], 1)
            self.assertEqual(result["zero_change_strokes"], 0)
            self.assertFalse(result["native_material_verified"])
            report = audit_corpus(tmp)
            self.assertEqual(report["cases"]["probe"], result)
            self.assertEqual(report["report_sha256"], sha(canonical({k: v for k, v in report.items() if k != "report_sha256"})))
            self.assertEqual(audit_corpus(tmp), report)

    def test_forged_manifest_hash_cannot_conceal_trajectory_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixture(tmp)
            edited = canonical([[2, 2]])
            (folder / "trajectory.json").write_bytes(edited)
            manifest = json.loads((folder / "manifest.json").read_bytes())
            manifest["artifacts"]["trajectory.json"] = {"bytes": len(edited), "sha256": sha(edited)}
            (folder / "manifest.json").write_bytes(canonical(manifest))
            with self.assertRaises(AssertionError):
                audit_episode(folder)

    def test_forged_material_png_hash_cannot_conceal_edit_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixture(tmp)
            edited = png_for(4, 4, {(2, 2)})
            (folder / "after.png").write_bytes(edited)
            manifest = json.loads((folder / "manifest.json").read_bytes())
            manifest["artifacts"]["after.png"] = {"bytes": len(edited), "sha256": sha(edited)}
            (folder / "manifest.json").write_bytes(canonical(manifest))
            with self.assertRaises(AssertionError):
                audit_episode(folder)

    def test_rejects_traversal_and_path_hijacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = fixture(tmp)
            manifest = json.loads((folder / "manifest.json").read_bytes())
            manifest["artifacts"]["../outside"] = {"bytes": 0, "sha256": sha(b"")}
            (folder / "manifest.json").write_bytes(canonical(manifest))
            with self.assertRaises(AssertionError):
                audit_episode(folder)

    def test_actual_lab_fixture_when_in_repository(self):
        try:
            from lab import VERSION, write, replay
        except ImportError:
            self.skipTest("lab.py not present outside checked-out D-2 tree")
        with tempfile.TemporaryDirectory() as tmp:
            case = {"version": VERSION, "family": "guided_coverage", "seed": 29,
                    "canvas": [64, 64], "steps": 96, "brush_radius": 0,
                    "order": 3, "curve": "moore", "exit_below": 48000, "reenter_at": 52000,
                    "recovery": "backtrack"}
            folder = Path(tmp) / "case"
            write(case, folder)
            self.assertTrue(replay(folder))
            self.assertEqual(audit_episode(folder)["strokes"], 96)


if __name__ == "__main__":
    unittest.main()
