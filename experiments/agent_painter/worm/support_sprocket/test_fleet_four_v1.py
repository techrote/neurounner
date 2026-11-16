#!/usr/bin/env python3
"""Sprocket W0 fleet handshake adversarial tests; no optimizer/model/native."""
import json
from pathlib import Path
import tempfile
import unittest

import fleet_four_v1 as fleet


class FourStreamTest(unittest.TestCase):
    def test_four_true_processes_distinct_and_replayed(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / "fleet"
            manifest = fleet.run_four(folder)
            self.assertEqual(len(manifest["streams"]), 4)
            self.assertEqual({s["world_seed"] for s in manifest["streams"]},
                             {313, 337, 347, 359})
            self.assertEqual(len({s["png_sha256"] for s in manifest["streams"]}), 4)
            self.assertEqual(manifest["valid_model_processes"], 0)
            self.assertFalse(manifest["native"])
            self.assertEqual(manifest["optimizer_updates"], 0)
            state = json.loads((folder / "runtime.json").read_text())
            self.assertEqual(len({r["pid"] for r in state["streams"].values()}), 4)
            self.assertGreater(state["overlap_ns"], 0)
            for s in manifest["streams"]:
                sid = s["stream_id"]
                r = state["streams"][sid]
                self.assertEqual(r["transitions"],
                                 ["spawned", "loaded_w0", "entered_host_run", "completed_w0"])
                self.assertGreater(r["cpu_ns"], 0)
                self.assertGreater(r["max_rss_kib"], 0)
                rec = json.loads((folder / (sid + ".json")).read_text())
                png = (folder / (sid + ".png")).read_bytes()
                self.assertEqual(fleet.verify((sid, s["family"], s["policy"],
                                               s["world_seed"]), rec, png), s["signature"])
                self.assertEqual(fleet.digest(png), s["png_sha256"])
                self.assertLessEqual(s["steps_executed"], 96)
                self.assertTrue(s["signature"]["wall_nearby_ticks"] >= 0)
            sums = (folder / "SHA256SUMS.txt").read_text()
            self.assertEqual(len(sums.splitlines()), 10)

    def test_fail_closed_on_wrong_identity_and_png(self):
        spec = fleet.SPECS[0]
        rec = fleet.worm.run(spec[3], spec[2], spec[1], fleet.STEPS, fleet.SIZE)
        png = fleet.worm.png_bytes(rec)
        self.assertEqual(fleet.verify(spec, rec, png),
                         fleet.signatures(rec))
        with self.assertRaises(ValueError):
            fleet.verify((spec[0], spec[1], spec[2], 999), rec, png)
        with self.assertRaises(ValueError):
            fleet.verify(spec, rec, png + b"tamper")
        rec["events"][0]["raw_legal"] = not rec["events"][0]["raw_legal"]
        with self.assertRaises(ValueError):
            fleet.verify(spec, rec, png)

    def test_refuses_overwrite_existing_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / "canary").write_text("must survive")
            with self.assertRaises(ValueError):
                fleet.run_four(folder)
            self.assertEqual((folder / "canary").read_text(), "must survive")


if __name__ == "__main__":
    unittest.main()
