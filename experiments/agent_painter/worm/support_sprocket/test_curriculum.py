"""SPROCKET-support v0: actual W0 contract, deterministic synthetic data and rollout tests."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

FILE = Path(__file__).resolve().parent / "curriculum.py"
spec = importlib.util.spec_from_file_location("sprocket_worm_support_curriculum", FILE)
curriculum = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = curriculum
spec.loader.exec_module(curriculum)


class SprocketSupportWormTests(unittest.TestCase):
    def test_exact_common_contract_and_legality(self):
        self.assertEqual(curriculum.ACTIONS,
                         ("STRAIGHT", "LEFT", "RIGHT", "BRANCH", "RETRACE", "STOP"))
        self.assertEqual(curriculum.GRID_SIZE, 96)
        for family in curriculum.FAMILIES:
            for seed in (101, 1009):
                for mode in curriculum.OBS_MODES:
                    record = curriculum.worm.run(seed, mode, family, 40)
                    for event in record["events"]:
                        obs = event["observation"]
                        action = curriculum.teacher_choice(obs)
                        self.assertIn(action, obs["legal"])
                        self.assertEqual(action, curriculum.teacher_choice(obs))

    def test_constructed_dead_end_branch_stop(self):
        for family in curriculum.FAMILIES:
            self.assertEqual(curriculum.teacher_choice(
                curriculum.synthetic_fixture(family, "junction")), "BRANCH")
            self.assertEqual(curriculum.teacher_choice(
                curriculum.synthetic_fixture(family, "blocked")), "RETRACE")
            self.assertEqual(curriculum.teacher_choice(
                curriculum.synthetic_fixture(family, "terminal")), "STOP")
            self.assertEqual(curriculum.teacher_choice(
                curriculum.synthetic_fixture(family, "deadline")), "STOP")

    def test_reject_malformed_or_injected_observation(self):
        base = curriculum.synthetic_fixture("pipework", "junction")
        for key, value in (
            ("legal", ["TELEPORT", "STOP"]),
            ("legal", ["STOP", "STOP"]),
            ("local_5x5", ["@@@"]),
            ("family", "FINAL"),
            ("heading", 13),
            ("tick", -1),
        ):
            bad = dict(base)
            bad[key] = value
            with self.assertRaises(ValueError):
                curriculum.teacher_choice(bad)

    def test_replay_native_independent_and_finite(self):
        for family in curriculum.FAMILIES:
            a = curriculum.teacher_rollout(2111, family, 42)
            b = curriculum.teacher_rollout(2111, family, 42)
            self.assertEqual(curriculum.digest(a), curriculum.digest(b))
            self.assertEqual(a["size"], 96)
            self.assertLessEqual(a["steps_executed"], 42)
            self.assertTrue(all(e["raw_legal"] for e in a["events"]))
            self.assertEqual(a["metrics"]["invalid_raw_choices"], 0)
            self.assertTrue(curriculum.worm.png_bytes(a).startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(a["engine"], "python-stdlib-baseline-not-native")
        with self.assertRaises(ValueError):
            curriculum.teacher_rollout(2111, "greebles", 129)

    def test_split_hygiene_and_target_transport(self):
        train, dev, manifest = curriculum.make_dataset()
        self.assertGreater(len(train), 100)
        self.assertGreater(len(dev), 30)
        self.assertFalse({x["world_seed"] for x in train if x["world_seed"] is not None}
                         & {x["world_seed"] for x in dev})
        self.assertFalse({x["observation_sha256"] for x in train}
                         & {x["observation_sha256"] for x in dev})
        self.assertEqual(set(manifest["action_histograms"]["TRAIN"]), set(curriculum.ACTIONS))
        for split in ("TRAIN", "DEVELOPMENT"):
            counts = manifest["action_histograms"][split]
            self.assertLess(counts["STOP"] / manifest["counts"][split], 0.1)
            self.assertGreater(counts["BRANCH"], 0)
            self.assertGreater(counts["RETRACE"], 0)
        for row in train + dev:
            self.assertIn(row["target"], row["observation"]["legal"])
            self.assertTrue(row["input"].endswith("\nACTION="))
            self.assertIn("OBS=", row["input"])
            self.assertEqual(row["observation_sha256"], curriculum.digest(row["observation"]))
            self.assertNotIn("FINAL", row["source"])

    def test_real_gallery_and_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            ds = curriculum.write_dataset(out / "dataset")
            gal = curriculum.write_gallery(out / "gallery")
            self.assertEqual(len(gal["entries"]), 24)
            self.assertEqual(len(list((out / "gallery").glob("*.png"))), 24)
            self.assertEqual(len(list((out / "gallery").glob("*.json"))), 25)
            for e in gal["entries"]:
                stem = e["id"]
                png = (out / "gallery" / (stem + ".png")).read_bytes()
                rec = json.loads((out / "gallery" / (stem + ".json")).read_text())
                self.assertEqual(curriculum.hashlib.sha256(png).hexdigest(), e["sha256_png"])
                self.assertEqual(rec["metrics"], e["metrics"])
                self.assertLessEqual(rec["steps_executed"], gal["requested_step_budget"])
            self.assertEqual(
                curriculum.hashlib.sha256((out / "dataset" / "train.jsonl").read_bytes()).hexdigest(),
                ds["train_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
