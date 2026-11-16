"""WORM-001 six-action anti-collapse probes: built from W0 legal-action contract."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import probes_v1 as p


class SprocketV1Probes(unittest.TestCase):
    def test_every_action_and_split_variant_is_teacher_legal(self):
        for split, variants in (("TRAIN", p.TRAIN_VARIANTS),
                                ("DEVELOPMENT", p.DEV_VARIANTS)):
            for f in p.FAMILIES:
                for a in p.ACTIONS:
                    for v in variants:
                        obs = p.local(f, a, split, v)
                        self.assertEqual(p.c.teacher_choice(obs), a)
                        self.assertIn(a, obs["legal"])
        self.assertEqual(len(p.probes("TRAIN")), 54)
        self.assertEqual(len(p.probes("DEVELOPMENT")), 36)

    def test_literal_collision_and_conflicting_target_fail_closed(self):
        original_train, original_dev, _ = p.c.make_dataset()
        train, dev = p.probes("TRAIN"), p.probes("DEVELOPMENT")
        self.assertEqual(p.verify_prompts(original_train + train, original_dev + dev)
                         ["shared_literal_model_prompts"], 0)
        with self.assertRaisesRegex(AssertionError, "leakage"):
            p.verify_prompts(train + [dev[0]], dev)
        with self.assertRaisesRegex(AssertionError, "conflicting"):
            p.verify_prompts([train[0], dict(train[0], target="STOP")], dev)

    def test_frozen_balanced_training_view_not_dev(self):
        train, dev, _ = p.c.make_dataset()
        ct, cd = p.probes("TRAIN"), p.probes("DEVELOPMENT")
        chosen = p.balanced_view(train, ct)
        self.assertEqual(len(chosen), 54)
        for f in p.FAMILIES:
            self.assertEqual(set(p.summarize(chosen)["family_action"][f].values()), {3})
        self.assertEqual(set(p.summarize(chosen)["actions"].values()), {9})
        p.verify_prompts(chosen, dev + cd)
        self.assertTrue(all(r["target"] in r["observation"]["legal"] for r in chosen))

    def test_generated_jsonl_digests_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            result = p.write_probe_set(out)
            self.assertEqual(result["datasets"]["constructed_train"]["rows"], 54)
            self.assertEqual(result["datasets"]["constructed_development"]["rows"], 36)
            self.assertEqual(result["datasets"]["balanced_train_view"]["rows"], 54)
            for meta in result["datasets"].values():
                data = (out / meta["filename"]).read_bytes()
                self.assertEqual(meta["sha256"], hashlib.sha256(data).hexdigest())
                self.assertEqual(meta["rows"], len(data.splitlines()))
                for line in data.splitlines():
                    row = json.loads(line)
                    self.assertIn(row["target"], row["observation"]["legal"])
            self.assertEqual(json.loads((out / "manifest.json").read_bytes()), result)

    def test_real_right_only_replay_png_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            receipt = p.write_right_stress(out)
            self.assertEqual(len(receipt["rows"]), 6)
            for row in receipt["rows"]:
                j = (out / (row["id"] + ".json")).read_bytes()
                png = (out / (row["id"] + ".png")).read_bytes()
                rec = json.loads(j)
                self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))
                self.assertEqual(hashlib.sha256(j).hexdigest(), row["json_sha256"])
                self.assertEqual(hashlib.sha256(png).hexdigest(), row["png_sha256"])
                self.assertTrue(all(e["raw_choice"] == "RIGHT" for e in rec["events"]))
                self.assertEqual(rec["metrics"]["invalid_raw_choices"],
                                 sum(not e["raw_legal"] for e in rec["events"]))
                self.assertLessEqual(rec["steps_executed"], p.c.GALLERY_STEPS)
                self.assertIn("NOT-trained-model", rec["policy_source"])


if __name__ == "__main__":
    unittest.main()
