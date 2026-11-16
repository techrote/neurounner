#!/usr/bin/env python3
"""Adversarial W3 original-gallery audit tests, with real source-W0 PNG bytes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import primary_gallery_audit as a
from motif_metrics import dominant_repeated_motif


class GalleryAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gallery = self.root / "gallery"
        self.gallery.mkdir()
        self.seed, self.budget = 2101, 12
        for family in a.worm.FAMILIES:
            selected = a.rollout(self.seed, family, self.budget, lambda obs: "STRAIGHT")
            raw = []
            for event in selected["events"]:
                obs = event["observation"]
                legal = "STRAIGHT" in obs["legal"]
                raw.append({"observation": obs, "raw_text": "STRAIGHT",
                            "token_ids": [17, 7], "output_tokens": 2,
                            "eos_terminated": True, "raw_legal": legal,
                            "valid_first_pass": legal})
            self._save(f"{family}-model-{self.seed}", selected)
            self._save_raw(f"{family}-model-{self.seed}-raw-first-passes.json", raw)
            for policy in ("random", "wall"):
                self._save(f"{family}-{policy}-{self.seed}",
                           a.worm.run(self.seed, policy, family, self.budget))
            self._save(f"{family}-oracle-{self.seed}",
                       a.rollout(self.seed, family, self.budget, a.oracle))

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, stem, record):
        (self.gallery / (stem + ".json")).write_bytes(a.canonical(record))
        (self.gallery / (stem + ".png")).write_bytes(a.worm.png_bytes(record))

    def _save_raw(self, name, record):
        (self.gallery / name).write_bytes(a.canonical(record))

    def _edit(self, name, edit):
        path = self.gallery / name
        record = json.loads(path.read_bytes())
        edit(record)
        path.write_bytes(a.canonical(record))

    def test_exact_w0_png_and_firstpass_receipt(self):
        data = a.audit(self.gallery, seed=self.seed, budget=self.budget)
        self.assertEqual(len(data["metrics"]), 12)
        self.assertEqual(len(data["matched_pairs"]), 9)
        self.assertEqual(len(data["original_file_hashes"]), 27)
        self.assertTrue(all(p["equal_requested_budget"] for p in data["matched_pairs"]))
        self.assertFalse(data["native"])
        self.assertEqual(data["schema"], "worm-lfm-primary-gallery-audit-v2")
        self.assertEqual(len(data["matched_pairs"]), 9)
        self.assertEqual(data["metrics"][0]["raw_cycle"]["period"], 1)
        self.assertEqual(data["metrics"][0]["raw_cycle"]["span"], self.budget)
        self.assertIn("raw_cycle_span_fraction_delta", data["matched_pairs"][0])

    def test_detects_multi_action_loop_not_only_constant_token(self):
        pattern = ["BRANCH", "RETRACE", "RIGHT"]
        data = dominant_repeated_motif(["STOP", "LEFT"] + pattern * 8 + ["STRAIGHT"])
        self.assertEqual(data["period"], 3)
        self.assertEqual(data["motif"], pattern)
        self.assertEqual(data["start"], 2)
        self.assertEqual(data["span"], 24)
        self.assertEqual(data["full_repeats"], 8)
        self.assertEqual(dominant_repeated_motif(["RIGHT"] * 32)["span"], 32)
        self.assertEqual(dominant_repeated_motif(["LEFT", "RIGHT", "STOP"])["span"], 0)
        with self.assertRaises(ValueError):
            dominant_repeated_motif(["LEFT", None])
        with self.assertRaises(ValueError):
            dominant_repeated_motif(["LEFT"] * 513)

    def test_literal_raw_mutation_rejected(self):
        name = f"greebles-model-{self.seed}-raw-first-passes.json"
        self._edit(name, lambda rows: rows[0].update(raw_text="LEFT"))
        with self.assertRaisesRegex(ValueError, "raw output"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_forged_eos_rejected(self):
        name = f"greebles-model-{self.seed}-raw-first-passes.json"
        self._edit(name, lambda rows: rows[0].update(eos_terminated=False))
        with self.assertRaisesRegex(ValueError, "EOS"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_forged_model_validity_rejected(self):
        name = f"greebles-model-{self.seed}-raw-first-passes.json"
        self._edit(name, lambda rows: rows[0].update(valid_first_pass=False))
        with self.assertRaisesRegex(ValueError, "legality receipt"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_altered_host_trajectory_rejected(self):
        name = f"greebles-model-{self.seed}.json"
        self._edit(name, lambda obj: obj["trajectory"][0].__setitem__(0, 21))
        with self.assertRaisesRegex(ValueError, "host replay diverged"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_png_mutation_rejected(self):
        file = self.gallery / f"greebles-model-{self.seed}.png"
        file.write_bytes(file.read_bytes() + b"spoof")
        with self.assertRaisesRegex(ValueError, "PNG differs"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_swapped_seed_or_budget_rejected(self):
        name = f"greebles-model-{self.seed}.json"
        self._edit(name, lambda obj: obj.update(step_budget=999))
        with self.assertRaisesRegex(ValueError, "decision budget"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_missing_first_passes_rejected(self):
        (self.gallery / f"greebles-model-{self.seed}-raw-first-passes.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or linked"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_fake_control_rejected(self):
        name = f"greebles-random-{self.seed}.json"
        self._edit(name, lambda obj: obj["events"][0].update(raw_choice="TELEPORT"))
        with self.assertRaisesRegex(ValueError, "host replay diverged"):
            a.audit(self.gallery, seed=self.seed, budget=self.budget)

    def test_real_negative_gallery_png_and_metrics(self):
        out = self.root / "negative"
        summary = a.demo(out, seed=self.seed, budget=self.budget)
        self.assertEqual(len(summary["reports"]), 18)
        self.assertEqual(len(summary["files"]), 36)
        self.assertTrue(all((out / f["file"]).is_file() for f in summary["files"]))
        broken = [r for r in summary["reports"] if r["policy"] == "illegal-teleport"]
        self.assertTrue(all(r["metrics"]["invalid_raw_count"] >= 1 for r in broken))
        self.assertTrue(all(r["metrics"]["host_fallback_count"] >= 1 for r in broken))
        with self.assertRaises(ValueError):
            a.demo(out, seed=self.seed, budget=self.budget)


if __name__ == "__main__":
    unittest.main()
