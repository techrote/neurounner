"""Offline-only regression tests for the D-2 loop escape study."""
import json
from pathlib import Path
import tempfile
import unittest

from worm_sketch import canonical, digest
from worm_scent_study import SCENES, SEEDS, run_one as old
from worm_escape_study import (ARMS, MIN_ESCAPE_TICKS, MAX_ESCAPE_TICKS,
                               corpus, evidence, run_one_escape, verify_folder)


class HysteresisEscapeTests(unittest.TestCase):
    def test_exact_frozen_controller_parity(self):
        for scene in SCENES:
            for seed in SEEDS:
                for arm, reference in (("strong_x16", "attract_x16"),
                                       ("fixed_x1", "attract"),
                                       ("sham_hysteresis", "attract_x16")):
                    actual = run_one_escape(scene, seed, arm)
                    original = old(scene, seed, reference)
                    self.assertEqual(actual["trail"], original["trail"])
                    self.assertEqual(actual["material"], original["material"])
                    self.assertEqual(actual["metrics"]["material_sha256"],
                                     original["metrics"]["material_sha256"])
                    self.assertEqual([e["heading"] for e in actual["events"] if e["kind"] == "STEP"],
                                     [e["heading"] for e in original["events"] if e["kind"] == "STEP"])

    def test_only_escape_intervention_can_change_hysteresis_path(self):
        different = 0
        for scene in SCENES:
            for seed in SEEDS:
                strong = run_one_escape(scene, seed, "strong_x16")
                adaptive = run_one_escape(scene, seed, "hysteresis")
                self.assertGreaterEqual(adaptive["metrics"]["escape_triggers"], 1)
                self.assertEqual(strong["permitted"], adaptive["permitted"])
                self.assertEqual(strong["guide"], adaptive["guide"])
                if strong["trail"] != adaptive["trail"]:
                    different += 1
                    s = [e for e in strong["events"] if e["kind"] == "STEP"]
                    a = [e for e in adaptive["events"] if e["kind"] == "STEP"]
                    first = next(i for i in range(min(len(s), len(a)))
                                 if s[i]["to_cursor"] != a[i]["to_cursor"])
                    self.assertTrue(a[first]["escape_mode"])
        self.assertGreaterEqual(different, 3)

    def test_hard_permissions_cardinal_material_journal(self):
        for scene in SCENES:
            for arm in ARMS:
                outcome = run_one_escape(scene, 23, arm)
                ink = set()
                prev = None
                for event in outcome["events"]:
                    if event["kind"] != "STEP":
                        self.assertIsNone(event.get("native_receipt"))
                        self.assertFalse(event.get("executed", False))
                        continue
                    source = tuple(event["from_cursor"])
                    target = tuple(event["to_cursor"])
                    if prev is not None:
                        self.assertEqual(source, prev)
                    self.assertEqual(sum(abs(t - s) for t, s in zip(target, source)), 1)
                    self.assertIn(target, outcome["permitted"])
                    self.assertIn([event["heading"], list(target)], event["legal_catalogue"])
                    self.assertEqual(event["material_before_sha256"],
                                     digest(canonical(sorted(map(list, ink)))))
                    if event["pen"] == "down":
                        ink.add(target)
                    self.assertEqual(event["material_after_sha256"],
                                     digest(canonical(sorted(map(list, ink)))))
                    self.assertIsNone(event["native_receipt"])
                    self.assertIsNone(event["model_decision"])
                    prev = target
                self.assertEqual(ink, outcome["material"])
                self.assertEqual(len(ink), outcome["metrics"]["painted_cells"])

    def test_hysteresis_state_transition_bounds(self):
        for scene in SCENES:
            for seed in SEEDS:
                outcome = run_one_escape(scene, seed, "hysteresis")
                events = outcome["events"]
                starts = [e for e in events if e["kind"] == "ESCAPE_ENTER"]
                ends = [e for e in events if e["kind"] == "ESCAPE_EXIT"]
                self.assertEqual(len(starts), outcome["metrics"]["escape_triggers"])
                self.assertEqual(len(ends), outcome["metrics"]["escape_exits"])
                self.assertLessEqual(len(ends), len(starts))
                for e in starts:
                    self.assertLessEqual(e["recent_unique"], 10)
                    self.assertEqual(e["window"], 12)
                for e in ends:
                    self.assertGreaterEqual(e["elapsed"], MIN_ESCAPE_TICKS)
                    self.assertLessEqual(e["elapsed"], MAX_ESCAPE_TICKS)
                    if e["reason"] == "novelty":
                        self.assertGreaterEqual(e["novel_cells"], 5)
                    else:
                        self.assertEqual(e["reason"], "timeout")
                self.assertLessEqual(outcome["metrics"]["steps"], 192)

    def test_all_changes_experimental_and_no_native_undo(self):
        for arm in ARMS:
            out = run_one_escape("ring", 79, arm)
            self.assertEqual(out["events"][-1]["kind"], "UNDO_MATERIAL_EDIT_UNAVAILABLE")
            self.assertFalse(out["events"][-1]["executed"])
            self.assertTrue(all(e.get("native_receipt") is None for e in out["events"]))
            self.assertTrue(all(e.get("model_decision") is None for e in out["events"]))

    def test_deterministic_corpus_and_original_artifact_verifier(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = corpus(a), corpus(b)
            self.assertEqual(first, second)
            self.assertEqual(len(first["outcomes"]), len(SCENES)*len(SEEDS)*len(ARMS))
            self.assertEqual((Path(a)/"index.json").read_bytes(),
                             (Path(b)/"index.json").read_bytes())
            for row in first["outcomes"]:
                case = Path(a)/row["case"]
                self.assertTrue(verify_folder(case))
                self.assertEqual((case/"after.png").read_bytes(),
                                 (Path(b)/row["case"]/"after.png").read_bytes())

    def test_tamper_rejected_even_when_rehashed(self):
        with tempfile.TemporaryDirectory() as temp:
            corpus(temp)
            case = Path(temp)/"ring-seed23-hysteresis"
            file = case/"events.json"
            original = file.read_bytes()
            tampered = json.loads(original)
            step = next(e for e in tampered if e["kind"] == "STEP")
            step["to_cursor"] = [1, 1]
            changed = canonical(tampered)
            file.write_bytes(changed)
            manifest_file = case/"manifest.json"
            manifest = json.loads(manifest_file.read_bytes())
            manifest["artifacts"]["events.json"] = digest(changed)
            manifest_file.write_bytes(canonical(manifest))
            with self.assertRaises(ValueError):
                verify_folder(case)

    def test_fail_closed_parameters(self):
        for args in (("no", 23, "hysteresis", 192),
                     ("ring", 999, "hysteresis", 192),
                     ("ring", 23, "bad", 192),
                     ("ring", 23, "hysteresis", 0),
                     ("ring", 23, "hysteresis", 513),
                     ("ring", 23, "hysteresis", True)):
            with self.assertRaises(ValueError):
                run_one_escape(*args)


if __name__ == "__main__":
    unittest.main()
