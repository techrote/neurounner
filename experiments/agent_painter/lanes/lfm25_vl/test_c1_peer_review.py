"""Adversarial, model-free C1 peer source-composition tests; no FINAL or Art."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import c1_peer_review as subject


HERE = Path(__file__).resolve().parent


def program(boxes):
    return {"family": "greebles", "components":
            [{"kind": "panel", "box": list(b)} for b in boxes]}


class C1PeerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lfm = json.loads((HERE / "contrastive_proposals_v1.json").read_text())
        cls.c1 = next(x["program"] for x in cls.lfm["proposals"]
                      if x["id"] == subject.C1_ID)

    def test_original_six_paired_tiers_and_central_chase(self):
        m = subject.measures(subject.panel_boxes(self.c1))
        self.assertEqual(m["panels"], 6)
        self.assertTrue(m["paired_three_tiers"])
        self.assertTrue(m["full_height_central_chase_clear"])
        self.assertEqual(m["cross_chase_panel_count"], 0)

    def test_dislocated_right_bank_rejected(self):
        altered = copy.deepcopy(self.c1)
        altered["components"][3]["box"][1] = 50
        altered["components"][5]["box"][1] = 75
        self.assertFalse(subject.paired_tiers(subject.panel_boxes(altered)))

    def test_cross_chase_versus_gap_is_not_just_panel_count(self):
        boxes = subject.panel_boxes(self.c1)
        changed = list(boxes)
        changed[1] = (38, boxes[1][1], 23, 20)
        m = subject.measures(changed)
        self.assertEqual(m["panels"], 6)
        self.assertFalse(m["full_height_central_chase_clear"])
        self.assertEqual(m["cross_chase_panel_count"], 1)

    def test_peer_nearest_motifs_and_no_admission(self):
        # Deliberately simple source boxes with the same macro motifs as peer
        # original programs. The workflow separately pins the *actual* files.
        vd = {"state": "unadmitted-research", "candidates": [
            {"id": "void-v3-G01", "training_admitted": False, "program":
                program([(8,10,20,24),(35,10,21,24),(64,10,22,24),(10,51,24,22),(63,55,20,24)])},
            {"id": "void-v3-G02", "training_admitted": False, "program":
                program([(9,11,24,18),(42,34,28,18),(10,63,27,20),(69,68,18,18)])},
        ]}
        sp = {"G08": {"program": program([(8,8,35,23),(56,8,26,22),
                  (9,39,23,21),(44,39,38,22),(11,69,41,19)])},
              "G09": {"program": program([(7,13,78,19),(15,42,61,17),(34,70,52,18)])}}
        r = subject.compare(self.lfm, vd, sp)
        self.assertEqual(len(r["comparisons"]), 4)
        self.assertFalse(r["final_targets_opened"])
        self.assertEqual(r["train_groups_admitted"], 0)
        self.assertEqual(r["native_positive_from_this_audit"], 0)
        for x in r["comparisons"].values():
            self.assertFalse(x["paired_three_tiers"])
            self.assertFalse(x["full_height_central_chase_clear"])
            self.assertGreater(x["cross_chase_panel_count"], 0)
            self.assertGreater(x["box_union"], x["box_intersection"])

    def test_peer_identity_change_is_not_auto_approved(self):
        source = {"state": "unadmitted-research", "candidates": [
            {"id": x, "training_admitted": False, "program": self.c1}
            for x in subject.PEERS[:2]]}
        same = {"G08": {"program": self.c1}, "G09": {"program": self.c1}}
        with self.assertRaisesRegex(ValueError, "same peer motif"):
            subject.compare(self.lfm, source, same)

    def test_compromised_source_blob_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "peer.py"
            f.write_text("STUDIES = {}\n")
            with self.assertRaisesRegex(ValueError, "foreign or modified"):
                subject.verified_bytes(f, subject.BLOBS["sprocket"])

    def test_ast_parser_does_not_execute_import_or_call(self):
        self.assertEqual(subject.sprocket_studies(b"STUDIES = {'G08': {'program': {}}}\n"),
                         {"G08": {"program": {}}})
        with self.assertRaises((ValueError, TypeError)):
            subject.sprocket_studies(b"STUDIES = __import__('os').getcwd()\n")

    def test_malformed_panel_bounds_rejected(self):
        for box in ([70, 0, 40, 10], [8, 8, 0, 9], [8, 8, True, 10]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                subject.panel_boxes(program([box]))


if __name__ == "__main__":
    unittest.main()
