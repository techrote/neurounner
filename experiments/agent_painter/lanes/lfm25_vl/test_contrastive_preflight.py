"""Model-free adversarial LFM contrast research fixtures; no native or model calls."""
import copy
import json
from pathlib import Path
import unittest

import contrastive_preflight as subject


class ContrastPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(subject.FILE.read_text(encoding="utf-8"))
        cls.items = {item["family"]: item for item in cls.document["proposals"]}

    def test_fixed_model_and_native_claims(self):
        self.assertEqual(subject.MANIFEST, self.document["original_manifest_sha256"])
        self.assertEqual(subject.SOURCE, self.document["original_native_source"])
        self.assertEqual(self.document["status"], "PROPOSED_NOT_TRAIN_NOT_NATIVE_ADMITTED")
        self.assertEqual(self.document["host_seeds"], [104729, 130363])

    def test_all_three_full_compiler_only_seed_witnesses(self):
        for family, item in self.items.items():
            with self.subTest(family=family):
                proof = subject.preflight(item)
                self.assertEqual(len(proof), 2)
                self.assertTrue(all(v["native_qualified"] is False for v in proof))
                self.assertTrue(all(v["compiler_structure_pass"] for v in proof))

    def test_architecture_room_field_leak_is_refused(self):
        bad = copy.deepcopy(self.items["greebles"])
        bad["program"]["components"][0]["openings"] = []
        with self.assertRaises(Exception):
            subject.preflight(bad)

    def test_bad_pipework_requirements_refused_not_repaired(self):
        bad = copy.deepcopy(self.items["pipework"])
        bad["program"]["requirements"] = {"closed": True, "region": [0, 0, 96, 96]}
        with self.assertRaises(Exception):
            subject.preflight(bad)

    def test_invalid_six_terminal_witness_refused(self):
        bad = copy.deepcopy(self.items["pipework"])
        bad["program"]["components"][0]["edges"].pop()
        with self.assertRaises(Exception):
            subject.preflight(bad)

    def test_semantic_diameter_drift_refused(self):
        bad = copy.deepcopy(self.items["pipework"])
        bad["program"]["components"][0]["diameter"] = 5
        with self.assertRaisesRegex(ValueError, "six-terminal"):
            subject.preflight(bad)

    def test_architecture_missing_route_refused(self):
        bad = copy.deepcopy(self.items["architecture"])
        bad["program"]["requirements"]["route"] = []
        with self.assertRaisesRegex(ValueError, "architecture hall"):
            subject.preflight(bad)

    def test_greeble_no_chase_refused(self):
        bad = copy.deepcopy(self.items["greebles"])
        bad["program"]["components"][0]["box"][0] = 40
        with self.assertRaises(ValueError):
            subject.preflight(bad)

    def test_greeble_dislocated_right_bank_not_three_paired_tiers(self):
        # Six still-schema-correct panel records, five permitted details, all
        # three panels per bank and a full-height empty centre chase. Their
        # altered vertical positions must not pass the literal tier witness.
        bad = copy.deepcopy(self.items["greebles"])
        bad["program"]["components"][3]["box"][1] = 52
        bad["program"]["components"][5]["box"][1] = 75
        p = bad["program"]
        self.assertEqual(sum(c["box"][0] < 40 for c in p["components"]), 3)
        self.assertEqual(sum(c["box"][0] >= 56 for c in p["components"]), 3)
        subject.compiler.validate_program(p)
        with self.assertRaisesRegex(ValueError, "paired three-tier"):
            subject.preflight(bad)

    def test_structural_signature_ignores_only_style_for_pipe(self):
        p = self.items["pipework"]["program"]
        q = copy.deepcopy(p)
        q["components"][0]["diameter"] = 5
        self.assertEqual(subject.signature(p), subject.signature(q))

    def test_dual_cross_not_isomorphic_to_pinned_peer_graphs(self):
        """Necessary topology comparison only; NOT global TRAIN admission.

        Read-only pinned sources:
          SPROCKET P01-P06: proposal_pipework.py blob
            3fd5fa0157eb0572ca7ee6fbe979a94fbadbed5b
          SPROCKET R09/R10: rich_pipework_v2.py blob
            37d9b7d3f698d5e851e213043d8d43c5f2c9750a
          VOID P01/P04/P05/P06/P07: pinned witness in peer_topology.py blob
            9f5c87705d6b4452941f65fc43d64b44d871383b
            (original VOID producer 54161e2217f709ffb8b23634f21efa351411395f)
          SMOL P01: curriculum_proposals_v2.json blob
            0b0adae3c02d1ccc9bda2cdca393f272ca7ecf63

        Degree-multiset inequality proves non-isomorphism to these known
        graphs, NOT spatial/semantic novelty, held-out safety or native
        approval. Source changes need new pinned comparative review.
        """
        graph = self.items["pipework"]["program"]["components"][0]
        degrees = [0] * len(graph["nodes"])
        for a, b in graph["edges"]:
            degrees[a] += 1
            degrees[b] += 1
        actual = tuple(sorted(degrees))
        self.assertEqual(actual, (1, 1, 1, 1, 1, 1, 4, 4))
        self.assertEqual(len(graph["edges"]) - len(graph["nodes"]) + 1, 0)
        peers = {
            "sprocket-P01/void-P01": (1, 1, 1, 1, 3, 3),
            "sprocket-P02": (1, 1, 1, 1, 4),
            "sprocket-P03/void-P07": (2, 2, 2, 2, 3, 3),
            "sprocket-P04/void-P04": (1, 1, 2, 2, 3, 3),
            "sprocket-P05": (1, 2, 2, 2, 2, 2, 3, 4),
            "sprocket-P06/void-P06": (2, 2, 2, 2, 3, 3, 3, 3, 4),
            "void-P05": (1, 2, 2, 2, 2, 2, 3, 4),
            "smol-P01": (1, 1, 1, 1, 1, 3, 4),
            "sprocket-R09": (1, 1, 1, 2, 2, 3, 3, 3),
            "sprocket-R10": (1, 1, 2, 2, 3, 3, 3, 3, 3, 3),
        }
        self.assertEqual(len(peers), 10)
        for name, seq in peers.items():
            with self.subTest(peer=name):
                self.assertEqual(tuple(sorted(seq)), seq)
                self.assertNotEqual(actual, seq)

    def test_no_final_file_in_visible_split_function(self):
        import inspect
        src = inspect.getsource(subject.original_visible_splits)
        self.assertIn('for split in ("TRAIN", "DEVELOPMENT"):', src)
        self.assertNotIn('m["split_files"]["FINAL"]', src)


if __name__ == "__main__":
    unittest.main()
