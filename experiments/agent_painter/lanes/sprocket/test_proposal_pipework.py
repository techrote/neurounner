#!/usr/bin/env python3
"""Model-free positive/adversarial checks for v2 composition prototypes."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proposal_pipework import GRAPHS, HOST_SEEDS, check_graph, evaluate, program_for, public_split_signatures
from compiler import CompilerError, compile_program, structural_signature, canonical_json

class ProposalTests(unittest.TestCase):
    def test_six_distinct_structural_programs(self):
        report = evaluate()
        self.assertEqual(report["reviewable_structures"], 6)
        self.assertEqual(report["host_seed_views_compiled"], 12)
        self.assertFalse(report["final_targets_opened"])
        self.assertEqual({x["id"] for x in report["candidates"]}, set(GRAPHS))
        self.assertTrue(all(x["native_admitted"] is False and x["train_eligible"] is False
                            for x in report["candidates"]))
        self.assertEqual(len({x["structural_signature_sha256"] for x in report["candidates"]}), 6)

    def test_graph_semantic_invariants(self):
        expected = {
            "P01":(0,4,2), "P02":(0,4,1), "P03":(2,0,2),
            "P04":(1,2,2), "P05":(2,1,2), "P06":(4,0,5),
        }
        for key, (cycles,endpoints,junctions) in expected.items():
            with self.subTest(key=key):
                result = check_graph(key, program_for(key))
                self.assertEqual((result["cycles"],result["endpoints"],result["junctions"]),
                                 (cycles,endpoints,junctions))

    def test_determinism_and_consistent_diameter(self):
        for key in GRAPHS:
            with self.subTest(key=key):
                program = program_for(key)
                self.assertEqual(program["components"][0]["diameter"], 3)
                for seed in HOST_SEEDS:
                    a = compile_program(program, seed=seed)
                    b = compile_program(program, seed=seed)
                    self.assertEqual(a["operations_sha256"], b["operations_sha256"])
                    self.assertTrue(a["proposal_only"])
                    self.assertFalse(a["native_qualified"])

    def test_seed_and_diameter_not_structural_novelty(self):
        p = program_for("P04")
        q = copy.deepcopy(p)
        q["components"][0]["diameter"] = 5
        self.assertEqual(structural_signature(p), structural_signature(q))
        self.assertEqual(canonical_json(structural_signature(p)),
                         canonical_json(structural_signature(q)))

    def test_duplicate_edge_refused(self):
        p = program_for("P04")
        p["components"][0]["edges"].append([1,0])
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0])

    def test_falsified_cycle_refused(self):
        p = program_for("P03")
        p["requirements"]["cycles"] = 1
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0])

    def test_unintended_overlapping_segment_refused(self):
        p = program_for("P04")
        p["components"][0]["nodes"].append([51,28])
        p["components"][0]["edges"].append([0,6])
        with self.assertRaises(CompilerError):
            compile_program(p, seed=HOST_SEEDS[0])

    def test_untrusted_corpus_refused_without_opening_final(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root/"admitted-corpus.json").write_text("{}")
            (root/"final.jsonl").write_text("SENSITIVE_SEALED_TARGET")
            with self.assertRaises(ValueError):
                public_split_signatures(root)
            self.assertEqual((root/"final.jsonl").read_text(), "SENSITIVE_SEALED_TARGET")

if __name__ == "__main__":
    unittest.main(verbosity=2)
