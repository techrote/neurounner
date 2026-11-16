#!/usr/bin/env python3
"""Adversarial and positive source-only gates for fitted Pipework research."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rich_pipework_v2 import RICH_GRAPHS, HOST_SEEDS, evaluate, program_for, topology_witness
from proposal_pipework import GRAPHS as ORIGINAL, public_split_signatures
from peer_topology import PEER_GRAPH_EDGES, isomorphic
from compiler import CompilerError, compile_program, structural_signature

class RichPipeworkTests(unittest.TestCase):
    def test_exact_two_new_topology_classes(self):
        report = evaluate()
        self.assertEqual((report["groups"], report["compiler_seed_views"]), (2,4))
        self.assertEqual((report["native_positive_views"],report["new_train_groups_admitted"]), (0,0))
        self.assertFalse(report["final_targets_opened"])
        self.assertEqual({x["id"] for x in report["candidates"]}, {"R09","R10"})
        self.assertEqual(len({x["structural_signature_sha256"] for x in report["candidates"]}),2)

    def test_declared_degree_and_cycle_witnesses(self):
        self.assertEqual((topology_witness("R09")["cycles"], topology_witness("R09")["endpoints"]), (1,3))
        self.assertEqual((topology_witness("R10")["cycles"], topology_witness("R10")["endpoints"]), (3,2))
        for name, graph in RICH_GRAPHS.items():
            self.assertTrue(all(
                not isomorphic(len(graph["nodes"]),graph["edges"],len(old["nodes"]),old["edges"])
                for old in ORIGINAL.values()))
            self.assertTrue(all(
                not isomorphic(len(graph["nodes"]),graph["edges"],n,edges)
                for n,edges in PEER_GRAPH_EDGES.values()))

    def test_fittings_are_real_geometry_not_just_declared(self):
        for name in RICH_GRAPHS:
            program = program_for(name)
            self.assertEqual(program["components"][0]["diameter"], 5)
            self.assertIs(program["components"][0]["fittings"], True)
            for seed in HOST_SEEDS:
                with self.subTest(name=name,seed=seed):
                    a = compile_program(program,seed=seed)
                    b = compile_program(program,seed=seed)
                    self.assertEqual(a["operations_sha256"],b["operations_sha256"])
                    self.assertGreater(len(a["semantic"]["fittings"]), 0)
                    self.assertEqual(a["semantic"]["expected_cycles"],RICH_GRAPHS[name]["cycles"])
                    self.assertTrue(a["structural_preflight"]["passed"])
                    self.assertFalse(a["native_qualified"])

    def test_false_cycle_claim_refused(self):
        p=program_for("R10")
        p["requirements"]["cycles"] = 2
        with self.assertRaises(CompilerError):
            compile_program(p,seed=HOST_SEEDS[0])

    def test_collar_width_not_new_structural_group(self):
        p=program_for("R09")
        q=copy.deepcopy(p)
        q["components"][0]["diameter"]=3
        q["components"][0]["fittings"]=False
        self.assertEqual(structural_signature(p),structural_signature(q))

    def test_relabelled_pinned_peer_cannot_pass_novelty(self):
        g=RICH_GRAPHS["R09"]
        original=copy.deepcopy(g)
        try:
            g["nodes"]=copy.deepcopy(ORIGINAL["P05"]["nodes"])
            g["edges"]=copy.deepcopy(ORIGINAL["P05"]["edges"])
            g["cycles"]=ORIGINAL["P05"]["cycles"]
            g["degrees"]=ORIGINAL["P05"]["degrees"]
            with self.assertRaisesRegex(ValueError,"duplicates original SPROCKET"):
                topology_witness("R09")
        finally:
            g.clear()
            g.update(original)

    def test_untrusted_corpus_rejected_without_final_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"admitted-corpus.json").write_text("{}")
            (root/"FINAL.jsonl").write_text("SEALED_EXCLUDED")
            with self.assertRaises(ValueError):
                public_split_signatures(root)
            self.assertEqual((root/"FINAL.jsonl").read_text(),"SEALED_EXCLUDED")

if __name__=="__main__":
    unittest.main(verbosity=2)
