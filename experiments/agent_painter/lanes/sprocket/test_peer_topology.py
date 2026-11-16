#!/usr/bin/env python3
"""Model-free cross-lane negative controls: proposal collisions are not TRAIN."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from peer_topology import PEER_BLOB_SHA, PEER_GRAPH_EDGES, PEER_SOURCE_SHA, isomorphic, peer_matches
from proposal_pipework import GRAPHS, program_for

class CrossLaneTests(unittest.TestCase):
    def match(self,key,program=None):
        component=(program or program_for(key))["components"][0]
        return peer_matches(component["nodes"],component["edges"])

    def test_exact_four_known_duplicates_refused_as_new_global_groups(self):
        expected={
            "P01":["void-pipe-P01"], "P03":["void-pipe-P07"],
            "P04":["void-pipe-P04"], "P06":["void-pipe-P06"],
        }
        for key,matches in expected.items():
            with self.subTest(key=key):
                self.assertEqual(self.match(key),matches)

    def test_coordinate_shift_and_diameter_variation_do_not_make_p04_novel(self):
        shifted=copy.deepcopy(program_for("P04"))
        part=shifted["components"][0]
        part["nodes"]=[[x+2,y+3] for x,y in part["nodes"]]
        part["diameter"]=5
        shifted["variation"]=77
        self.assertEqual(self.match("P04",shifted),["void-pipe-P04"])

    def test_identical_p06_lattice_even_with_relabeling(self):
        original=program_for("P06")["components"][0]
        peer_n,peer_e=PEER_GRAPH_EDGES["void-pipe-P06"]
        self.assertTrue(isomorphic(len(original["nodes"]),original["edges"],peer_n,peer_e))
        mapping={i:(i+3)%9 for i in range(9)}
        permuted=[[mapping[a],mapping[b]] for a,b in original["edges"]]
        self.assertTrue(isomorphic(9,permuted,peer_n,peer_e))

    def test_avoid_false_positive_p05_spur_attachment(self):
        self.assertEqual(self.match("P05"),[])
        self.assertFalse(isomorphic(
            len(GRAPHS["P05"]["nodes"]),GRAPHS["P05"]["edges"],
            *PEER_GRAPH_EDGES["void-pipe-P05"]))

    def test_p02_has_no_exact_peer_graph_but_remains_heldout_review_pending(self):
        self.assertEqual(self.match("P02"),[])

    def test_invalid_duplicate_or_self_edge_refused(self):
        with self.assertRaises(ValueError):
            isomorphic(3,[[0,1],[1,0]],3,[[0,1],[1,2]])
        with self.assertRaises(ValueError):
            isomorphic(3,[[0,0],[1,2]],3,[[0,1],[1,2]])

    def test_peer_witness_provenance_is_pinned(self):
        self.assertEqual(PEER_SOURCE_SHA,"54161e2217f709ffb8b23634f21efa351411395f")
        self.assertEqual(PEER_BLOB_SHA,"446f0ed7a6f7f16f8de9a6bfd97ebbde3a425b16")
        self.assertEqual(len(PEER_GRAPH_EDGES),5)

if __name__=="__main__":
    unittest.main(verbosity=2)
