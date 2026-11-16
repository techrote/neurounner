"""Adversarial model-free tests for pinned SMOL/LFM research source comparison."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import smol_peer_review as s

LF=json.loads((Path(__file__).resolve().parent/"contrastive_proposals_v1.json").read_text())


def smol_fixture():
    return {
        "status":"PROPOSED_NOT_NATIVE_ADMITTED_NOT_TRAIN",
        "original_native_source":LF["original_native_source"],
        "original_manifest_sha256":LF["original_manifest_sha256"],
        "proposals":[
            {"id":"smol_g06_stepped_absence","family":"greebles",
             "program":{"version":2,"family":"greebles","components":[
                 {"kind":"panel","box":[8,66,22,18]},
                 {"kind":"panel","box":[32,48,19,20]},
                 {"kind":"panel","box":[70,12,20,22]}]}},
            {"id":"smol_p01_nested_branch_tree","family":"pipework",
             "program":{"version":2,"family":"pipework","components":[{
                 "kind":"pipe_network",
                 "nodes":[[12,64],[50,64],[82,64],[50,36],[26,36],[74,36],[50,14]],
                 "edges":[[0,1],[1,2],[1,3],[3,4],[3,5],[3,6]],
                 "diameter":3,"fittings":True}]}}
        ]}


class SmolPeerWitness(unittest.TestCase):
    def test_positive_source_witness_no_admission(self):
        r=s.compare(LF,smol_fixture())
        self.assertEqual((r["greebles"]["lfm_panels"],r["greebles"]["smol_panels"]),(6,3))
        self.assertEqual(r["greebles"]["intersect"],716)
        self.assertEqual(r["greebles"]["union"],3174)
        self.assertEqual(r["pipework"]["intersect"],5)
        self.assertEqual(r["pipework"]["union"],365)
        self.assertEqual(r["pipework"]["lfm_degrees"],[1,1,1,1,1,1,4,4])
        self.assertEqual(r["pipework"]["smol_degrees"],[1,1,1,1,1,3,4])
        self.assertEqual(r["train_admissions"],0)
        self.assertFalse(r["final_opened"])

    def test_mutated_peer_blob_refused(self):
        with tempfile.TemporaryDirectory() as root:
            file=Path(root)/"file"
            file.write_text("{}")
            with self.assertRaisesRegex(ValueError,"blob mismatch"):
                s.verify(file,s.PINS["smol"][1])

    def test_fake_admission_status_refused(self):
        a=smol_fixture()
        a["status"]="TRAIN_ADMITTED"
        with self.assertRaisesRegex(ValueError,"status"):
            s.compare(LF,a)

    def test_changed_chase_requires_review(self):
        a=smol_fixture()
        a["proposals"][0]["program"]["components"][1]["box"]=[60,48,19,20]
        with self.assertRaisesRegex(ValueError,"Greebles motif"):
            s.compare(LF,a)

    def test_altered_topology_requires_review(self):
        a=smol_fixture()
        a["proposals"][1]["program"]["components"][0]["edges"]=[
            [0,1],[1,2],[1,3],[3,4],[3,5],[1,6]]
        with self.assertRaisesRegex(ValueError,"undeclared pipe node"):
            s.compare(LF,a)

    def test_bad_edges_refused(self):
        base=smol_fixture()["proposals"][1]["program"]
        for edge_list in (
            [[0,3],[1,2],[1,3],[3,4],[3,5],[3,6]], # diagonal
            [[0,1],[0,1],[1,3],[3,4],[3,5],[3,6]], # duplicate
            [[0,1],[1,2],[1,3],[3,4],[3,5],[5,4]], # loop
        ):
            p=copy.deepcopy(base)
            p["components"][0]["edges"]=edge_list
            with self.subTest(edge_list=edge_list), self.assertRaises(ValueError):
                s.tree(p)

    def test_invalid_panel_coordinates_refused(self):
        for box in ([95,0,2,1],[5,5,True,10]):
            with self.subTest(box=box),self.assertRaises(ValueError):
                s.panels({"components":[{"kind":"panel","box":box}]})

    def test_bad_diameter_refused(self):
        p=smol_fixture()["proposals"][1]["program"]
        p["components"][0]["diameter"]=5
        with self.assertRaisesRegex(ValueError,"diameter"):
            s.tree(p)

    def test_duplicate_proposal_id_refused(self):
        a=smol_fixture()
        a["proposals"].append(copy.deepcopy(a["proposals"][0]))
        with self.assertRaisesRegex(ValueError,"duplicate"):
            s.compare(LF,a)


if __name__=="__main__":
    unittest.main()
