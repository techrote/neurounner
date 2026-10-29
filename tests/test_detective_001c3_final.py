"""Offline custody tests for DETECTIVE-001C3 post-training final."""
from __future__ import annotations
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from scripts import detective_001c3_dataset as dataset
from scripts import detective_001c3_final as final


class C3FinalTests(unittest.TestCase):
    def _adapter(self,root,name,payload):
        p=root/name;p.mkdir()
        (p/"adapter_model.safetensors").write_bytes(payload)
        (p/"adapter_config.json").write_text(json.dumps({"name":name})+"\n")
        return p

    def _dataset(self,root):
        p=root/"pairs";dataset.build(p);return p

    def test_requires_real_complete_adapter(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);pairs=self._dataset(root)
            with patch.object(final,"DATASET",pairs):
                with self.assertRaises(FileNotFoundError):
                    final.materialize(root/"missing",root/"final.json")
                bad=root/"bad";bad.mkdir()
                with self.assertRaises(final.C3FinalError):
                    final.materialize(bad,root/"final.json")

    def test_reproducible_balanced_and_post_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);pairs=self._dataset(root)
            a=self._adapter(root,"a",b"c3")
            with patch.object(final,"DATASET",pairs):
                one=final.materialize(a,root/"one.json")
                two=final.materialize(a,root/"two.json")
            self.assertEqual(one,two)
            self.assertEqual(len(one["scenarios"]),18)
            self.assertTrue(one["generated_after_training"])
            self.assertFalse(one["c1_final_used_as_training"])
            self.assertFalse(one["c2_final_used_as_training"])
            counts={}
            for row in one["scenarios"]:
                counts[row["relation_label"]]=counts.get(row["relation_label"],0)+1
                self.assertTrue(row["mandatory_clue_fallback"]["available"])
            self.assertTrue(all(v==3 for v in counts.values()))
            self.assertEqual(len(counts),6)

    def test_trained_hash_changes_seed_and_suite(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);pairs=self._dataset(root)
            a=self._adapter(root,"a",b"one")
            with patch.object(final,"DATASET",pairs):
                x=final.materialize(a,root/"x.json")
                (a/"adapter_model.safetensors").write_bytes(b"two")
                y=final.materialize(a,root/"y.json")
            self.assertNotEqual(x["seed_sha256"],y["seed_sha256"])
            self.assertNotEqual(
                [r["prompt"] for r in x["scenarios"]],
                [r["prompt"] for r in y["scenarios"]]
            )

    def test_final_families_disjoint_from_c1_c2_c3_training(self):
        cfg,_=dataset.load_config()
        final_names=set(cfg["final_recipe"]["family_names"])
        prior=set(cfg["dataset"]["family_ids"])
        c1=json.loads(
            (dataset.ROOT/"datasets/detective-001c/manifest.json").read_text())
        for s in c1["member_summaries"].values():
            prior.update(s["families"])
        c2=json.loads(
            (dataset.ROOT/"datasets/detective-001c2/manifest.json").read_text())
        for s in c2["member_summaries"].values():
            prior.update(
                x.split(":",1)[-1] for x in s["relations"] if False
            )
        c2_rows=[]
        for line in (dataset.ROOT/"datasets/detective-001c2/training/V-G02-A.jsonl").read_text().splitlines():
            if line.strip(): c2_rows.append(json.loads(line))
        prior.update(x["case_family"].split(":",1)[-1] for x in c2_rows)
        self.assertFalse(final_names & prior)

    def test_relation_metadata_has_lie_belief_and_stale_sentinels(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);pairs=self._dataset(root)
            a=self._adapter(root,"a",b"meta")
            with patch.object(final,"DATASET",pairs):
                doc=final.materialize(a,root/"final.json")
            for row in doc["scenarios"]:
                ev=row["evaluation"]
                if row["relation_label"]=="intentional_lie":
                    self.assertTrue(ev["lie_marker"])
                    self.assertTrue(ev["truth_marker"])
                    self.assertNotEqual(ev["lie_marker"],ev["truth_marker"])
                elif row["relation_label"]=="corrupted_belief":
                    self.assertTrue(ev["required_marker"])
                    self.assertTrue(ev["truth_marker"])
                    self.assertNotEqual(ev["required_marker"],ev["truth_marker"])
                elif row["relation_label"]=="contradiction_repair":
                    self.assertTrue(ev["stale_marker"])
                elif row["relation_label"]=="uninformed_speculation":
                    self.assertTrue(ev["unsupported_marker"])

    def test_prompts_hide_member_and_relation_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);pairs=self._dataset(root)
            a=self._adapter(root,"a",b"hide")
            with patch.object(final,"DATASET",pairs):
                doc=final.materialize(a,root/"final.json")
            for row in doc["scenarios"]:
                low=row["prompt"].casefold()
                self.assertNotIn("v-g02-a",low)
                self.assertNotIn("v-g09-c",low)
                self.assertNotIn(row["relation_label"],low)
                self.assertNotIn("truth_marker",low)

    def test_verify_refuses_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=pathlib.Path(td);pairs=self._dataset(root)
            a=self._adapter(root,"a",b"verify")
            path=root/"final.json"
            with patch.object(final,"DATASET",pairs):
                final.materialize(a,path);final.verify(path,a)
                doc=json.loads(path.read_text())
                doc["scenarios"][0]["prompt"]+=" altered"
                path.write_text(json.dumps(doc)+"\n")
                with self.assertRaises(final.C3FinalError):
                    final.verify(path,a)


if __name__=="__main__":
    unittest.main()
