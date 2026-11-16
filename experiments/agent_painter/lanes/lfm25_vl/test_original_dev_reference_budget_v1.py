"""Model-free adversarial tests for original visible DEVELOPMENT teacher budget gate."""
from __future__ import annotations
import hashlib
import json
import sys
import types
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import original_dev_reference_budget_v1 as gate
from common import canonical, digest_json


class DevelopmentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.corpus = Path(self.temp.name)
        self.rows = []
        groups = []
        for family in sorted(gate.FAMILIES):
            for i in range(2):
                group = f"{family}-{i}"
                groups.append(group)
                for seed in sorted(gate.SEEDS):
                    program = {"version": 2, "family": family,
                               "geometry": {"name": group, "seed": seed}}
                    self.rows.append({
                        "id": f"{group}-{seed}", "split": "DEVELOPMENT",
                        "structural_group": group, "family": family, "seed": seed,
                        "prompt": f"Reference-only {family} group {i}",
                        "program": program, "completion": canonical(program),
                        "program_sha256": digest_json(program), "native_qualified": True,
                        "native_admission": {
                            "state": "PASS", "native_execution": True, "verified": True,
                            "source_sha": gate.ORIGINAL_NATIVE_SOURCE,
                            "runtime_sha256": "a"*64,
                            "program_sha256": digest_json(program),
                        },
                    })
        self.groups = groups
        self.make_manifest()

    def make_manifest(self):
        data = "".join(canonical(x)+"\n" for x in self.rows)
        (self.corpus/"development.jsonl").write_text(data, encoding="utf-8")
        manifest = {
            "schema_version": 1, "state": "native-admitted",
            "source_sha": gate.ORIGINAL_NATIVE_SOURCE, "runtime_sha256": "a"*64,
            "structural_groups": {
                "TRAIN": ["train-"+str(i) for i in range(6)],
                "DEVELOPMENT": self.groups, "FINAL": ["sealed-"+str(i) for i in range(3)]
            },
            "split_files": {
                "TRAIN": {"path":"train.jsonl", "sha256":"b"*64, "count":12},
                "DEVELOPMENT": {
                    "path": "development.jsonl",
                    "sha256": gate.sha(data.encode("utf-8")), "count": 12},
                "FINAL": {"path": "final.jsonl", "sha256": "c"*64, "count":6}
            }
        }
        # TRAIN/FINAL do not exist: success must never try opening their contents.
        (self.corpus/"admitted-corpus.json").write_text(canonical(manifest),encoding="utf-8")
        self.digest = gate.sha((self.corpus/"admitted-corpus.json").read_bytes())

    def read(self):
        return gate.select_visible_development(self.corpus, expected_manifest_sha256=self.digest)

    def test_reference_path_refuses_ambient_adapter_impostor(self):
        impostor = types.ModuleType("adapter")
        impostor.MODEL_ID = "wrong-cached-adapter"
        previous = sys.modules.get("adapter")
        sys.modules["adapter"] = impostor
        try:
            actual = gate.load_qualified_adapter()
            self.assertIsNot(actual, impostor)
            self.assertEqual(actual.MODEL_ID, gate.MODEL_ID)
            self.assertEqual(Path(actual.__file__).resolve(), (gate.LANE / "adapter.py").resolve())
        finally:
            if previous is None:
                sys.modules.pop("adapter", None)
            else:
                sys.modules["adapter"] = previous

    def test_six_whole_groups_twelve_rows_without_final_files(self):
        rows, custody = self.read()
        self.assertEqual(len(rows), 12)
        self.assertEqual(custody["independent_development_groups"], 6)
        self.assertFalse(custody["sealed_final_content_opened"])
        self.assertFalse((self.corpus/"final.jsonl").exists())
        self.assertEqual(custody["families"], {f:2 for f in sorted(gate.FAMILIES)})

    def test_reject_corpus_manifest_byte_drift(self):
        self.assertRaisesRegex(ValueError, "manifest bytes changed",
                               gate.select_visible_development, self.corpus, "0"*64)

    def test_reject_development_file_mutation(self):
        with (self.corpus/"development.jsonl").open("a") as f:
            f.write("{}\n")
        with self.assertRaisesRegex(ValueError, "hash|mismatch"):
            self.read()

    def test_reject_final_partition_collision(self):
        raw=json.loads((self.corpus/"admitted-corpus.json").read_text())
        raw["structural_groups"]["FINAL"][0]=self.groups[0]
        (self.corpus/"admitted-corpus.json").write_text(canonical(raw),encoding="utf-8")
        self.digest=gate.sha((self.corpus/"admitted-corpus.json").read_bytes())
        with self.assertRaisesRegex(ValueError,"leakage"):
            self.read()

    def test_reject_relabelled_development_row(self):
        self.rows[0]["split"]="TRAIN"
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT row"):
            self.read()

    def test_reject_missing_native_admission(self):
        self.rows[0]["native_admission"]["verified"]=False
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"native admission"):
            self.read()

    def test_reject_altered_teacher_bytes(self):
        self.rows[0]["completion"]="{}"
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"teacher bytes"):
            self.read()

    def test_reject_missing_second_seed(self):
        self.rows[1]["seed"]=7
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"DEVELOPMENT row"):
            self.read()

    def test_reject_wrong_original_source(self):
        self.rows[0]["native_admission"]["source_sha"]="deadbeef"
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"native admission"):
            self.read()

    def test_reject_missing_family_balance(self):
        self.rows[0]["family"]="pipework"
        self.rows[1]["family"]="pipework"
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"one independent structure|expected 2 groups"):
            self.read()

    def test_reject_duplicate_ids(self):
        self.rows[1]["id"]=self.rows[0]["id"]
        self.make_manifest()
        with self.assertRaisesRegex(ValueError,"IDs"):
            self.read()

    def test_stable_complete_eos_and_budget_measurement(self):
        rows, _ = self.read()
        class FakeAdapter:
            @staticmethod
            def encode_row(processor, row, system, max_length):
                assert system == gate.SYSTEM and max_length == 2048
                return {
                    "input_ids": [1,2,3,4,5,7],
                    "labels": [-100,-100,-100,4,5,7],
                    "boundary":3, "target_tokens":3,
                    "prompt_bytes":len(row["prompt"].encode("utf-8")),
                    "completion_bytes":len(row["completion"].encode("utf-8")),
                }
        out=gate.encode_references(rows, object(), FakeAdapter)
        self.assertEqual(len(out),12)
        self.assertTrue(all(x["eos_token_id"]==7 and x["fits_provisional_512"] for x in out))

    def test_refuse_missing_true_eos(self):
        rows,_=self.read()
        class BadAdapter:
            @staticmethod
            def encode_row(processor,row,system,max_length):
                return {"input_ids":[1,2,3], "labels":[-100,2,3],
                        "boundary":1, "target_tokens":2,
                        "prompt_bytes":len(row["prompt"].encode()),
                        "completion_bytes":len(row["completion"].encode())}
        with self.assertRaisesRegex(ValueError,"EOS"):
            gate.encode_references(rows, object(), BadAdapter)

if __name__=="__main__":
    unittest.main()
