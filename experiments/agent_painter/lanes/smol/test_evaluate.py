"""Model-free regression checks: DEV custody and sealed FINAL isolation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _module():
    path = Path(__file__).with_name("evaluate.py")
    spec = importlib.util.spec_from_file_location("smol_development_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DevelopmentCustodyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.source = "a" * 40
        self.compiler = "b" * 64
        self.rows = [{
            "id": "dev-1-seed104729", "split": "DEVELOPMENT",
            "family": "pipework", "structural_group": "dev-one",
            "prompt": "Paint a connected T-junction with two bends.", "seed": 104729,
        }]
        self._write()

    def _write(self):
        body = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in self.rows)
        (self.dir / "development.jsonl").write_text(body, encoding="utf-8")
        manifest = {
            "schema_version": 1, "state": "native-admitted",
            "source_sha": self.source, "compiler_sha256": self.compiler,
            "structural_groups": {
                "TRAIN": ["training-structure"],
                "DEVELOPMENT": ["dev-one"],
                "FINAL": ["sealed-structure"],
            },
            "split_files": {
                "TRAIN": {"path": "train.jsonl", "count": 2, "sha256": "c" * 64},
                "DEVELOPMENT": {
                    "path": "development.jsonl", "count": len(self.rows),
                    "sha256": hashlib.sha256(body.encode()).hexdigest(),
                },
                "FINAL": {"path": "final.jsonl", "count": 1, "sha256": "d" * 64},
            },
        }
        (self.dir / "admitted-corpus.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_reads_only_development_without_any_final_file(self):
        self.assertFalse((self.dir / "final.jsonl").exists())
        rows, _ = _module().development_rows(self.dir, self.source, self.compiler)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seed"], 104729)
        self.assertFalse((self.dir / "final.jsonl").exists())

    def test_rejects_development_tampering(self):
        (self.dir / "development.jsonl").write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "custody hash"):
            _module().development_rows(self.dir, self.source, self.compiler)

    def test_rejects_final_group_leaking_into_development(self):
        self.rows[0]["structural_group"] = "sealed-structure"
        self._write()
        with self.assertRaisesRegex(ValueError, "invalid DEVELOPMENT row"):
            _module().development_rows(self.dir, self.source, self.compiler)

    def test_refuses_native_source_mismatch(self):
        with self.assertRaisesRegex(ValueError, "native corpus identity"):
            _module().development_rows(self.dir, "c" * 40, self.compiler)


if __name__ == "__main__":
    unittest.main()
