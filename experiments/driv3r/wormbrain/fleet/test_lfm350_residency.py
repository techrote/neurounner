"""Source-only policy tests; no actual model download/inference/Godot execution."""
import ast
import json
import os
from pathlib import Path
import unittest

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
class TestLfm350(unittest.TestCase):
    def test_wrapper_and_pinned_untuned_base(self):
        source = (HERE / "lfm350_residency.py").read_text()
        self.assertTrue(ast.parse(source))
        self.assertIn("LiquidAI/LFM2.5-350M", source)
        self.assertIn("9921cd4d82c0e3986887aa4b376f9446729ef8a3", source)
        self.assertIn("import smol_residency as base", source)
        self.assertNotIn("PeftModel", source)
        self.assertNotIn("optimizer.step(", source)
        self.assertNotIn(".backward(", source)
    def test_entrypoint_inherited_by_genuine_subprocess(self):
        src = (HERE / "smol_residency.py").read_text()
        self.assertIn('os.environ.get("DRIV3R_RESIDENCY_ENTRYPOINT", __file__)', src)
        lfm = (HERE / "lfm350_residency.py").read_text()
        self.assertIn('os.environ["DRIV3R_RESIDENCY_ENTRYPOINT"]', lfm)
        self.assertIn('configure().main()', lfm)
    def test_bounded_request_contract(self):
        r = json.loads((HERE / "lfm350-residency-request.json").read_text())
        self.assertEqual(r["schema"], "driv3r.lfm350-residency/1")
        self.assertEqual(r["model"], "LiquidAI/LFM2.5-350M")
        self.assertEqual(r["model_revision"], "9921cd4d82c0e3986887aa4b376f9446729ef8a3")
        self.assertEqual(r["seeds"], [4201,4219])
        self.assertEqual(r["turns"], 8)
        self.assertEqual(r["runner_max"], 1)
        self.assertEqual(r["models_per_runner_max"], 2)
        self.assertFalse(r["training"])
        self.assertFalse(r["native"])
    def test_workflow_source_lease_and_failure_artifacts(self):
        src = Path(".github/workflows/driv3r-wormfleet-lfm350-residency.yml").read_text()
        self.assertIn("lfm350-residency-request.json", src)
        self.assertNotIn("workflow_dispatch:", src)
        self.assertIn("timeout-minutes:", src)
        self.assertIn("if: always()", src)
        self.assertIn("allow_patterns", src)
        self.assertIn("lfm350_residency.py", src)

if __name__ == "__main__":
    unittest.main()
