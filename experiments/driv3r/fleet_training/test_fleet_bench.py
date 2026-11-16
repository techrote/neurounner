"""Adversarial preflight for source-fenced fleet benchmark, no model or game run."""
import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import fleet_bench as f


class TestFleetBench(unittest.TestCase):
    def setUp(self):
        self.r = json.loads((Path(__file__).parent / "request.json").read_text())

    def test_exact_pins_and_four_distinct_lease_seeds(self):
        f.request_check(self.r)
        self.r["seeds"]["beta"][1] = self.r["seeds"]["alpha"][0]
        with self.assertRaises(AssertionError):
            f.request_check(self.r)

    def test_external_checkpoint_or_optimizer_policy_change_rejected(self):
        self.r["upstream_commit"] = "0" * 40
        with self.assertRaises(AssertionError):
            f.request_check(self.r)

    def test_nonpopulation_mock_results_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "results.json").write_text(json.dumps({
                "format": "mock", "seed": 9101, "native": True
            }))
            with self.assertRaises((AssertionError, KeyError)):
                f.verified_result(p, 9101, self.r)

    def test_readonly_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "file"
            p.write_bytes(b"original")
            self.assertEqual(f.sha_file(p), hashlib.sha256(b"original").hexdigest())

    def test_rss_missing_process_is_zero(self):
        self.assertEqual(f.rss_kib(2147483647), 0)


if __name__ == "__main__":
    unittest.main()
