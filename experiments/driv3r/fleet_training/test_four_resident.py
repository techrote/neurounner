"""FT-002 four-resident source and CPU-progress custody preflight."""
import json
from pathlib import Path
import unittest
import four_resident as q


class TestFourResident(unittest.TestCase):
    def setUp(self):
        self.r = json.loads((Path(__file__).parent / "request_quad.json").read_text())

    def test_exact_immutable_four_seed_request(self):
        self.assertEqual(len(q.validated_request(self.r)["seeds"]), 4)
        self.r["seeds"][1] = self.r["seeds"][0]
        with self.assertRaises(ValueError):
            q.validated_request(self.r)

    def test_no_request_or_source_expansion(self):
        self.r["upstream_commit"] = "0" * 40
        with self.assertRaises(ValueError):
            q.validated_request(self.r)

    def test_unsafe_optimizer_budget_rejected(self):
        self.r["train_episodes"] = 1000
        with self.assertRaises(ValueError):
            q.validated_request(self.r)

    def test_real_proc_cpu_stat_sanity(self):
        import os
        ticks = q.cpu_ticks(os.getpid())
        self.assertIsInstance(ticks, int)
        self.assertGreaterEqual(ticks, 0)


if __name__ == "__main__":
    unittest.main()
