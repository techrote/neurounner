"""Model-free DEVELOPMENT structural and seed-custody regression tests."""
import unittest
from evaluate import verify_development_pairs


def rows():
    return [
        {"group": f"g{i}", "prompt": f"request {i}", "family": ("greebles", "pipework", "architecture")[i // 2],
         "seed": seed}
        for i in range(6) for seed in (104729, 130363)
    ]


class DevelopmentPairs(unittest.TestCase):
    def test_six_unique_requests_and_twelve_host_views(self):
        self.assertEqual(verify_development_pairs(rows()), 6)

    def test_changed_second_seed_prompt_refused(self):
        changed = rows()
        changed[1]["prompt"] += " changed"
        with self.assertRaisesRegex(ValueError, "structural group"):
            verify_development_pairs(changed)

    def test_duplicate_seed_and_missing_view_refused(self):
        changed = rows()
        changed[1]["seed"] = 104729
        with self.assertRaisesRegex(ValueError, "seed"):
            verify_development_pairs(changed)
        with self.assertRaisesRegex(ValueError, "six requests"):
            verify_development_pairs(rows()[:-1])


if __name__ == "__main__":
    unittest.main()
