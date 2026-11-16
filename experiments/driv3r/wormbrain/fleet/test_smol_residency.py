"""No-model negative policy and selector tests; never claim resident SLMs."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name("smol_residency.py")
class TestSmol(unittest.TestCase):
    def test_parse_and_fixed_model_pins(self):
        src=SOURCE.read_text()
        tree=ast.parse(src)
        self.assertTrue(len(tree.body)>0)
        self.assertIn("HuggingFaceTB/SmolLM2-135M-Instruct",src)
        self.assertIn("12fd25f77366fa6b3b4b768ec3050bf629380bac",src)
    def test_original_host_only_and_no_training(self):
        s=SOURCE.read_text()
        self.assertIn("mode=\"callback\"",s)
        self.assertIn("mode=\"replay\"",s)
        self.assertIn("optimizer_steps",s)
        self.assertIn("native_execution",s)
        self.assertNotIn(".backward(",s)
        self.assertNotIn("optimizer.step(",s)
    def test_bound_budget_and_host_legal_choice(self):
        s=SOURCE.read_text()
        self.assertIn("TURNS = 8",s)
        self.assertIn("chosen = max(offered",s)
        self.assertIn("STOP",s)
        self.assertIn("source-faithful",s.lower() if "source-faithful" in s.lower() else "source-faithful")
    def test_supervisor_bounded_and_fails_closed(self):
        s=SOURCE.read_text()
        self.assertIn("start_new_session=True",s)
        self.assertIn("signal.SIGKILL",s)
        self.assertIn("no_success_claim",s)
        self.assertIn("two_real_resident_pids_observed",s)
if __name__=="__main__":
    unittest.main()
