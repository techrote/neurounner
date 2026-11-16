#!/usr/bin/env python3
"""Cheap WORM W4 regression: never shadow the official multimodal model profile."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent

class W4ProfileBindingTests(unittest.TestCase):
    def test_profile_binding_is_not_overwritten_by_sampler(self):
        tree = ast.parse((ROOT / "worm_train_w4_v1.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        assigns = []
        for node in ast.walk(main):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(t, ast.Name) and t.id == "selected" for t in targets):
                    assigns.append(node)
        self.assertEqual(len(assigns), 1, "official profile selected must have only one binding")
        self.assertIsInstance(assigns[0].value, ast.Call)
        self.assertEqual(assigns[0].value.func.id, "profile")
    def test_evaluation_requires_profile_mapping(self):
        tree = ast.parse((ROOT / "worm_train_w4_v1.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
        invokes = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id in ("evaluation", "rollout")]
        self.assertGreaterEqual(len(invokes), 2)
        for node in invokes:
            self.assertIsInstance(node.args[2], ast.Name)
            self.assertEqual(node.args[2].id, "selected")

if __name__ == "__main__":
    unittest.main()
