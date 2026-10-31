"""Offline adversarial tests for source-only Sprocket MAC proxy. No torch."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ncr_001f_sprocket_mac_proxy as m


class SprocketMACProxy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = json.loads((ROOT / "experiments/ncr-001/protocol.json").read_text())
        cls.q = json.loads((ROOT / "experiments/ncr-001/source-pins.json").read_text())

    def test_exact_original_source_and_closed_scientific_gate(self):
        result = m.audit()
        self.assertEqual(set(result["lanes"]), set(m.SOURCES))
        self.assertFalse(result["actual_compute_budget_qualified"])
        self.assertFalse(result["full_72_conformance_pass"])
        self.assertFalse(result["train_authorized"])
        self.assertFalse(result["new_model_execution"])

    def test_three_taps_three_cell_steps_and_parameter_difference(self):
        r = m.mac_proxy(1280)
        self.assertEqual(r["mlp_dense_mac_per_token"], 1966080)
        self.assertEqual(r["nca_dense_conv_mac_per_token"], 2371584)
        self.assertEqual(r["nca_conv_mac_per_token"], 405504)
        self.assertEqual(r["nca_over_mlp_mac_ratio"], 1.20625)
        self.assertEqual(r["nca_trainable_parameter_formula"], 658904)
        self.assertEqual(r["mlp_trainable_parameter_formula"], 658432)
        self.assertEqual(r["nca_field_float32_bytes_per_token_one_field_only"], 1024)

    def test_one_step_is_not_current_preregistered_budget(self):
        self.assertEqual(m.mac_proxy(1280, steps=1)["nca_conv_mac_per_token"], 135168)
        self.assertEqual(m.mac_proxy(1280, steps=3)["nca_conv_mac_per_token"], 405504)

    def rejects(self, change):
        p, q = copy.deepcopy(self.p), copy.deepcopy(self.q)
        change(p, q)
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            m.check_contract(p, q)

    def test_refuse_training_or_source_variant_conflation(self):
        self.rejects(lambda p, q: p.__setitem__("train_authorized", True))
        self.rejects(lambda p, q: p["models"]["sprocket_chat"].__setitem__(
            "weight_file", "model.safetensors"))
        self.rejects(lambda p, q: q["models"]["sprocket_instruct"].__setitem__(
            "weight_declared_lfs_sha256_not_downloaded", m.SOURCES["sprocket_chat"][2]))

    def test_refuse_unpreregistered_geometry_and_tolerance(self):
        self.rejects(lambda p, q: p["architecture"].__setitem__("nca_steps_per_tap", 1))
        self.rejects(lambda p, q: p["architecture"]["state_geometry"].__setitem__(
            "scalars_per_token", 128))
        self.rejects(lambda p, q: p["match_constraints"].__setitem__(
            "adapter_trainable_param_tolerance_fraction", 0.1))

    def test_refuse_semantically_equivalent_source_git_blob_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in m.BLOBS:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            self.assertEqual(set(m.audit(root)["lanes"]), set(m.SOURCES))
            src = root / "scripts/ncr_001_core.py"
            src.write_bytes(src.read_bytes() + b"\n# harmless-looking change")
            with self.assertRaisesRegex(ValueError, "Git blob"):
                m.audit(root)

    def test_refuse_invalid_mac_parameters(self):
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            m.mac_proxy(0)
        with self.assertRaisesRegex(ValueError, "REFUSED"):
            m.mac_proxy(1280, steps=True)

    def test_refuse_unregistered_mac_proxy_geometry_and_budget(self):
        # The old helper accepted these inputs yet returned parameter counts
        # hardcoded for C=4, 8x8 and bottleneck=256: false #73 fairness.
        for changes in (
            {"hidden": 576}, {"hidden": 640},
            {"taps": 2}, {"taps": 4},
            {"steps": 2}, {"steps": 4},
            {"grid": 4}, {"grid": 16},
            {"channels": 2}, {"channels": 8},
            {"bottleneck": 128}, {"bottleneck": 512},
        ):
            hidden = changes.get("hidden", 1280)
            kwargs = {k: v for k, v in changes.items() if k != "hidden"}
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError,
                                            "unregistered Sprocket adapter geometry"):
                    m.mac_proxy(hidden, **kwargs)

    def test_module_shape_parameter_arithmetic_is_independent_of_steps(self):
        # Re-derive from pinned PyTorch definitions, including both Conv2d
        # biases, observation bias, LN affine and per-feature MLP gate.
        h, field, c, b = 1280, 8 * 8 * 4, 4, 256
        nca = (2 * h + h * field + field + 4 * c * c * 9 + 4 * c
               + 2 * c * 4 * c + 2 * c + field * h)
        mlp = 2 * h + h * b + b + b + b * h
        self.assertEqual(nca, 658904)
        self.assertEqual(mlp, 658432)
        for steps in (1, 3):
            with self.subTest(steps=steps):
                r = m.mac_proxy(h, steps=steps)
                self.assertEqual(r["nca_trainable_parameter_formula"], nca)
                self.assertEqual(r["mlp_trainable_parameter_formula"], mlp)
                self.assertEqual(r["nca_trainable_parameter_formula"] -
                                 r["mlp_trainable_parameter_formula"], 472)


if __name__ == "__main__":
    unittest.main()
